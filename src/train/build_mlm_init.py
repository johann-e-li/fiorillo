"""The MLM start for session 5's reader (docs/prompt_session5.md, READER METHOD).

Builds a transformers ModernBertForMaskedLM with ModernBERT-large's configuration: every encoder tensor comes from Laya
(Laya's encoder.<name> becomes model.<name>, cast to float32); the MLM head (head.dense.weight, head.norm.weight and
decoder.bias) comes from ModernBERT-large, cast to float32; the decoder weight is tied to the token embeddings. It
stops with an error unless ModernBERT-large's config equals Laya's encoder config in every architecture field (hidden
size, layers, attention heads, intermediate size, vocabulary size, maximum positions, local attention window, global
attention interval, layer types and rotary settings), each model.<name> tensor (170 for ModernBERT-large) is filled
from Laya exactly once with an identical shape, and the two tokenizer vocabularies are identical, compared as content:
the BPE model's vocabulary and merges, with every merge read as a pair whatever its JSON form (older tokenizer.json files
write a merge as "a b", newer ones as ["a", "b"]), and the added tokens. The saved config turns on sparse prediction (the MLM loss and logits are computed only
at masked positions; it changes no weights) and records the attention implementation sdpa.

Writes <out>/: config.json and model.safetensors (save_pretrained), Laya's tokenizer.json and tokenizer_config.json,
and init_report.json (the checks, the tensor counts and both sources).

Usage: python src/train/build_mlm_init.py --laya-dir /root/eddm_s5/models/laya \
    --modernbert-dir /root/eddm_s5/models/modernbert-large --out /root/eddm_s5/models/mlm_init \
    [--laya-revision <sha> --modernbert-revision <sha>]
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ARCH_FIELDS = ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size",
               "max_position_embeddings", "local_attention", "global_attn_every_n_layers")
HEAD_TENSORS = ("head.dense.weight", "head.norm.weight", "decoder.bias")


def vocabulary_content(tokenizer_json):
    """The vocabulary as content: the BPE model's fields with every merge as a pair (JSON form aside), and the added
    tokens. ModernBERT-large's tokenizer.json writes merges as "a b" strings and Laya's as ["a", "b"] pairs."""
    data = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    model = dict(data["model"])
    model["merges"] = [list(m) if isinstance(m, (list, tuple)) else m.split(" ", 1) for m in model.get("merges", [])]
    return {"model": model, "added_tokens": data.get("added_tokens", [])}


def vocabulary_digest(tokenizer_json):
    return hashlib.sha256(json.dumps(vocabulary_content(tokenizer_json), sort_keys=True).encode("utf-8")).hexdigest()


def rotary(cfg):
    """(global theta, local theta): from rope_parameters (transformers 5) or global_rope_theta / local_rope_theta."""
    rope = getattr(cfg, "rope_parameters", None)
    if isinstance(rope, dict) and "full_attention" in rope and "sliding_attention" in rope:
        return float(rope["full_attention"]["rope_theta"]), float(rope["sliding_attention"]["rope_theta"])
    return float(getattr(cfg, "global_rope_theta")), float(getattr(cfg, "local_rope_theta"))


def compare_configs(modernbert_cfg, laya_cfg):
    """The architecture fields that differ, as (field, ModernBERT-large value, Laya value)."""
    diffs = [(f, getattr(modernbert_cfg, f, None), getattr(laya_cfg, f, None)) for f in ARCH_FIELDS
             if getattr(modernbert_cfg, f, None) != getattr(laya_cfg, f, None)]
    if list(getattr(modernbert_cfg, "layer_types", []) or []) != list(getattr(laya_cfg, "layer_types", []) or []):
        diffs.append(("layer_types", getattr(modernbert_cfg, "layer_types", None), getattr(laya_cfg, "layer_types", None)))
    if rotary(modernbert_cfg) != rotary(laya_cfg):
        diffs.append(("rotary (global, local theta)", rotary(modernbert_cfg), rotary(laya_cfg)))
    return diffs


def build(laya_dir, modernbert_dir, out_dir, laya_revision="", modernbert_revision=""):
    from transformers import AutoModelForMaskedLM, ModernBertConfig

    laya_dir, modernbert_dir, out_dir = Path(laya_dir), Path(modernbert_dir), Path(out_dir)
    config = ModernBertConfig.from_pretrained(str(modernbert_dir))
    laya_encoder = ModernBertConfig.from_pretrained(str(laya_dir / "encoder"))
    diffs = compare_configs(config, laya_encoder)
    if diffs:
        raise ValueError(f"ModernBERT-large's config differs from Laya's encoder config: {diffs}")
    laya_vocab = vocabulary_digest(laya_dir / "tokenizer" / "tokenizer.json")
    mb_vocab = vocabulary_digest(modernbert_dir / "tokenizer.json")
    if laya_vocab != mb_vocab:
        raise ValueError(f"tokenizer vocabularies differ: Laya {laya_vocab[:16]}, ModernBERT-large {mb_vocab[:16]}")

    config.sparse_prediction = True
    config.sparse_pred_ignore_index = -100
    config.tie_word_embeddings = True
    model = AutoModelForMaskedLM.from_config(config, attn_implementation="sdpa")
    targets = [k for k in model.state_dict() if k.startswith("model.")]
    with safe_open(str(laya_dir / "model.safetensors"), framework="pt") as laya:
        source = {"model." + k[len("encoder."):]: k for k in laya.keys() if k.startswith("encoder.")}
        missing, extra = sorted(set(targets) - set(source)), sorted(set(source) - set(targets))
        if missing or extra:
            raise ValueError(f"encoder tensors do not match: missing from Laya {missing[:5]}, extra in Laya {extra[:5]}")
        state, shapes = {}, model.state_dict()
        for key in targets:
            tensor = laya.get_tensor(source[key]).to(torch.float32)
            if tuple(tensor.shape) != tuple(shapes[key].shape):
                raise ValueError(f"{key}: Laya shape {tuple(tensor.shape)} differs from {tuple(shapes[key].shape)}")
            state[key] = tensor
    with safe_open(str(modernbert_dir / "model.safetensors"), framework="pt") as mb:
        for key in HEAD_TENSORS:
            tensor = mb.get_tensor(key).to(torch.float32)
            if tuple(tensor.shape) != tuple(shapes[key].shape):
                raise ValueError(f"{key}: ModernBERT-large shape {tuple(tensor.shape)} differs from {tuple(shapes[key].shape)}")
            state[key] = tensor
    result = model.load_state_dict(state, strict=False)
    if set(result.missing_keys) - {"decoder.weight"} or result.unexpected_keys:
        raise ValueError(f"unexpected load result: missing {result.missing_keys}, unexpected {result.unexpected_keys}")
    model.tie_weights()
    if model.decoder.weight.data_ptr() != model.model.embeddings.tok_embeddings.weight.data_ptr():
        raise ValueError("the decoder weight is not tied to the token embeddings")
    loaded = model.state_dict()
    for key in targets:
        if not torch.equal(loaded[key], state[key]):
            raise ValueError(f"{key} did not load exactly")
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir), safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(laya_dir / "tokenizer" / name, out_dir / name)
    report = {
        "encoder_tensors_from_laya": len(targets), "head_tensors_from_modernbert": list(HEAD_TENSORS),
        "decoder_weight": "tied to model.embeddings.tok_embeddings.weight",
        "architecture_fields_checked": list(ARCH_FIELDS) + ["layer_types", "rotary"],
        "rotary_global_local_theta": list(rotary(config)), "vocabulary_digest": laya_vocab,
        "laya": str(laya_revision or laya_dir), "modernbert_large": str(modernbert_revision or modernbert_dir),
        "sparse_prediction": True, "attention": "sdpa", "dtype": "float32",
    }
    (out_dir / "init_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--laya-dir", type=Path, required=True, help="Laya-format folder: model.safetensors, encoder/, tokenizer/")
    parser.add_argument("--modernbert-dir", type=Path, required=True, help="ModernBERT-large: config.json, model.safetensors, tokenizer.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--laya-revision", default="")
    parser.add_argument("--modernbert-revision", default="")
    args = parser.parse_args()
    report = build(args.laya_dir, args.modernbert_dir, args.out, args.laya_revision, args.modernbert_revision)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    sys.exit(main())
