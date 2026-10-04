"""Build a Laya-format model folder whose reader (encoder) comes from another ModernBERT-large checkpoint
(docs/prompt_session4.md, step 6c). Used for arm B: BioClinical ModernBERT-large's reader with Laya's decision head.

Every Laya tensor named encoder.<name> is replaced by the source checkpoint's model.<name>, cast to the dtype Laya
stores it in; every other Laya tensor (head, scorer, type embedding, act head, temperature buffer) is kept. The
source's masked-language-model head (head.*, decoder.*) is not used. The script stops with an error unless every
Laya encoder tensor is replaced exactly once with an identical shape and the two tokenizer vocabularies are
identical. Laya's encoder config, tokenizer and rl_agent_config.json are copied; REVISION.txt records both sources.

Usage: python src/train/assemble_reader.py --laya-dir models/laya-base --reader-dir /path/to/BioClinical-ModernBERT-large \
    --out models/public_v0/arm_bioclinical_init --laya-revision <sha> --reader-revision <sha>
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def vocabulary_digest(tokenizer_json):
    model = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))["model"]
    return hashlib.sha256(json.dumps(model, sort_keys=True).encode("utf-8")).hexdigest()


def assemble(laya_dir, reader_dir, out_dir, laya_revision="", reader_revision=""):
    laya_dir, reader_dir, out_dir = Path(laya_dir), Path(reader_dir), Path(out_dir)
    laya_vocab = vocabulary_digest(laya_dir / "tokenizer" / "tokenizer.json")
    reader_vocab = vocabulary_digest(reader_dir / "tokenizer.json")
    if laya_vocab != reader_vocab:
        raise ValueError(f"tokenizer vocabularies differ: Laya {laya_vocab[:16]}, reader {reader_vocab[:16]}")
    tensors = {}
    with safe_open(str(laya_dir / "model.safetensors"), framework="pt") as laya, \
            safe_open(str(reader_dir / "model.safetensors"), framework="pt") as reader:
        laya_keys = list(laya.keys())
        encoder_keys = [k for k in laya_keys if k.startswith("encoder.")]
        source = {"encoder." + k[len("model."):]: k for k in reader.keys() if k.startswith("model.")}
        missing, extra = sorted(set(encoder_keys) - set(source)), sorted(set(source) - set(encoder_keys))
        if missing or extra:
            raise ValueError(f"encoder tensors do not match: missing from reader {missing[:5]}, extra in reader {extra[:5]}")
        for key in laya_keys:
            kept = laya.get_tensor(key)
            if key in source:
                new = reader.get_tensor(source[key])
                if tuple(new.shape) != tuple(kept.shape):
                    raise ValueError(f"{key}: reader shape {tuple(new.shape)} differs from Laya {tuple(kept.shape)}")
                tensors[key] = new.to(kept.dtype).contiguous()
            else:
                tensors[key] = kept.contiguous()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_dir / "model.safetensors"))
    for folder in ("encoder", "tokenizer"):
        if (out_dir / folder).exists():
            shutil.rmtree(out_dir / folder)
        shutil.copytree(laya_dir / folder, out_dir / folder)
    config = json.loads((laya_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    config["model_name"] = out_dir.name
    config["encoder_source"] = str(reader_revision or reader_dir)
    (out_dir / "rl_agent_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out_dir / "REVISION.txt").write_text(
        f"decision head and configs: Laya {laya_revision or laya_dir}\nreader (encoder): {reader_revision or reader_dir}\n"
        f"encoder tensors replaced: {len(encoder_keys)}\n", encoding="utf-8")
    return len(encoder_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--laya-dir", type=Path, required=True)
    parser.add_argument("--reader-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--laya-revision", default="")
    parser.add_argument("--reader-revision", default="")
    args = parser.parse_args()
    n = assemble(args.laya_dir, args.reader_dir, args.out, args.laya_revision, args.reader_revision)
    print(f"wrote {args.out}: {n} encoder tensors replaced")


if __name__ == "__main__":
    sys.exit(main())
