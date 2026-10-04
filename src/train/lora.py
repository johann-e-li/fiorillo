"""Low-rank adaptation (LoRA) for torch.nn.Linear layers (docs/prompt_session7.md, ARM B).

Written in this project, with no new package. apply_lora() replaces each targeted torch.nn.Linear inside a model with
a LoRALinear that keeps the original layer, frozen, and adds a trainable low-rank update:

    y = base(x) + scale * B(A(dropout(x))),   scale = alpha / r

A is initialized as torch.nn.Linear initializes its weight (Kaiming uniform), B with zeros, so at the start the
adapted model computes exactly what the base computes. The adapters are float32 whatever the base's precision; under
autocast their matrix products run in the autocast type. Adapter parameters are named lora_A and lora_B, which is how
the trainer finds them (their own learning rate) and how lora_state_dict() saves them.

Session 7's settings: rank 16, alpha 32 (scale 2), dropout 0.05, on q_proj, k_proj, v_proj, o_proj, gate_proj,
up_proj and down_proj of every decoder layer (TARGETS).
"""
import torch
import torch.nn as nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=32, dropout=0.05):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear wraps torch.nn.Linear, not {type(base).__name__}")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.alpha, self.scale = int(r), float(alpha), float(alpha) / float(r)
        self.lora_A = nn.Linear(base.in_features, self.r, bias=False, dtype=torch.float32, device=base.weight.device)
        self.lora_B = nn.Linear(self.r, base.out_features, bias=False, dtype=torch.float32, device=base.weight.device)
        nn.init.zeros_(self.lora_B.weight)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        out = self.base(x)
        z = self.dropout(x)
        if not torch.is_autocast_enabled(x.device.type) and z.dtype != self.lora_A.weight.dtype:
            z = z.to(self.lora_A.weight.dtype)
        update = self.lora_B(self.lora_A(z)) * self.scale
        return out + update.to(out.dtype)

    def extra_repr(self):
        return f"r={self.r}, alpha={self.alpha}, scale={self.scale}"


def apply_lora(model, targets=TARGETS, r=16, alpha=32, dropout=0.05):
    """Wrap every torch.nn.Linear whose attribute name is in targets; returns the number wrapped. Freezes nothing else:
    the caller freezes the base first."""
    wrapped = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                wrapped += 1
    if wrapped == 0:
        raise ValueError(f"no torch.nn.Linear named {targets} was found")
    return wrapped


def is_adapter(name):
    return ".lora_A." in name or ".lora_B." in name or name.startswith(("lora_A.", "lora_B."))


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if is_adapter(n)]


def count_parameters(params):
    return int(sum(p.numel() for p in params))
