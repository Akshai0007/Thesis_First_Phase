"""Weight initialization schemes.

Directly addresses the requirement: initialize from a proper random
DISTRIBUTION, never a repeated constant. Filling a layer with a single
repeated value (e.g. 0.5 everywhere) breaks symmetry-breaking: every unit in
that layer computes the exact same function of the input, receives the exact
same gradient, and stays identical forever -- the layer degenerates to
functionally one neuron no matter how many parameters it has. Every scheme
below draws each weight independently from a distribution instead.

Two schemes, matching what real GPT-style / Qwen-style models actually do:
  - "normal":       every weight ~ Normal(0, std). The naive-but-valid baseline.
  - "gpt2_scaled":  same, EXCEPT weights that feed directly into a residual
                    stream (attention output proj, MLP down proj) are scaled
                    by 1/sqrt(2 * num_layers). Without this, residual variance
                    grows with depth and deep models become unstable early in
                    training -- this is the actual technique GPT-2/GPT-3 and
                    most modern LLMs use (confirmed against nanoGPT's and
                    OLMo's public init code before writing this).
Both support optional truncation (redraw any sample beyond `truncate_sigma`
standard deviations) -- HF's own from-scratch init utilities do this too, to
avoid rare huge activations from the normal distribution's tails.
"""
import math
import torch
import torch.nn as nn

from .registry import INIT_SCHEMES


def _draw(shape, std: float, truncate_sigma: float, generator: torch.Generator) -> torch.Tensor:
    if truncate_sigma and truncate_sigma > 0:
        t = torch.empty(shape)
        nn.init.trunc_normal_(t, mean=0.0, std=std, a=-truncate_sigma * std, b=truncate_sigma * std, generator=generator)
        return t
    return torch.empty(shape).normal_(mean=0.0, std=std, generator=generator)


def _is_residual_output(name: str) -> bool:
    """True for the two projection types that write directly back into the
    residual stream: attention's output projection, and the MLP's down
    projection. These get the depth-scaled std; everything else gets the
    plain std."""
    return name.endswith("attn.o_proj.weight") or name.endswith("mlp.down_proj.weight")


@INIT_SCHEMES.register("normal")
def init_normal(model: nn.Module, std: float = 0.02, truncate_sigma: float = 0.0, seed: int = 1234, **_):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.copy_(_draw(module.weight.shape, std, truncate_sigma, g))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.copy_(_draw(module.weight.shape, std, truncate_sigma, g))
            elif hasattr(module, "weight") and getattr(module, "_is_norm", False):
                module.weight.fill_(1.0)
    return model


@INIT_SCHEMES.register("gpt2_scaled")
def init_gpt2_scaled(model: nn.Module, std: float = 0.02, truncate_sigma: float = 0.0,
                      seed: int = 1234, num_layers: int = 1, **_):
    g = torch.Generator().manual_seed(seed)
    residual_std = std / math.sqrt(2 * max(num_layers, 1))
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                full_name = name + ".weight"
                this_std = residual_std if _is_residual_output(full_name) else std
                module.weight.copy_(_draw(module.weight.shape, this_std, truncate_sigma, g))
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.copy_(_draw(module.weight.shape, std, truncate_sigma, g))
            elif hasattr(module, "weight") and getattr(module, "_is_norm", False):
                module.weight.fill_(1.0)
    return model


def apply_init(model: nn.Module, init_cfg) -> nn.Module:
    fn = INIT_SCHEMES.get(init_cfg.scheme)
    return fn(model, std=init_cfg.std, truncate_sigma=init_cfg.truncate_sigma,
              seed=init_cfg.seed, num_layers=getattr(model, "num_layers", 1))
