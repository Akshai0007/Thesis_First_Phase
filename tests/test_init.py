"""Directly verifies the professor's instruction: weights must come from a
real distribution, never a repeated constant. Runs on the actual
QwenForCausalLM model, not a toy nn.Module, so it also exercises whether
init.py correctly recognizes every real module type in this architecture
(Linear, Embedding, RMSNorm) -- missing one silently would be a real bug.
"""
import torch
from qwenlab.config import ModelConfig, InitConfig
from qwenlab.models.qwen import QwenForCausalLM
from qwenlab.init import apply_init


def _small_model():
    cfg = ModelConfig(vocab_size=1000, hidden_size=64, intermediate_size=176,
                       num_layers=4, num_heads=8, num_kv_heads=2,
                       qkv_bias=False, tie_embeddings=True)
    return QwenForCausalLM(cfg), cfg


def test_no_weight_bearing_tensor_is_constant_filled():
    """The actual instruction: never repeat a single value (e.g. 0.5) across
    a whole tensor. Norm-layer weights are the one correct exception --
    initializing a per-channel scale factor to a constant (1.0) is standard
    practice everywhere (real Qwen/LLaMA/GPT all do this), not a
    symmetry-breaking risk, since there's no redundant-neuron concern for a
    scale parameter."""
    model, _ = _small_model()
    apply_init(model, InitConfig(scheme="gpt2_scaled", std=0.02, seed=1))
    offenders = [n for n, p in model.named_parameters()
                 if p.numel() > 1 and "norm" not in n and p.std().item() < 1e-6]
    assert offenders == [], f"found constant-filled non-norm tensor(s): {offenders}"


def test_norm_weights_are_correctly_left_at_ones():
    model, _ = _small_model()
    apply_init(model, InitConfig(scheme="gpt2_scaled", std=0.02, seed=1))
    for n, p in model.named_parameters():
        if "norm" in n:
            assert torch.equal(p, torch.ones_like(p)), f"{n} should be all-ones after init"


def test_gpt2_scaled_depth_scales_residual_output_projections_only():
    model, cfg = _small_model()
    apply_init(model, InitConfig(scheme="gpt2_scaled", std=0.02, seed=1))
    expected_residual_std = 0.02 / (2 * cfg.num_layers) ** 0.5

    o_proj_std = model.layers[0].self_attn.o_proj.weight.std().item()
    down_proj_std = model.layers[0].mlp.down_proj.weight.std().item()
    q_proj_std = model.layers[0].self_attn.q_proj.weight.std().item()

    assert abs(o_proj_std - expected_residual_std) < 0.003
    assert abs(down_proj_std - expected_residual_std) < 0.003
    assert abs(q_proj_std - 0.02) < 0.005
    assert o_proj_std < q_proj_std, "residual-output projection should have smaller std than a plain one"


def test_normal_scheme_does_not_depth_scale():
    """The plain 'normal' scheme should NOT apply the residual-output
    scaling -- that distinguishes it from 'gpt2_scaled' and is what makes it
    the deliberately naive baseline to compare against."""
    model, _ = _small_model()
    apply_init(model, InitConfig(scheme="normal", std=0.02, seed=1))
    o_proj_std = model.layers[0].self_attn.o_proj.weight.std().item()
    q_proj_std = model.layers[0].self_attn.q_proj.weight.std().item()
    assert abs(o_proj_std - q_proj_std) < 0.005, "plain 'normal' scheme should treat all linears the same"


def test_same_seed_is_reproducible_different_seed_is_not():
    m1, _ = _small_model()
    m2, _ = _small_model()
    m3, _ = _small_model()
    apply_init(m1, InitConfig(scheme="gpt2_scaled", std=0.02, seed=42))
    apply_init(m2, InitConfig(scheme="gpt2_scaled", std=0.02, seed=42))
    apply_init(m3, InitConfig(scheme="gpt2_scaled", std=0.02, seed=43))

    w1 = m1.layers[0].self_attn.q_proj.weight
    w2 = m2.layers[0].self_attn.q_proj.weight
    w3 = m3.layers[0].self_attn.q_proj.weight
    assert torch.equal(w1, w2), "same seed must give identical init (reproducibility)"
    assert not torch.equal(w1, w3), "different seed must give different init"


def test_truncated_init_actually_bounds_the_tail():
    model, _ = _small_model()
    apply_init(model, InitConfig(scheme="normal", std=0.02, truncate_sigma=2.0, seed=5))
    w = model.embed_tokens.weight
    assert w.abs().max().item() <= 2.0 * 0.02 + 1e-6, "truncated init exceeded its stated sigma bound"
