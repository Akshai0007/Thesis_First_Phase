"""Architecture presets for the generic Hugging Face model (`model.name = "hf"`, `model.arch = "<preset>"`).

Design: the SHARED shape fields of ModelConfig (hidden_size, intermediate_size, num_layers, num_heads, num_kv_heads,
rope_theta, rms_norm_eps, tie_embeddings, max_seq_len, vocab_size) are applied to every architecture, so a fair
comparison holds width/depth/heads/RoPE-base constant and changes ONLY the block design. Everything that is specific to
one architecture (QK-norm, NoPE layers, sliding windows, ...) lives in its preset `extra` dict, and a run can override or
add fields with `model.hf_config` in the config. Nothing here is loaded unless `name == "hf"`.

status: "reference"  = matches the hand-written `qwen` model (used for parity / speed comparison)
        "unverified" = field names written from memory for the transformers versions around 4.5x; `scripts/tournament.py --phase check`
                       builds every preset and fails loudly (with the transformers version) if a field is wrong or silently ignored
        "experimental" = hybrid / state-space models: need extra CUDA kernels (flash-linear-attention, mamba-ssm, causal-conv1d)
"""
from typing import Any, Dict

ARCH_PRESETS: Dict[str, Dict[str, Any]] = {
    "qwen2": {
        "config_class": "Qwen2Config", "status": "reference",
        "extra": {"use_sliding_window": False},
        "note": "Hugging Face Qwen2 (QKV bias, GQA). Same maths as the hand-written `qwen` model; use it to compare speed/memory of HF vs hand-written code.",
    },
    "qwen3": {
        "config_class": "Qwen3Config", "status": "unverified",
        "extra": {"attention_bias": False},
        "uses_head_dim": True,
        "note": "QK-Norm on query/key, no QKV bias.",
    },
    "llama": {
        "config_class": "LlamaConfig", "status": "unverified",
        "extra": {"attention_bias": False, "mlp_bias": False},
        "note": "Plain Llama block (no QKV bias, no QK-norm).",
    },
    "olmo2": {
        "config_class": "Olmo2Config", "status": "unverified",
        "extra": {"attention_bias": False},
        "note": "OLMo-2: QK-norm over the full projection and post-sublayer norms (stability at higher learning rates).",
    },
    "smollm3": {
        "config_class": "SmolLM3Config", "status": "unverified",
        "extra": {"no_rope_layer_interval": 4},
        "note": "Llama block with NoPE (no rotary) on every 4th layer: better length generalisation.",
    },
    "gemma3": {
        "config_class": "Gemma3TextConfig", "status": "unverified",
        "extra": {"sliding_window": 512, "sliding_window_pattern": 6, "rope_local_base_freq": 10000.0,
                  "hidden_activation": "gelu_pytorch_tanh"},
        "uses_head_dim": True, "scale_query_pre_attn": True,
        "note": "5 local sliding-window layers (window 512) : 1 global layer. Cheap long-context memory; field names differ across transformers versions.",
    },
    "falcon_h1": {
        "hf_config_from": "tiiuae/Falcon-H1-0.5B-Base", "status": "experimental",
        "note": "Hybrid Mamba-2 + attention, taken from the public config (random init, our vocabulary). Shape fields are NOT applied; use hf_config to change them. Needs mamba-ssm / causal-conv1d.",
    },
    "qwen3_5": {
        "hf_config_from": "Qwen/Qwen3.5-0.8B", "status": "experimental",
        "note": "Gated-DeltaNet + attention hybrid from the public config (random init, our vocabulary). Needs very recent transformers and flash-linear-attention.",
    },
}


def effective(hf_cfg, key):
    """Value a transformers config really holds for `key` (newer versions move rope_theta into rope_parameters)."""
    rp = getattr(hf_cfg, "rope_parameters", None)
    if key == "rope_theta" and isinstance(rp, dict) and "rope_theta" in rp:
        return rp["rope_theta"]
    return getattr(hf_cfg, key, None)


def build_hf_config(cfg):
    """ModelConfig -> transformers config for the preset. Raises with a clear message instead of silently dropping fields."""
    import transformers
    if cfg.arch not in ARCH_PRESETS:
        raise KeyError(f"model.arch {cfg.arch!r} is not a preset. Available: {sorted(ARCH_PRESETS)} "
                       f"(or add one to models/arch_presets.py)")
    preset = ARCH_PRESETS[cfg.arch]

    if "hf_config_from" in preset:                                   # hybrids: architecture comes from the public config
        hf_cfg = transformers.AutoConfig.from_pretrained(preset["hf_config_from"])
        hf_cfg = getattr(hf_cfg, "text_config", hf_cfg)
        for k, v in cfg.hf_config.items():
            setattr(hf_cfg, k, v)
        hf_cfg.vocab_size = cfg.vocab_size
        hf_cfg.max_position_embeddings = max(int(getattr(hf_cfg, "max_position_embeddings", 0) or 0), cfg.max_seq_len)
        hf_cfg.use_cache = False
        return hf_cfg

    cls = getattr(transformers, preset["config_class"], None)
    if cls is None:
        raise ImportError(f"{preset['config_class']} does not exist in transformers {transformers.__version__}; "
                          f"upgrade transformers to use model.arch={cfg.arch!r}")
    shape = dict(
        vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size, intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_layers, num_attention_heads=cfg.num_heads, num_key_value_heads=cfg.num_kv_heads,
        max_position_embeddings=cfg.max_seq_len, rms_norm_eps=cfg.rms_norm_eps, rope_theta=cfg.rope_theta,
        tie_word_embeddings=cfg.tie_embeddings,
    )
    if preset.get("uses_head_dim"):
        shape["head_dim"] = cfg.head_dim
    if preset.get("scale_query_pre_attn"):
        shape["query_pre_attn_scalar"] = cfg.head_dim
    kw = {**shape, **preset.get("extra", {}), **cfg.hf_config}
    hf_cfg = cls(**kw)
    hf_cfg.use_cache = False
    for k, v in kw.items():                    # catch renamed / silently ignored fields (version drift)
        got = effective(hf_cfg, k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and got is not None and got != v:
            raise ValueError(f"{cfg.arch}: field {k} was set to {v} but transformers {transformers.__version__} "
                             f"resolved {got}; the field was renamed or ignored in this version.")
    return hf_cfg
