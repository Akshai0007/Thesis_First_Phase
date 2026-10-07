"""Typed, validated configuration. JSON in, dataclasses out, `--set a.b.c=value` overrides.
Unknown keys raise immediately (typos never silently fall back to defaults)."""
import json
import typing
from dataclasses import dataclass, field, fields, is_dataclass, asdict
from typing import Any, Dict, List, Optional


# ----------------------------------------------------------------------------- model
#
# Simplified to Qwen2.5-style only (Qwen3's QK-norm and decoupled head_dim
# removed on purpose -- this project targets Qwen2.5-0.5B, and that extra
# generality existed solely to support Qwen3, which is no longer in scope).
@dataclass
class ModelConfig:
    name: str = "qwen"                    # key in MODELS registry
    vocab_size: int = 151936
    hidden_size: int = 896
    intermediate_size: int = 4864
    num_layers: int = 24
    num_heads: int = 14
    num_kv_heads: int = 2
    head_dim: int = field(init=False, default=0)   # always hidden_size // num_heads -- computed
                                           # in __post_init__ (default=0 is a placeholder,
                                           # immediately overwritten; not settable via the
                                           # constructor). Previously overridable for Qwen3's
                                           # decoupled head_dim; no longer needed.
    max_seq_len: int = 32768
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    qkv_bias: bool = True                 # real Qwen2/2.5 hardcodes this True regardless of any
                                           # config field (verified from the actual HF source) --
                                           # kept as a field here mainly for clarity/documentation.
    tie_embeddings: bool = True           # True at this size (0.5B); larger Qwen2.5 sizes untie.
    gradient_checkpointing: bool = False
    loss_chunk_tokens: int = 4096         # chunked LM-head loss (151k vocab logits are huge); 0 = off
    z_loss_coef: float = 0.0

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        self.head_dim = self.hidden_size // self.num_heads
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be a multiple of num_kv_heads"

    def num_params(self) -> int:
        """Analytical parameter count (no model instantiation needed)."""
        H, I, V, hd = self.hidden_size, self.intermediate_size, self.vocab_size, self.head_dim
        q, kv = self.num_heads * hd, self.num_kv_heads * hd
        attn = H * q + H * kv * 2 + q * H
        if self.qkv_bias:
            attn += q + 2 * kv
        per_layer = attn + 3 * H * I + 2 * H
        total = V * H + self.num_layers * per_layer + H
        if not self.tie_embeddings:
            total += V * H
        return total


# Real Qwen2.5-0.5B dimensions. head_dim is no longer a settable field (see
# above), so it's omitted here -- __post_init__ derives it as 896/14=64,
# matching the real model. Hand-verified twice during review: 494,032,768
# total params, matching the published figure exactly.
QWEN2_5_0_5B = ModelConfig(
    vocab_size=151936, hidden_size=896, intermediate_size=4864,
    num_layers=24, num_heads=14, num_kv_heads=2,
    max_seq_len=32768, rope_theta=1_000_000.0, rms_norm_eps=1e-6,
    qkv_bias=True, tie_embeddings=True,
)


@dataclass
class InitConfig:
    scheme: str = "gpt2_scaled"           # key in INIT_SCHEMES
    std: float = 0.02
    truncate_sigma: float = 0.0           # >0 -> truncated normal at +-k sigma
    seed: int = 1234


@dataclass
class TokenizerConfig:
    name: str = "hf"                      # key in TOKENIZERS
    path: str = "Qwen/Qwen2.5-0.5B"       # HF id or local dir
    eos_token: str = "<|endoftext|>"
    pad_token: Optional[str] = None       # None -> eos
    im_start: str = "<|im_start|>"
    im_end: str = "<|im_end|>"
    fim_prefix: str = "<|fim_prefix|>"
    fim_middle: str = "<|fim_middle|>"
    fim_suffix: str = "<|fim_suffix|>"


@dataclass
class SourceConfig:
    """One data source. `kind` is a key in SOURCES; `params` are passed to it."""
    kind: str = "hf_dataset"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PretrainDataConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    cache_dir: str = "data_cache/pretrain"
    seq_len: int = 1024
    max_train_tokens: Optional[int] = None
    val_tokens: int = 2_000_000
    shard_tokens: int = 100_000_000
    fim_rate: float = 0.0                 # fraction of windows turned into fill-in-the-middle samples
    fim_spm_rate: float = 0.5             # of those, fraction using SPM ordering (rest PSM)


@dataclass
class TaskMixEntry:
    task: str = "causal_lm"               # key in TASKS ("causal_lm" = packed pretraining stream)
    weight: float = 1.0
    source: SourceConfig = field(default_factory=SourceConfig)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    output_dir: str = "runs/exp"
    stage: str = "pretrain"               # "pretrain" (packed stream) | "sft" (task mixture)
    max_steps: int = 1000
    micro_batch_size: int = 8
    grad_accum_steps: int = 1
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    schedule: str = "cosine"              # key in SCHEDULES: cosine | wsd | linear | constant
    warmup_steps: int = 100
    decay_fraction: float = 0.1           # WSD: last fraction of steps used for decay
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    no_decay_embeddings: bool = True
    precision: str = "auto"               # auto | bf16 | fp16 | fp32
    compile: bool = False
    log_every: int = 10
    eval_every: int = 200
    eval_batches: int = 20
    save_every: int = 500
    keep_last: int = 2
    resume: bool = True
    init_from: Optional[str] = None       # checkpoint dir to start (weights only) e.g. SFT from a pretrain ckpt
    time_budget_minutes: Optional[float] = None   # stop cleanly before a session limit (Kaggle)
    peak_tflops: float = 0.0              # for MFU logging; 0 disables
    seed: int = 42


@dataclass
class EvalConfig:
    mc_tasks: List[str] = field(default_factory=lambda: ["hellaswag", "arc_easy", "arc_challenge", "piqa"])
    mc_limit: Optional[int] = 500
    ppl_sets: List[str] = field(default_factory=lambda: ["wikitext103"])
    gen_tasks: List[str] = field(default_factory=list)
    gen_limit: int = 100
    reference: str = "Qwen/Qwen2.5-0.5B"
    batch_size: int = 16
    max_seq_len: int = 1024


@dataclass
class ExperimentConfig:
    run_name: str = "exp"
    model: ModelConfig = field(default_factory=ModelConfig)
    init: InitConfig = field(default_factory=InitConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: PretrainDataConfig = field(default_factory=PretrainDataConfig)
    tasks: List[TaskMixEntry] = field(default_factory=list)   # used when train.stage == "sft"
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# ----------------------------------------------------------------------------- (de)serialisation
def _unwrap_optional(tp):
    if typing.get_origin(tp) is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return tp


def from_dict(cls, d: Dict[str, Any]):
    if not is_dataclass(cls):
        return d
    hints = typing.get_type_hints(cls)
    valid = {f.name for f in fields(cls)}
    unknown = set(d) - valid
    if unknown:
        raise ValueError(f"Unknown config key(s) {sorted(unknown)} for {cls.__name__}. Valid: {sorted(valid)}")
    kwargs = {}
    for k, v in d.items():
        tp = _unwrap_optional(hints[k])
        if is_dataclass(tp) and isinstance(v, dict):
            kwargs[k] = from_dict(tp, v)
        elif typing.get_origin(tp) is list and v is not None:
            (inner,) = typing.get_args(tp)
            kwargs[k] = [from_dict(inner, x) if is_dataclass(inner) and isinstance(x, dict) else x for x in v]
        else:
            kwargs[k] = v
    return cls(**kwargs)


def _set_path(d: Any, parts: List[str], value: Any):
    key = parts[0]
    if isinstance(d, list):
        key = int(key)
    if len(parts) == 1:
        d[key] = value
        return
    if isinstance(d, dict) and key not in d:
        d[key] = {}
    _set_path(d[key], parts[1:], value)


def apply_overrides(d: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like a.b.c=value")
        path, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        _set_path(d, path.split("."), value)
    return d


def _deep_merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(paths, overrides: Optional[List[str]] = None) -> ExperimentConfig:
    """`paths` may be one file or a list; later files override earlier ones (deep-merge).
    Typical use: [model_spec.json, experiment.json] -> model spec is reusable across experiments."""
    if isinstance(paths, str):
        paths = [paths]
    merged: Dict[str, Any] = {}
    for p in paths:
        with open(p) as f:
            merged = _deep_merge(merged, json.load(f))
    apply_overrides(merged, overrides or [])
    return from_dict(ExperimentConfig, merged)


def to_dict(cfg) -> Dict[str, Any]:
    return asdict(cfg)


def save_config(cfg, path: str):
    with open(path, "w") as f:
        json.dump(to_dict(cfg), f, indent=2)
