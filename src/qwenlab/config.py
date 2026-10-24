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
    vocab_pad_multiple: int = 128         # used when auto_vocab is True
    auto_vocab: bool = True               # True: vocab_size = len(tokenizer) rounded up to vocab_pad_multiple (new language).
                                          # False: use vocab_size exactly as written (e.g. 151936 to match official Qwen2.5 / earlier checkpoints)

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
    """Next-word-prediction data: continuous text, packed into windows of `seq_len`+1 tokens.
    seq_len is THE training context window: generation is limited to it (inference/generate.py)."""
    source: SourceConfig = field(default_factory=SourceConfig)
    cache_dir: str = "data_cache/pretrain"
    seq_len: int = 512
    max_train_tokens: Optional[int] = None    # required for streaming hf_dataset sources
    val_tokens: int = 1_000_000               # at most this many held-out validation tokens ...
    max_val_fraction: float = 0.1             # ... and never more than this share of the text (small datasets still work)
    min_chars: int = 200                      # shorter documents are dropped
    drop_qa_format: bool = True               # drop documents with chat/FIM markers or Question:/Answer: layout
    normalizer: str = "default"               # text cleaning, by name: "default" | "malayalam" (data/normalizers.py)
    script: str = ""                          # optional script filter, e.g. "malayalam"
    min_script_ratio: float = 0.0             # drop documents with a smaller share of letters in `script`


@dataclass
class TaskMixEntry:
    task: str = "causal_lm"               # key in TASKS ("causal_lm" = packed pretraining stream)
    weight: float = 1.0
    source: SourceConfig = field(default_factory=SourceConfig)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainConfig:
    output_dir: str = "runs/exp"
    stage: str = "pretrain"               # only "pretrain" (continuous text) is wired into scripts/train.py
    max_steps: int = 1000                 # the PLANNED length: the learning-rate schedule is built from this
    schedule_start_step: int = 0          # >0: the schedule restarts here (new warmup/stable/decay over the steps that remain),
                                          # e.g. continuing a run that already decayed, without a jump back to the old peak
    stop_at_step: Optional[int] = None    # pause cleanly (save + eval) at this step; the schedule is unchanged. null = run to max_steps
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
    time_budget_minutes: Optional[float] = None   # stop cleanly before a session limit (Kaggle)
    hub_repo_id: Optional[str] = None     # e.g. "user/my-run": back up / restore checkpoints on the Hugging Face Hub (needs HF_TOKEN)
    hub_upload_every_n_saves: int = 4     # full checkpoints are ~6 GB each: upload rarely (and always at the end)
    export_hf: bool = False               # at the end, also write hf_model/ (Hugging Face Qwen2 format, safetensors)
    seed: int = 42


@dataclass
class EvalConfig:
    """Held-out evaluation: cross-entropy loss AND perplexity on the validation windows."""
    eval_at_start: bool = True            # also score the untrained model (loss ~ ln(vocab), perplexity ~ vocab)
    batch_size: int = 8                   # validation micro-batch size


@dataclass
class GenerationConfig:
    """Guard-rails used by scripts/generate.py and the training-time sample probes."""
    max_new_tokens: int = 48
    max_new_tokens_cap: int = 64          # hard ceiling whatever the caller asks for (~2 sentences)
    temperature: float = 0.0              # 0 = greedy
    top_k: Optional[int] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    max_sentences: Optional[int] = 2
    probe_prompts: List[str] = field(default_factory=lambda: ["The capital of France is", "Once upon a time"])


@dataclass
class ExperimentConfig:
    run_name: str = "exp"
    model: ModelConfig = field(default_factory=ModelConfig)
    init: InitConfig = field(default_factory=InitConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: PretrainDataConfig = field(default_factory=PretrainDataConfig)
    tasks: List[TaskMixEntry] = field(default_factory=list)   # reserved for the later SFT stage (not used by scripts/train.py)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


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
            raw = json.load(f)
        merged = _deep_merge(merged, {k: v for k, v in raw.items() if not k.startswith("_")})   # "_note" keys = comments
    apply_overrides(merged, overrides or [])
    return from_dict(ExperimentConfig, merged)


def to_dict(cfg) -> Dict[str, Any]:
    d = asdict(cfg)
    if isinstance(d.get("model"), dict):
        d["model"].pop("head_dim", None)       # derived from hidden_size // num_heads, not a settable field
    return d


def save_config(cfg, path: str):
    with open(path, "w") as f:
        json.dump(to_dict(cfg), f, indent=2)
