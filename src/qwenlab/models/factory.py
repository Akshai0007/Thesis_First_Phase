"""Build a model from a config + tokenizer. The ONLY place where tokenizer and model meet:
the embedding table is sized from the tokenizer's vocabulary, so swapping the tokenizer (English -> Malayalam)
needs no model edits. The config object is copied, so the shared QWEN2_5_0_5B preset is never modified."""
import dataclasses
import math

from .. import models  # noqa: F401  (registers "qwen")
from ..config import ModelConfig
from ..init import apply_init
from ..registry import MODELS


def resolve_model_config(model_cfg: ModelConfig, tokenizer=None, context_len: int = None) -> ModelConfig:
    """Return a COPY of model_cfg with vocab_size taken from the tokenizer and max_seq_len set to the training
    context. With auto_vocab (default) vocab is rounded up to a multiple of `vocab_pad_multiple`; with
    auto_vocab=False the vocab_size written in the config is used as is (matches official Qwen2.5 / old checkpoints)."""
    updates = {}
    if tokenizer is not None and model_cfg.auto_vocab:
        m = max(int(model_cfg.vocab_pad_multiple), 1)
        updates["vocab_size"] = int(math.ceil(len(tokenizer) / m) * m)
    if context_len is not None:
        updates["max_seq_len"] = int(context_len)
    cfg = dataclasses.replace(model_cfg, **updates) if updates else dataclasses.replace(model_cfg)
    if tokenizer is not None:
        # with auto_vocab=False the written vocab_size is kept (e.g. 151936 = official Qwen2.5) - it must still fit the tokenizer
        assert cfg.vocab_size >= len(tokenizer), \
            f"model.vocab_size {cfg.vocab_size} is smaller than the tokenizer ({len(tokenizer)} tokens)"
        assert cfg.vocab_size >= len(tokenizer) and tokenizer.eos_id < cfg.vocab_size and tokenizer.pad_id < cfg.vocab_size
    return cfg


def build_model(model_cfg: ModelConfig, init_cfg, tokenizer=None, context_len: int = None):
    """Fresh, randomly initialised model (never pretrained weights). Returns (model, resolved_config)."""
    cfg = resolve_model_config(model_cfg, tokenizer, context_len)
    model = MODELS.get(cfg.name)(cfg)
    apply_init(model, init_cfg)
    return model, cfg
