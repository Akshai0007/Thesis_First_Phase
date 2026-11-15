"""Data-side v0.4 options: byte accounting, uint16 cache, exact dedup, and that DEFAULT settings leave cache keys unchanged."""
import hashlib
import json

import numpy as np
import pytest

from qwenlab.config import PretrainDataConfig, SourceConfig, TokenizerConfig
from qwenlab.data.pretrain_data import _fingerprint, build_pretrain_data
from qwenlab.data.tokenizers import ByteTokenizer

TEXT = "\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02 \u0d35\u0d3e\u0d15\u0d4d\u0d2f\u0d02 "


def _corpus(tmp_path, n=40, dup=True):
    f = tmp_path / "c.jsonl"
    rows = [{"text": TEXT * (8 + i % 5) + str(i)} for i in range(n)]
    if dup:
        rows += [rows[3], rows[7]]
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(f)


def _cfg(tmp_path, **kw):
    return PretrainDataConfig(source=SourceConfig(kind="text_files", params={"pattern": _corpus(tmp_path)}),
                              cache_dir=str(tmp_path / "cache"), seq_len=32, val_tokens=2000, min_chars=20, **kw)


def test_default_fingerprint_is_exactly_the_old_one():
    """Old caches (built before v0.4) must still be found: with default v0.4 options the cache key is the v0.3 key."""
    cfg, tok = PretrainDataConfig(), TokenizerConfig()
    old = json.dumps({"source": [cfg.source.kind, cfg.source.params], "tok": [tok.name, tok.path], "vocab": 257,
                      "max_train_tokens": cfg.max_train_tokens, "val_tokens": cfg.val_tokens, "min_chars": cfg.min_chars,
                      "norm": [cfg.normalizer, cfg.script, cfg.min_script_ratio], "val_frac": cfg.max_val_fraction,
                      "drop_qa": cfg.drop_qa_format}, sort_keys=True)
    assert _fingerprint(cfg, tok, 257) == hashlib.md5(old.encode()).hexdigest()[:12]
    assert _fingerprint(PretrainDataConfig(token_dtype="uint16"), tok, 257) != _fingerprint(cfg, tok, 257)
    assert _fingerprint(PretrainDataConfig(dedup_exact=True), tok, 257) != _fingerprint(cfg, tok, 257)


def test_byte_accounting_matches_the_text(tmp_path):
    cfg = _cfg(tmp_path)
    d = build_pretrain_data(cfg, TokenizerConfig(name="bytes"), ByteTokenizer(), log=lambda *a: None)
    s = d.stats
    assert s["val_bytes"] > 0 and s["train_bytes"] > 0
    # byte tokenizer: one token per byte + one EOS per document -> tokens == bytes + documents
    assert s["train_tokens"] + s["val_tokens"] == s["train_bytes"] + s["val_bytes"] + s["kept"]


def test_uint16_cache_and_dedup(tmp_path):
    cfg = _cfg(tmp_path, token_dtype="uint16", dedup_exact=True)
    d = build_pretrain_data(cfg, TokenizerConfig(name="bytes"), ByteTokenizer(), log=lambda *a: None)
    assert d.train.tokens.dtype == np.uint16 and d.stats["dropped_duplicate"] == 2
    again = build_pretrain_data(cfg, TokenizerConfig(name="bytes"), ByteTokenizer(), log=lambda *a: None)   # cached read
    assert again.train.tokens.dtype == np.uint16 and len(again.train) == len(d.train)
    from qwenlab.data.pretrain_data import lm_make_batch
    x, y, _ = lm_make_batch([d.train[0], d.train[1]])
    assert x.dtype.is_floating_point is False and x.shape == (2, 32)


def test_uint16_refuses_a_big_vocabulary(tmp_path):
    class Big(ByteTokenizer):
        def __len__(self): return 70000
    with pytest.raises(AssertionError, match="65,536"):
        build_pretrain_data(_cfg(tmp_path, token_dtype="uint16"), TokenizerConfig(name="bytes"), Big(), log=lambda *a: None)
