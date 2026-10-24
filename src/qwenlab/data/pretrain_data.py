"""Raw text -> packed causal-LM windows (next-word prediction), with a disk cache.

Steps: source (yields documents) -> clean (drop short / Q&A-formatted documents) -> tokenizer -> concatenate
documents separated by end-of-text -> cut into windows of seq_len+1 tokens (input = window[:-1],
target = window[1:]). The first documents go to the held-out validation set until `val_tokens` is reached, so a
document is never in both train and validation. Tokens are stored as one compact numpy array (4 bytes each),
not as Python lists, so hundreds of millions of tokens fit in memory.
"""
import hashlib
import json
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .preprocess import clean_document
from .sources import build_source


class WindowView:
    """Read-only view of a flat token array as non-overlapping windows of seq_len+1 tokens."""

    def __init__(self, tokens: np.ndarray, seq_len: int):
        self.tokens, self.seq_len = tokens, seq_len
        self.n = max((len(tokens) - 1) // seq_len, 0)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        return self.tokens[s: s + self.seq_len + 1]


def lm_make_batch(windows, device="cpu"):
    """list of windows -> (input_ids, labels, loss_mask=None). Every position is a target (plain next-word)."""
    arr = torch.as_tensor(np.stack([np.asarray(w) for w in windows]).astype(np.int64))
    return arr[:, :-1].to(device), arr[:, 1:].to(device), None


@dataclass
class PretrainData:
    train: WindowView
    val: WindowView
    stats: dict


def _fingerprint(data_cfg, tok_cfg, vocab: int) -> str:
    blob = json.dumps({"source": [data_cfg.source.kind, data_cfg.source.params], "tok": [tok_cfg.name, tok_cfg.path],
                       "vocab": vocab, "max_train_tokens": data_cfg.max_train_tokens, "val_tokens": data_cfg.val_tokens,
                       "min_chars": data_cfg.min_chars, "norm": [data_cfg.normalizer, data_cfg.script, data_cfg.min_script_ratio],
                       "val_frac": data_cfg.max_val_fraction, "drop_qa": data_cfg.drop_qa_format}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def build_pretrain_data(data_cfg, tok_cfg, tokenizer, log=print) -> PretrainData:
    cache = Path(data_cfg.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    fp = _fingerprint(data_cfg, tok_cfg, len(tokenizer))
    f_train, f_val, f_meta = cache / f"{fp}_train.npy", cache / f"{fp}_val.npy", cache / f"{fp}_meta.json"

    if f_train.exists() and f_val.exists() and f_meta.exists():
        stats = json.loads(f_meta.read_text())
        log(f"[data] loaded cached tokens ({fp}): {stats['train_tokens']:,} train / {stats['val_tokens']:,} val")
        return PretrainData(WindowView(np.load(f_train, mmap_mode="r"), data_cfg.seq_len),
                            WindowView(np.load(f_val, mmap_mode="r"), data_cfg.seq_len), stats)

    if data_cfg.max_train_tokens is None and data_cfg.source.kind == "hf_dataset":
        raise ValueError("data.max_train_tokens must be set for a streaming hf_dataset source (it has no end)")
    target_total = None if data_cfg.max_train_tokens is None else data_cfg.max_train_tokens + data_cfg.val_tokens

    val, train = array("I"), array("I")
    docs = {"kept": 0, "dropped_short": 0, "dropped_qa_format": 0, "dropped_wrong_script": 0}
    for text in build_source(data_cfg.source):
        text, why = clean_document(text, data_cfg.min_chars, data_cfg.normalizer, data_cfg.script,
                                   data_cfg.min_script_ratio, data_cfg.drop_qa_format)
        if text is None:
            docs["dropped_" + why] += 1
            continue
        ids = tokenizer.encode(text)
        ids.append(tokenizer.eos_id)
        # a document goes to validation only while validation is below val_tokens AND below max_val_fraction of
        # all text so far, so a small dataset still keeps most of its text for training
        to_val = (len(val) < data_cfg.val_tokens and
                  len(val) + len(ids) <= data_cfg.max_val_fraction * (len(val) + len(train) + len(ids)))
        (val if to_val else train).extend(ids)
        docs["kept"] += 1
        if target_total is not None and len(val) + len(train) >= target_total:
            break
        if docs["kept"] % 20000 == 0:
            log(f"[data] {docs['kept']:,} docs, {len(val) + len(train):,} tokens")
    assert len(train) > data_cfg.seq_len + 1, "not enough training text for even one window"
    assert len(val) > data_cfg.seq_len + 1, (
        f"validation set has only {len(val)} tokens (< one window of {data_cfg.seq_len + 1}) out of "
        f"{len(val) + len(train):,} in total: use more text, a smaller data.seq_len, or a larger data.max_val_fraction")

    train_np, val_np = np.frombuffer(train, dtype=np.uint32), np.frombuffer(val, dtype=np.uint32)
    assert int(train_np.max()) < len(tokenizer) and int(val_np.max()) < len(tokenizer)
    stats = {**docs, "train_tokens": int(len(train_np)), "val_tokens": int(len(val_np)), "fingerprint": fp,
             "dropped_qa_format_note": "documents containing chat/FIM markers or Question:/Answer: layouts are excluded"}
    np.save(f_train, train_np); np.save(f_val, val_np); f_meta.write_text(json.dumps(stats, indent=2))
    log(f"[data] built {fp}: {stats}")
    return PretrainData(WindowView(train_np, data_cfg.seq_len), WindowView(val_np, data_cfg.seq_len), stats)


def build_eval_windows(data_cfg, tokenizer, source_cfg, max_tokens: int = 200_000) -> WindowView:
    """Windows from ANY other text source (e.g. WikiText for out-of-domain perplexity), cleaned and tokenized
    exactly like the training text. Not cached, not used for training."""
    ids = array("I")
    for text in build_source(source_cfg):
        text, _ = clean_document(text, data_cfg.min_chars, data_cfg.normalizer, data_cfg.script,
                                 data_cfg.min_script_ratio, data_cfg.drop_qa_format)
        if text is None:
            continue
        ids.extend(tokenizer.encode(text) + [tokenizer.eos_id])
        if len(ids) >= max_tokens:
            break
    view = WindowView(np.frombuffer(ids, dtype=np.uint32), data_cfg.seq_len)
    assert len(view) > 0, "the evaluation source gave less than one window of text after cleaning"
    return view
