"""The chunked and unchunked loss paths must agree, including with a loss_mask.
Until now every test used loss_chunk_tokens=0, so the chunked path (the one
the real 151936-vocab run uses) was never exercised, and the unchunked path
silently ignored loss_mask."""
import dataclasses

import torch

from qwenlab.config import ModelConfig, InitConfig
from qwenlab.init import apply_init
from qwenlab.models.qwen import QwenForCausalLM


def _model(chunk):
    cfg = ModelConfig(vocab_size=200, hidden_size=32, intermediate_size=88,
                      num_layers=2, num_heads=4, num_kv_heads=2,
                      qkv_bias=False, tie_embeddings=True, loss_chunk_tokens=chunk)
    m = QwenForCausalLM(cfg)
    apply_init(m, InitConfig(scheme="gpt2_scaled", std=0.02, seed=0))
    return m


def _batch():
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 200, (3, 17), generator=g)
    x, y = ids[:, :-1], ids[:, 1:]           # non-contiguous slices, like real tasks
    mask = torch.zeros_like(y, dtype=torch.bool)
    mask[:, 8:] = True                        # "prompt" = first 8 positions
    return x, y, mask


def _loss(model, x, y, mask):
    return model(x, labels=y, loss_mask=mask)[1]


def test_chunked_equals_unchunked_without_mask():
    x, y, _ = _batch()
    a, b = _model(0), _model(7)               # 7 does not divide 3*16=48
    b.load_state_dict(a.state_dict())
    assert torch.allclose(_loss(a, x, y, None), _loss(b, x, y, None), atol=1e-5)


def test_chunked_equals_unchunked_with_mask():
    x, y, mask = _batch()
    a, b = _model(0), _model(7)
    b.load_state_dict(a.state_dict())
    assert torch.allclose(_loss(a, x, y, mask), _loss(b, x, y, mask), atol=1e-5)


def test_mask_actually_changes_the_loss_in_both_paths():
    x, y, mask = _batch()
    for chunk in (0, 7):
        m = _model(chunk)
        assert not torch.allclose(_loss(m, x, y, mask), _loss(m, x, y, None))


def test_masked_loss_equals_manual_ignore_labels():
    x, y, mask = _batch()
    m = _model(0)
    manual = y.masked_fill(~mask, -100)
    assert torch.allclose(_loss(m, x, y, mask), m(x, labels=manual)[1], atol=1e-6)


def test_gradients_match_between_paths():
    x, y, mask = _batch()
    a, b = _model(0), _model(7)
    b.load_state_dict(a.state_dict())
    _loss(a, x, y, mask).backward()
    _loss(b, x, y, mask).backward()
    assert torch.allclose(a.embed_tokens.weight.grad, b.embed_tokens.weight.grad, atol=1e-5)
