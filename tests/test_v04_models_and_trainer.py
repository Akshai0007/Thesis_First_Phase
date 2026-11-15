"""v0.4 additions that need torch (and transformers for the Hugging Face adapter). Run on Kaggle: python -m pytest tests -q
NOT executed in the authoring sandbox (no torch there) - treat the first Kaggle run as their real test."""
import dataclasses
import json
import math

import pytest
import torch

from qwenlab.config import InitConfig, ModelConfig, TrainConfig
from qwenlab.models.factory import build_model
from qwenlab.models.qwen import QwenForCausalLM
from qwenlab.training.trainer import Trainer, build_optimizer

TINY = dict(hidden_size=64, intermediate_size=128, num_layers=2, num_heads=4, num_kv_heads=2, loss_chunk_tokens=0)


class Tok:                                                   # stands in for a Malayalam 32k tokenizer
    eos_id, pad_id = 2, 0
    def __len__(self): return 300


def _batch(vocab=300, b=3, s=17, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (b, s), generator=g)
    return ids[:, :-1], ids[:, 1:]


# ------------------------------------------------------------------ gradient checkpointing / chunk checkpointing (hand-written model)
def _qwen(**kw):
    cfg = ModelConfig(vocab_size=200, qkv_bias=False, **{**TINY, **kw})
    m = QwenForCausalLM(cfg)
    return m


def test_gradient_checkpointing_changes_memory_not_maths():
    a, b = _qwen(), _qwen(gradient_checkpointing=True)
    b.load_state_dict(a.state_dict())
    x, y = _batch(200)
    a.train(); b.train()
    la, lb = a(x, labels=y)[1], b(x, labels=y)[1]
    la.backward(); lb.backward()
    assert torch.allclose(la, lb, atol=1e-6)
    assert torch.allclose(a.layers[0].self_attn.q_proj.weight.grad, b.layers[0].self_attn.q_proj.weight.grad, atol=1e-5)


def test_chunk_checkpoint_matches_plain_chunked_loss_and_grads():
    a, b = _qwen(loss_chunk_tokens=7), _qwen(loss_chunk_tokens=7, loss_chunk_checkpoint=True)
    b.load_state_dict(a.state_dict())
    x, y = _batch(200)
    la, lb = a(x, labels=y)[1], b(x, labels=y)[1]
    la.backward(); lb.backward()
    assert torch.allclose(la, lb, atol=1e-6)
    assert torch.allclose(a.embed_tokens.weight.grad, b.embed_tokens.weight.grad, atol=1e-5)


# ------------------------------------------------------------------ generic Hugging Face adapter
transformers = pytest.importorskip("transformers")
ARCHS = ["qwen2", "qwen3", "llama", "olmo2", "smollm3", "gemma3"]


def _hf(arch, **kw):
    cfg = ModelConfig(name="hf", arch=arch, qkv_bias=True, **{**TINY, **kw})
    try:
        return build_model(cfg, InitConfig(seed=0), Tok(), context_len=32)
    except ImportError as e:                                 # this transformers version has no such architecture
        pytest.skip(str(e))


@pytest.mark.parametrize("arch", ARCHS)
def test_every_arch_builds_trains_one_step_and_obeys_the_contract(arch):
    model, cfg = _hf(arch)
    assert model.num_layers == 2 and cfg.vocab_size >= 300 and cfg.max_seq_len == 32
    x, y = _batch()
    logits, none = model(x)
    assert logits.shape == (3, 16, cfg.vocab_size) and none is None
    _, loss = model(x, labels=y)
    assert abs(float(loss) - math.log(cfg.vocab_size)) < 1.0         # untrained = guessing
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="context window"):
        model(torch.zeros(1, 33, dtype=torch.long))


@pytest.mark.parametrize("arch", ["qwen3", "llama"])
def test_hf_chunked_loss_equals_full_loss_with_and_without_mask(arch):
    a, _ = _hf(arch)
    b, _ = _hf(arch, loss_chunk_tokens=7, loss_chunk_checkpoint=True)
    b.load_state_dict(a.state_dict())
    x, y = _batch()
    mask = torch.zeros_like(y, dtype=torch.bool); mask[:, 8:] = True
    for m in (None, mask):
        assert torch.allclose(a(x, labels=y, loss_mask=m)[1], b(x, labels=y, loss_mask=m)[1], atol=1e-5)


def test_hf_qwen2_has_the_same_parameter_count_as_the_hand_written_model():
    hf, _ = _hf("qwen2")
    mine, _ = build_model(ModelConfig(name="qwen", qkv_bias=True, **TINY), InitConfig(seed=0), Tok(), context_len=32)
    assert sum(p.numel() for p in hf.parameters()) == sum(p.numel() for p in mine.parameters())


def test_init_scheme_reaches_into_the_wrapped_model():
    model, cfg = _hf("llama")
    emb = model.hf.get_input_embeddings().weight
    assert 0.01 < float(emb.std()) < 0.03                                    # N(0, 0.02)
    o = model.hf.model.layers[0].self_attn.o_proj.weight
    assert abs(float(o.std()) - 0.02 / math.sqrt(2 * 2)) < 0.004             # depth-scaled residual projection


def test_unknown_arch_lists_the_choices():
    with pytest.raises(KeyError, match="Available"):
        build_model(ModelConfig(name="hf", arch="nope", **TINY), InitConfig(), Tok(), 32)


def test_hf_model_trains_with_the_unchanged_trainer_and_resumes(tmp_path):
    model, _ = _hf("llama")
    cfg = TrainConfig(output_dir=str(tmp_path), max_steps=6, lr=3e-3, warmup_steps=1, precision="fp32", log_every=2,
                      eval_every=100, save_every=3, keep_last=1, micro_batch_size=3)
    x, y = _batch()
    t = Trainer(model, cfg, device="cpu")
    t.fit(lambda: (x, y, None))
    rows = [json.loads(l) for l in open(tmp_path / "metrics.jsonl")]
    train = [r for r in rows if "train_loss" in r]
    assert train[-1]["train_loss"] < train[0]["train_loss"]
    assert all("tokens_seen" in r and "tok_s" in r and "train_time_s" in r for r in train)
    assert train[-1]["tokens_seen"] == 6 * x.numel()
    model2, _ = _hf("llama")
    t2 = Trainer(model2, cfg, device="cpu")
    assert t2.start_step == 6 and t2.tokens_seen == 6 * x.numel() and t2.train_time_s > 0


# ------------------------------------------------------------------ optimizers
def test_default_optimizer_is_the_old_adamw_with_two_groups():
    m = _qwen()
    opt = build_optimizer(m, TrainConfig(weight_decay=0.1))
    assert isinstance(opt, torch.optim.AdamW) and [g["weight_decay"] for g in opt.param_groups] == [0.1, 0.0]


def test_muon_orthogonalises_and_the_bundle_trains_and_round_trips(tmp_path):
    from qwenlab.training.optimizers import newton_schulz
    s = torch.linalg.svdvals(newton_schulz(torch.randn(64, 128)).float())
    assert 0.3 < float(s.min()) and float(s.max()) < 1.7
    cfg = TrainConfig(output_dir=str(tmp_path), optimizer="muon_adamw", max_steps=4, lr=2e-3, warmup_steps=1,
                      precision="fp32", save_every=2, keep_last=1, log_every=1)
    m = _qwen()
    t = Trainer(m, cfg, device="cpu")
    ids = {id(p) for g in t.optimizer.optimizers[0].param_groups for p in g["params"]}
    assert all(p.dim() == 2 for p in m.parameters() if id(p) in ids)              # Muon only sees matrices
    assert id(m.embed_tokens.weight) not in ids                                    # embeddings stay on AdamW
    x, y = _batch(200)
    t.fit(lambda: (x, y, None))
    t2 = Trainer(_qwen(), cfg, device="cpu")                                      # optimizer state round-trips
    assert t2.start_step == 4


def test_auto_precision_is_fp32_on_cpu():
    from qwenlab.training.trainer import resolve_dtype
    assert resolve_dtype("auto", "cpu") == torch.float32


# ------------------------------------------------------------------ loss by position
def test_loss_by_position_buckets_cover_the_window():
    from qwenlab.evaluation.position_loss import loss_by_position, position_edges
    assert position_edges(100) == [0, 16, 32, 64, 100] and position_edges(16) == [0, 16]
    m = _qwen()
    x, y = _batch(200, b=2, s=65)
    out = loss_by_position(m, lambda: (x, y, None), 2, "cpu")
    assert set(out) == {"0-16", "16-32", "32-64", "overall"}
    expected = float(m(x, labels=y)[1])
    assert abs(out["overall"] - expected) < 1e-4
