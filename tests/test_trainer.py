import shutil
import tempfile
from pathlib import Path

import torch
import pytest

from qwenlab.config import ModelConfig, TrainConfig, InitConfig
from qwenlab.models.qwen import QwenForCausalLM
from qwenlab.init import apply_init
from qwenlab.training.trainer import Trainer, build_optimizer


def _tiny_model(seed=0):
    """Applies our own apply_init, same as any real usage would -- without
    it, PyTorch's raw default nn.Linear/nn.Embedding init is used instead,
    which is a much larger-magnitude distribution than our std=0.02 scheme
    and (combined with tied embeddings over a small vocab) produced a
    starting loss far ABOVE the random-guess baseline in practice, making a
    'does loss go down' test noisier and slower to converge than it should be."""
    cfg = ModelConfig(vocab_size=200, hidden_size=32, intermediate_size=88,
                       num_layers=2, num_heads=4, num_kv_heads=2,
                       qkv_bias=False, tie_embeddings=True, loss_chunk_tokens=0)
    model = QwenForCausalLM(cfg)
    apply_init(model, InitConfig(scheme="gpt2_scaled", std=0.02, seed=seed))
    return model


def _synthetic_batch_fn(vocab_size=200, seq_len=16, batch_size=4, seed=0):
    """A trivially learnable synthetic task: predict token (t+1) mod vocab_size
    from token t -- deterministic pattern, so loss going down is unambiguous."""
    g = torch.Generator().manual_seed(seed)

    def _fn():
        start = torch.randint(0, vocab_size, (batch_size, 1), generator=g)
        seq = (start + torch.arange(seq_len + 1)) % vocab_size
        x, y = seq[:, :-1], seq[:, 1:]
        return x, y, None
    return _fn


@pytest.fixture
def tmp_output_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_build_optimizer_excludes_norms_and_embeddings_from_weight_decay():
    model = _tiny_model()
    cfg = TrainConfig(weight_decay=0.1, no_decay_embeddings=True)
    opt = build_optimizer(model, cfg)
    decay_group, no_decay_group = opt.param_groups
    assert decay_group["weight_decay"] == 0.1
    assert no_decay_group["weight_decay"] == 0.0

    decay_params = set(id(p) for p in decay_group["params"])
    for name, p in model.named_parameters():
        is_norm_or_embed = "norm" in name or "embed_tokens" in name or "lm_head" in name or p.dim() < 2
        if is_norm_or_embed:
            assert id(p) not in decay_params, f"{name} should NOT have weight decay"
        else:
            assert id(p) in decay_params, f"{name} SHOULD have weight decay"


def test_loss_decreases_on_a_learnable_synthetic_task(tmp_output_dir):
    model = _tiny_model()
    cfg = TrainConfig(output_dir=tmp_output_dir, max_steps=150, lr=3e-3, warmup_steps=5,
                       grad_accum_steps=1, log_every=1, eval_every=1000, save_every=1000,
                       resume=False, precision="fp32")
    trainer = Trainer(model, cfg, device="cpu")
    batch_fn = _synthetic_batch_fn()

    losses = []
    trainer.fit(batch_fn, on_log=lambda step, loss, lr, gn: losses.append(loss))

    # Trend-based check (robust to a single noisy sample) rather than a hard
    # absolute threshold: average of the first 10 logged steps vs the last 10.
    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10
    assert late_avg < early_avg * 0.5, (
        f"loss should have dropped substantially on a trivial synthetic task: "
        f"early avg={early_avg:.3f}, late avg={late_avg:.3f}"
    )


def test_checkpoint_resume_survives_a_simulated_crash(tmp_output_dir):
    """Same methodology proven twice already in the earlier project: kill
    training partway through, confirm a fresh Trainer resumes from the exact
    saved step with matching optimizer state, not from zero."""
    # save_every=5, NOT 1000: the resumable checkpoint is now only written at
    # the save_every cadence (fixed for Kaggle's disk quota -- see
    # trainer.py's fit()), not every step. This test's crash happens at
    # ~step 20, so save_every must be small enough that a checkpoint already
    # exists by then, or there's nothing to resume from. Caught on real
    # Kaggle hardware: this was still 1000 from before that fix, when
    # checkpointing-every-step made the exact value irrelevant to this test.
    cfg = TrainConfig(output_dir=tmp_output_dir, max_steps=100, lr=1e-3, warmup_steps=5,
                       grad_accum_steps=1, log_every=1, eval_every=1000, save_every=5,
                       resume=True, precision="fp32")
    batch_fn = _synthetic_batch_fn(seed=1)

    model1 = _tiny_model()
    trainer1 = Trainer(model1, cfg, device="cpu")

    call_count = {"n": 0}
    orig_step = trainer1.training_step

    def crash_after_n(batches):
        call_count["n"] += 1
        if call_count["n"] == 21:
            raise RuntimeError("SIMULATED CRASH")
        return orig_step(batches)
    trainer1.training_step = crash_after_n

    with pytest.raises(RuntimeError, match="SIMULATED CRASH"):
        trainer1.fit(batch_fn)

    progress = torch.load(Path(tmp_output_dir) / "checkpoints" / "progress.pt", weights_only=False)
    saved_step = progress["step"]
    assert 0 < saved_step < 100, f"expected a partial checkpoint, got step={saved_step}"

    model2 = _tiny_model()
    trainer2 = Trainer(model2, cfg, device="cpu")
    assert trainer2.start_step == saved_step

    for n1, p1 in model1.named_parameters():
        p2 = dict(model2.named_parameters())[n1]
        assert torch.equal(p1, p2), f"{n1} did not match after resume"


def test_gradient_accumulation_matches_one_large_batch(tmp_output_dir):
    """grad_accum_steps=4 with batch_size=B should produce (approximately,
    modulo float summation order) the same gradient as one batch of size 4B
    -- this is the whole point of accumulation, and easy to get subtly wrong
    (e.g. forgetting to divide the loss by the number of accumulation steps)."""
    torch.manual_seed(0)
    model_a = _tiny_model()
    model_b = _tiny_model()
    model_b.load_state_dict(model_a.state_dict())  # identical starting weights

    cfg_accum = TrainConfig(output_dir=tmp_output_dir, max_steps=1, lr=1e-3,
                             grad_accum_steps=4, warmup_steps=0, resume=False, precision="fp32")
    trainer_accum = Trainer(model_a, cfg_accum, device="cpu")

    torch.manual_seed(123)
    micro_batches = [(torch.randint(0, 200, (2, 8)), torch.randint(0, 200, (2, 8)), None) for _ in range(4)]
    trainer_accum.training_step(micro_batches)
    grad_accum = {n: p.grad.clone() for n, p in model_a.named_parameters() if p.grad is not None}

    cfg_single = TrainConfig(output_dir=tmp_output_dir, max_steps=1, lr=1e-3,
                              grad_accum_steps=1, warmup_steps=0, resume=False, precision="fp32")
    trainer_single = Trainer(model_b, cfg_single, device="cpu")
    big_x = torch.cat([m[0] for m in micro_batches], dim=0)
    big_y = torch.cat([m[1] for m in micro_batches], dim=0)
    trainer_single.training_step([(big_x, big_y, None)])
    grad_single = {n: p.grad.clone() for n, p in model_b.named_parameters() if p.grad is not None}

    for n in grad_accum:
        assert torch.allclose(grad_accum[n], grad_single[n], atol=1e-5, rtol=1e-3), \
            f"{n}: accumulated grad diverges from single-large-batch grad"
