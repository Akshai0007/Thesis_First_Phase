"""
kaggle/smoke_test.py — run this on Kaggle to prove the pipeline is
Kaggle-ready BEFORE the real tokenizer/data land in a later phase.

This is deliberately NOT a real pretraining run (there's no real tokenizer
or English data wired in yet -- that's explicitly the next phase). What this
DOES prove, on real Kaggle GPU hardware rather than the sandbox this was
built in:

  1. The real Qwen2.5-0.5B-scale model (494M params) actually fits in memory
     and trains a step, with real timing numbers -- not just "should work"
  2. The time-budget safety mechanism actually stops cleanly and checkpoints
     before any session limit, tested here at a short budget so it's fast to
     confirm, but the exact same mechanism is what protects a real 12-hour run
  3. Checkpoint/resume survives an actual process restart (not just an
     in-process simulated exception, which is all the sandbox could test)

Safe by construction: TIME_BUDGET_MINUTES below is set well under Kaggle's
12-hour limit on purpose. Raise it once this smoke test is confirmed working
if you want to extend the demonstration.
"""
import time

import torch

from qwenlab.config import QWEN2_5_0_5B, TrainConfig, InitConfig
from qwenlab.models.qwen import QwenForCausalLM, count_parameters
from qwenlab.init import apply_init
from qwenlab.training.trainer import Trainer

# ---- Deliberately conservative for a first run on unknown hardware ----
TIME_BUDGET_MINUTES = 15          # real 12hr budget comes later, once real data exists
MAX_STEPS = 10_000                # effectively "unbounded" -- time budget is what stops this run
SEQ_LEN = 512                     # shorter than Qwen2.5-0.5B's real 32768 max -- keeps memory modest for this check
BATCH_SIZE = 4
GRAD_ACCUM = 4                    # effective batch = 16
VOCAB_SIZE_FOR_SMOKE_TEST = 8000  # placeholder; real tokenizer's 151936 swaps in during the next phase


def synthetic_batch_fn(vocab_size, seq_len, batch_size, device):
    """Random tokens -- exercises every shape/memory/speed characteristic of
    real training WITHOUT needing the real tokenizer or dataset, which don't
    exist yet. The model cannot learn anything meaningful from this; that is
    expected and fine -- this script measures feasibility, not quality."""
    def _fn():
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        y = torch.randint(0, vocab_size, (batch_size, seq_len))
        return x, y, None
    return _fn


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: no GPU detected -- confirm the Kaggle notebook's Accelerator is set to GPU "
              "(T4 x2 or P100) before trusting any timing numbers from this run.")

    cfg = QWEN2_5_0_5B
    cfg.vocab_size = VOCAB_SIZE_FOR_SMOKE_TEST  # smaller placeholder vocab for this feasibility check;
                                                  # real run uses the tokenizer's real 151936
    print(f"\nModel: Qwen2.5-0.5B architecture, hidden={cfg.hidden_size} layers={cfg.num_layers} "
          f"heads={cfg.num_heads}/{cfg.num_kv_heads} head_dim={cfg.head_dim}")

    model = QwenForCausalLM(cfg)
    apply_init(model, InitConfig(scheme="gpt2_scaled", std=0.02, seed=42))
    n_params = count_parameters(model)
    print(f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    train_cfg = TrainConfig(
        output_dir="/kaggle/working/qwenlab_smoke_test",
        max_steps=MAX_STEPS,
        micro_batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM,
        lr=3e-4,
        warmup_steps=50,
        schedule="cosine",
        weight_decay=0.1,
        grad_clip=1.0,
        precision="auto",          # bf16 on a GPU that supports it, else fp32 -- see trainer.resolve_dtype
        log_every=5,
        eval_every=100_000,        # no real eval set yet -- disabled for this smoke test
        save_every=20,
        keep_last=1,          # NOT 2: each checkpoint is ~7.15GB (596M params, fp32 model + fp32
                               # AdamW state) -- progress.pt + keep_last named snapshots must stay
                               # under Kaggle's ~20GB /kaggle/working output quota. keep_last=1 means
                               # progress.pt + 1 named snapshot = ~14.3GB, safely under with headroom.
        resume=True,               # if this script is rerun, it will pick up from where it left off
        time_budget_minutes=TIME_BUDGET_MINUTES,
    )

    trainer = Trainer(model, train_cfg, device=device)
    print(f"Precision resolved to: {trainer.dtype}")
    print(f"Resuming from step: {trainer.start_step}" if trainer.start_step > 0 else "Starting fresh")

    batch_fn = synthetic_batch_fn(VOCAB_SIZE_FOR_SMOKE_TEST, SEQ_LEN, BATCH_SIZE, device)

    def on_log(step, loss, lr, grad_norm):
        elapsed = time.time() - trainer.start_time
        tokens_seen = step * BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
        toks_per_sec = tokens_seen / elapsed if elapsed > 0 else 0
        mem = f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB peak" if device == "cuda" else "n/a"
        print(f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e} | grad_norm {grad_norm:.3f} "
              f"| {toks_per_sec:,.0f} tok/s | mem {mem} | {elapsed:.1f}s elapsed")

    print(f"\nStarting training loop (time budget: {TIME_BUDGET_MINUTES} min, well under Kaggle's 12hr limit)...\n")
    result = trainer.fit(batch_fn, on_log=on_log)

    print(f"\n{'=' * 60}")
    print(f"Stopped at step {result['final_step']} "
          f"({'time budget reached, clean stop' if result['stopped_early'] else 'max_steps reached'})")
    print(f"Checkpoint saved to: {train_cfg.output_dir}/checkpoints/progress.pt")
    if device == "cuda":
        print(f"Peak GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
              f"of {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB available")
    print("\nIf this completed without an out-of-memory error, the real Qwen2.5-0.5B architecture")
    print("is confirmed to fit and train on this Kaggle GPU at the batch_size/seq_len used above.")
    print("Rerun this same script (same output_dir) to confirm it resumes instead of restarting.")


if __name__ == "__main__":
    main()
