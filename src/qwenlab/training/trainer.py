"""The training loop. Ties together: model + init (already applied before
Trainer is constructed) + a data iterator + LR schedule + optimizer +
checkpointing/resume + a Kaggle session-time budget.

Deliberately takes a plain Python iterator for `data_iter` (yields
(input_ids, labels, loss_mask) triples) rather than importing anything from
tasks.py or data/ directly -- the trainer has no idea whether it's doing
causal_lm pretraining, an SFT task, or FIM. That independence is what makes
"swap the task without touching the trainer" actually true rather than
aspirational.
"""
import os
import shutil
import time
from pathlib import Path

import torch

import json
import math

from ..evaluation.perplexity import evaluate_lm
from ..registry import OPTIMIZERS
from ..training import optimizers as _optimizers   # noqa: F401  (registers "adamw" and "muon_adamw")
from ..training.schedules import lr_multiplier


def build_optimizer(model: torch.nn.Module, train_cfg):
    """Optimizer chosen by `train.optimizer` (registry in training/optimizers.py). The default, "adamw", is the
    unchanged AdamW with weight decay only on >=2-D weight matrices (no decay on norms, biases, embeddings)."""
    return OPTIMIZERS.get(train_cfg.optimizer)(model, train_cfg)


def resolve_dtype(precision: str, device: str):
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    # auto: bf16 on Ampere/Hopper (A100 ...), fp16 + GradScaler on older GPUs (T4: its bf16 is emulated and slow;
    # the Trainer already handles the GradScaler whenever the dtype is fp16), fp32 on CPU.
    if str(device).startswith("cuda"):
        return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    return torch.float32


class Trainer:
    def __init__(self, model, train_cfg, device: str = None):
        self.model = model
        self.cfg = train_cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = build_optimizer(model, train_cfg)
        self.dtype = resolve_dtype(train_cfg.precision, self.device)
        self.use_scaler = self.dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        self.ckpt_dir = Path(train_cfg.output_dir) / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.start_step = 0
        self.start_time = None
        # v0.4 throughput / progress accounting (logged with every training metric row)
        self._fwd = model
        if train_cfg.compile and str(self.device).startswith("cuda"):
            self._fwd = torch.compile(model)
        self.tokens_seen = 0
        self.train_time_s = 0.0           # cumulative training time (data loading included, evals/saves excluded), survives resume (for equal-wall-clock comparisons)
        self._acc_time, self._acc_tokens, self._acc_steps = 0.0, 0, 0
        self._last_step_tokens = 0
        self.flops_per_token = None        # set by the pipeline; enables the "mfu" column
        self.peak_flops = None
        self.bpb_factor = None             # tokens per UTF-8 byte of the validation text; enables "val_bpb"
        self.metrics_path = Path(train_cfg.output_dir) / "metrics.jsonl"   # one JSON line per log / eval event
        self.last_eval = None

        if train_cfg.resume:
            self._maybe_resume()

    # ------------------------------------------------------------------ checkpointing
    def _progress_path(self) -> Path:
        return self.ckpt_dir / "progress.pt"

    def _maybe_resume(self):
        p = self._progress_path()
        if p.exists():
            # Load on CPU first: loading straight to GPU holds checkpoint + model +
            # optimizer copy at once (~12GB for this model on a 15GB T4).
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.use_scaler and "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.start_step = ckpt["step"]
            self.tokens_seen = int(ckpt.get("tokens_seen", 0))
            self.train_time_s = float(ckpt.get("train_time_s", 0.0))
            print(f"[resume] loaded {p}, continuing from step {self.start_step}")

    def save_checkpoint(self, step: int, extra: dict = None, final: bool = False):
        for name, prm in self.model.named_parameters():
            if not bool(torch.isfinite(prm).all()):
                raise FloatingPointError(f"parameter {name} contains NaN/inf at step {step}: NOT saving, "
                                         f"the previous checkpoint stays untouched")
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "cfg": self.cfg,
            "tokens_seen": self.tokens_seen,
            "train_time_s": self.train_time_s,
        }
        if self.use_scaler:
            payload["scaler_state_dict"] = self.scaler.state_dict()
        if extra:
            payload.update(extra)

        # Write to a temp file, flush to disk, then rename over the target.
        # os.replace is atomic, so a kill mid-write leaves the previous good
        # progress.pt untouched (a half-written .tmp is simply overwritten next time).
        progress = self._progress_path()
        tmp = progress.with_suffix(".pt.tmp")
        with open(tmp, "wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, progress)

        if final or (self.cfg.save_every and step % self.cfg.save_every == 0):
            named = self.ckpt_dir / f"step_{step}.pt"
            if named.exists():
                named.unlink()
            try:
                os.link(progress, named)        # hard link: instant, no extra disk
            except OSError:
                shutil.copyfile(progress, named)  # filesystem without hard links
            self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        named = sorted(self.ckpt_dir.glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        for old in named[:-self.cfg.keep_last] if self.cfg.keep_last > 0 else []:
            old.unlink()

    # ------------------------------------------------------------------ core loop
    def set_lr(self, step: int):
        mult = lr_multiplier(step, self.cfg)
        lr = self.cfg.lr * mult
        for g in self.optimizer.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)       # lr_scale only exists for multi-optimizer setups (Muon); 1.0 otherwise
        return lr

    def training_step(self, batches: list):
        """`batches` is a list of grad_accum_steps micro-batches, each
        (input_ids, labels, loss_mask)."""
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        step_tokens = 0
        for input_ids, labels, loss_mask in batches:
            step_tokens += int(input_ids.numel())
            input_ids, labels = input_ids.to(self.device), labels.to(self.device)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device)
            with torch.autocast(device_type=self.device if self.device != "mps" else "cpu",
                                 dtype=self.dtype, enabled=self.dtype != torch.float32):
                _, loss = self._fwd(input_ids, labels=labels, loss_mask=loss_mask)
            loss = loss / len(batches)
            if self.use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

        opts = getattr(self.optimizer, "optimizers", [self.optimizer])      # a bundle of optimizers, or just one
        if self.use_scaler:
            for o in opts:
                self.scaler.unscale_(o)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        if self.use_scaler:
            for o in opts:
                self.scaler.step(o)
            self.scaler.update()
        else:
            self.optimizer.step()

        self._last_step_tokens = step_tokens
        return total_loss, grad_norm.item()

    def time_budget_exceeded(self) -> bool:
        if self.cfg.time_budget_minutes is None or self.start_time is None:
            return False
        return (time.time() - self.start_time) / 60.0 >= self.cfg.time_budget_minutes

    def fit(self, train_batch_fn, val_batch_fn=None, on_log=None, on_eval=None, on_checkpoint=None):
        """train_batch_fn() -> (input_ids, labels, loss_mask), called
        grad_accum_steps times per training_step. val_batch_fn, if given, is
        called cfg.eval_batches times at each eval_every interval.
        on_log(step, loss, lr, grad_norm) and on_eval(step, val_loss) are
        optional callbacks -- keeps this loop free of any assumption about
        HOW results get reported (print, a logger, a notebook cell, etc)."""
        self.start_time = time.time()
        step = self.start_step
        stopped_early = False
        stop_at = self.cfg.stop_at_step
        if stop_at is not None and stop_at <= self.start_step:
            print(f"[stop_at_step] {stop_at} is not after the resume step {self.start_step}: nothing to train.")
        end = self.cfg.max_steps if stop_at is None else min(self.cfg.max_steps, stop_at)
        last_saved = None

        while step < end:
            if self.time_budget_exceeded():
                print(f"[time budget] {self.cfg.time_budget_minutes} min reached at step {step}, "
                      f"stopping cleanly and checkpointing.")
                stopped_early = True
                break

            lr = self.set_lr(step)
            t_step = time.perf_counter()                       # includes fetching the batches: this is END-TO-END throughput
            batches = [train_batch_fn() for _ in range(self.cfg.grad_accum_steps)]
            loss, grad_norm = self.training_step(batches)      # ends with .item() calls, so the wall time is accurate
            dt = time.perf_counter() - t_step
            step += 1
            ntok = int(getattr(self, "_last_step_tokens", 0))
            self.tokens_seen += ntok
            self.train_time_s += dt
            self._acc_time += dt; self._acc_tokens += ntok; self._acc_steps += 1

            if step % self.cfg.log_every == 0:
                row = {"step": step, "train_loss": loss, "lr": lr, "grad_norm": grad_norm, "tokens_seen": self.tokens_seen,
                       "train_time_s": self.train_time_s}
                if self._acc_time > 0:
                    row["tok_s"] = self._acc_tokens / self._acc_time          # end-to-end training throughput (data loading included; evals/saves excluded)
                    row["s_per_step"] = self._acc_time / max(self._acc_steps, 1)
                    if self.flops_per_token and self.peak_flops:
                        row["mfu"] = row["tok_s"] * self.flops_per_token / self.peak_flops
                if str(self.device).startswith("cuda"):
                    row["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
                self._acc_time, self._acc_tokens, self._acc_steps = 0.0, 0, 0
                self._log_metric(row)
                if on_log:
                    on_log(step, loss, lr, grad_norm)

            if val_batch_fn and step % self.cfg.eval_every == 0:
                val_loss = self.evaluate(val_batch_fn, step=step)
                if on_eval:
                    on_eval(step, val_loss)

            # Checkpoint only every save_every steps, NOT every step. Caught by
            # manual review before any Kaggle run: a checkpoint for this
            # project's active target (Qwen2.5-0.5B, 494M params, fp32 model +
            # fp32 AdamW moments) is ~5.93GB -- was ~7.15GB when Qwen3-0.6B was
            # the active target, still large either way. Writing that every
            # single step would both cripple throughput (I/O-bound instead of
            # compute-bound) and, combined with keep_last named snapshots,
            # risk exceeding Kaggle's ~20GB /kaggle/working output quota.
            # save_every trades a small amount of resume-granularity (lose at
            # most save_every-1 steps on a crash) for both speed and staying
            # safely under quota. This math scales with whichever model is
            # actually active -- re-check it if the target model changes again.
            if self.cfg.save_every and step % self.cfg.save_every == 0:
                self.save_checkpoint(step)
                last_saved = step
                if on_checkpoint:
                    on_checkpoint(step, self._progress_path(), False)

        stopped_early = stopped_early or step < self.cfg.max_steps      # paused (time budget / stop_at_step), not finished
        if last_saved != step:                      # no second copy of a checkpoint that was just written
            self.save_checkpoint(step, final=True)
        if on_checkpoint:
            on_checkpoint(step, self._progress_path(), True)
        return {"final_step": step, "stopped_early": stopped_early}

    def _log_metric(self, row: dict):
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def evaluate(self, val_batch_fn, step: int = None) -> float:
        """Held-out loss AND perplexity (token-weighted). Prints and logs both; returns the loss
        (the full result is kept in self.last_eval)."""
        step = self.start_step if step is None else step
        res = evaluate_lm(self.model, val_batch_fn, self.cfg.eval_batches, self.device, self.dtype)
        self.last_eval = res
        print(f"[eval] step {step}: val loss {res['loss']:.4f} | perplexity {res['perplexity']:.2f} | {res['tokens']:,} tokens")
        row = {"step": step, "val_loss": res["loss"], "val_perplexity": res["perplexity"], "val_tokens": res["tokens"],
               "tokens_seen": self.tokens_seen, "train_time_s": self.train_time_s}
        if self.bpb_factor:
            row["val_bpb"] = res["loss"] / math.log(2) * self.bpb_factor
            print(f"[eval] step {step}: val bits-per-byte {row['val_bpb']:.4f}")
        self._log_metric(row)
        return res["loss"]

    _evaluate = evaluate      # old name kept so existing callers keep working
