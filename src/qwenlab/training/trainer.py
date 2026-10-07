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
import time
from pathlib import Path

import torch

from ..training.schedules import lr_multiplier


def build_optimizer(model: torch.nn.Module, train_cfg):
    """AdamW with weight decay applied only to >=2D weight matrices --
    excludes norm weights, biases, and (if no_decay_embeddings) the token
    embedding/lm_head. Standard practice: decaying a per-channel scale
    factor or a bias term doesn't prevent overfitting the way it does for a
    weight matrix, it just distorts them."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_embedding = "embed_tokens" in name or "lm_head" in name
        if p.dim() < 2 or (train_cfg.no_decay_embeddings and is_embedding):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": train_cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=train_cfg.lr, betas=(train_cfg.beta1, train_cfg.beta2),
                              eps=train_cfg.eps)


def resolve_dtype(precision: str, device: str):
    if precision == "fp32":
        return torch.float32
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    # auto: bf16 on a GPU that supports it, else fp32 (fp16 needs a GradScaler
    # to be safe, deliberately not auto-selected silently)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
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

        if train_cfg.resume:
            self._maybe_resume()

    # ------------------------------------------------------------------ checkpointing
    def _progress_path(self) -> Path:
        return self.ckpt_dir / "progress.pt"

    def _maybe_resume(self):
        p = self._progress_path()
        if p.exists():
            ckpt = torch.load(p, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if self.use_scaler and "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.start_step = ckpt["step"]
            print(f"[resume] loaded {p}, continuing from step {self.start_step}")

    def save_checkpoint(self, step: int, extra: dict = None, final: bool = False):
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step": step,
            "cfg": self.cfg,
        }
        if self.use_scaler:
            payload["scaler_state_dict"] = self.scaler.state_dict()
        if extra:
            payload.update(extra)

        torch.save(payload, self._progress_path())  # always overwrite the resumable "latest"
        if final or (self.cfg.save_every and step % self.cfg.save_every == 0):
            named = self.ckpt_dir / f"step_{step}.pt"
            torch.save(payload, named)
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
            g["lr"] = lr
        return lr

    def training_step(self, batches: list):
        """`batches` is a list of grad_accum_steps micro-batches, each
        (input_ids, labels, loss_mask)."""
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for input_ids, labels, loss_mask in batches:
            input_ids, labels = input_ids.to(self.device), labels.to(self.device)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device)
            with torch.autocast(device_type=self.device if self.device != "mps" else "cpu",
                                 dtype=self.dtype, enabled=self.dtype != torch.float32):
                _, loss = self.model(input_ids, labels=labels, loss_mask=loss_mask)
            loss = loss / len(batches)
            if self.use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

        if self.use_scaler:
            self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

        if self.use_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        return total_loss, grad_norm.item()

    def time_budget_exceeded(self) -> bool:
        if self.cfg.time_budget_minutes is None or self.start_time is None:
            return False
        return (time.time() - self.start_time) / 60.0 >= self.cfg.time_budget_minutes

    def fit(self, train_batch_fn, val_batch_fn=None, on_log=None, on_eval=None):
        """train_batch_fn() -> (input_ids, labels, loss_mask), called
        grad_accum_steps times per training_step. val_batch_fn, if given, is
        called cfg.eval_batches times at each eval_every interval.
        on_log(step, loss, lr, grad_norm) and on_eval(step, val_loss) are
        optional callbacks -- keeps this loop free of any assumption about
        HOW results get reported (print, a logger, a notebook cell, etc)."""
        self.start_time = time.time()
        step = self.start_step
        stopped_early = False

        while step < self.cfg.max_steps:
            if self.time_budget_exceeded():
                print(f"[time budget] {self.cfg.time_budget_minutes} min reached at step {step}, "
                      f"stopping cleanly and checkpointing.")
                stopped_early = True
                break

            lr = self.set_lr(step)
            batches = [train_batch_fn() for _ in range(self.cfg.grad_accum_steps)]
            loss, grad_norm = self.training_step(batches)
            step += 1

            if on_log and step % self.cfg.log_every == 0:
                on_log(step, loss, lr, grad_norm)

            if val_batch_fn and on_eval and step % self.cfg.eval_every == 0:
                val_loss = self._evaluate(val_batch_fn)
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

        self.save_checkpoint(step, final=True)
        return {"final_step": step, "stopped_early": stopped_early}

    @torch.no_grad()
    def _evaluate(self, val_batch_fn):
        self.model.eval()
        losses = []
        for _ in range(self.cfg.eval_batches):
            input_ids, labels, loss_mask = val_batch_fn()
            input_ids, labels = input_ids.to(self.device), labels.to(self.device)
            if loss_mask is not None:
                loss_mask = loss_mask.to(self.device)
            with torch.autocast(device_type=self.device if self.device != "mps" else "cpu",
                                 dtype=self.dtype, enabled=self.dtype != torch.float32):
                _, loss = self.model(input_ids, labels=labels, loss_mask=loss_mask)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)
