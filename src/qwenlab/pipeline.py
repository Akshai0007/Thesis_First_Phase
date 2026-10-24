"""The whole English/Malayalam/storytelling run as four small, separate steps. Every step reads ONLY the config:

    1. tokenizer  <- cfg.tokenizer          (swap: tokenizer.name / tokenizer.path)
    2. data       <- cfg.data               (swap: data.source.kind / data.source.params)
    3. model      <- cfg.model + tokenizer  (embedding table is sized from the tokenizer automatically)
    4. train/eval <- cfg.train / cfg.eval   (checkpoint every save_every steps, loss + perplexity on held-out text)

scripts/train.py, scripts/evaluate.py and scripts/generate.py are thin wrappers around these functions.
"""
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig, load_config, save_config
from .data.mixing import ResumableBatcher, make_cycler
from .data.pretrain_data import build_pretrain_data, lm_make_batch
from .data.tokenizers import build_tokenizer
from .evaluation.perplexity import evaluate_lm
from .export import export_hf
from .hub import HubBackup
from .inference.generate import generate_text
from .models.factory import build_model, resolve_model_config
from .training.trainer import Trainer


def validate(cfg: ExperimentConfig) -> ExperimentConfig:
    if cfg.train.stage != "pretrain":
        raise NotImplementedError("scripts/train.py trains on continuous text only (train.stage='pretrain'); "
                                  "question/answer or chat-formatted training is a separate, later stage.")
    assert cfg.data.seq_len >= 8, "data.seq_len (the training context window) is too small"
    assert cfg.generation.max_new_tokens_cap < cfg.data.seq_len, \
        "generation.max_new_tokens_cap must be smaller than the training context window (data.seq_len)"
    return cfg


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_experiment(paths, overrides=None) -> ExperimentConfig:
    return validate(load_config(paths, overrides))


def prepare(cfg: ExperimentConfig, log=print):
    """steps 1-3: tokenizer, data, fresh model."""
    tokenizer = build_tokenizer(cfg.tokenizer)
    data = build_pretrain_data(cfg.data, cfg.tokenizer, tokenizer, log=log)
    model, mcfg = build_model(cfg.model, cfg.init, tokenizer, context_len=cfg.data.seq_len)
    cfg.model = mcfg                                   # the resolved config (real vocab size, real context length)
    return tokenizer, data, model


def init_health(model, mcfg, val_batch_fn, device, log=print):
    """Checks that a from-scratch start is healthy: weights come from a distribution (not a constant) and the
    untrained loss is about ln(vocab) (= guessing uniformly)."""
    emb = getattr(getattr(model, "embed_tokens", None), "weight", None)
    if emb is not None:                                # these two checks are skipped for architectures without this layout
        emb = emb.detach().float()
        sample = emb.flatten()[:1_000_000]
        most_common = torch.unique(sample, return_counts=True)[1].max().item()
        log(f"[init] embedding std {emb.std():.5f} (target 0.02) | most repeated embedding value occurs {most_common}x in 1M weights")
        assert most_common < 50, "weights look constant/repeated - not a random initialisation"
    try:
        o = model.layers[0].self_attn.o_proj.weight.detach().float()
        log(f"[init] o_proj std {o.std():.5f} (expected {0.02 / math.sqrt(2 * mcfg.num_layers):.5f})")
    except (AttributeError, IndexError, TypeError):
        pass
    res = evaluate_lm(model, val_batch_fn, 2, device)
    log(f"[init] untrained loss {res['loss']:.3f} vs ln(vocab)={math.log(mcfg.vocab_size):.3f}")


def data_fingerprint(data) -> str:
    """Short hash of the training tokens. Stored with the run and compared on resume, so a dataset that silently
    changed between sessions stops the run instead of corrupting it."""
    h = hashlib.md5()
    h.update(np.ascontiguousarray(data.train.tokens[:200_000]).tobytes())
    h.update(str(len(data.train.tokens)).encode())
    return h.hexdigest()[:12]


def run_training(cfg: ExperimentConfig, log=print, hub=None) -> dict:
    set_seed(cfg.train.seed)
    out = Path(cfg.train.output_dir); out.mkdir(parents=True, exist_ok=True)
    if hub is None and cfg.train.hub_repo_id:
        hub = HubBackup(cfg.train.hub_repo_id, log=log)
    tokenizer, data, model = prepare(cfg, log)
    save_config(cfg, str(out / "experiment.json"))     # everything needed to rebuild the model / generate later
    if hub and cfg.train.resume:                       # a fresh Kaggle session: pull the last backup first
        hub.restore_file("checkpoints/progress.pt", out / "checkpoints" / "progress.pt")
        hub.restore_file("data_fingerprint.txt", out / "data_fingerprint.txt")
    log(f"[model] {sum(p.numel() for p in model.parameters()):,} parameters | vocab {cfg.model.vocab_size} | "
        f"context {cfg.data.seq_len} | train windows {len(data.train):,} | val windows {len(data.val):,}")

    trainer = Trainer(model, cfg.train)
    device = trainer.device
    fp_file, fp = out / "data_fingerprint.txt", data_fingerprint(data)
    if trainer.start_step > 0 and fp_file.exists() and fp_file.read_text().strip() != fp:
        raise RuntimeError(f"the training data changed since this run started (fingerprint {fp_file.read_text().strip()} "
                           f"-> {fp}); resuming would repeat or skip text. Use the same data, or start a new output_dir.")
    fp_file.write_text(fp)

    # fixed held-out batches, evaluated in full every time so the curve is comparable between evaluations
    bs = cfg.eval.batch_size
    n_val = (len(data.val) // bs) * bs
    assert n_val > 0, "validation set smaller than one batch"
    val_batches = [lm_make_batch([data.val[i + j] for j in range(bs)]) for i in range(0, n_val, bs)]
    trainer.cfg.eval_batches = len(val_batches)
    val_fn = make_cycler(val_batches)

    if trainer.start_step == 0:
        init_health(model, cfg.model, val_fn, device, log)
        if cfg.eval.eval_at_start:
            trainer.evaluate(val_fn, step=0)
    else:
        log(f"[resume] continuing from step {trainer.start_step}")

    # data position is a pure function of the step number -> exact, repeat-free resume
    batcher = ResumableBatcher(data.train, cfg.train.micro_batch_size, lm_make_batch, seed=cfg.train.seed,
                               start_batches=trainer.start_step * cfg.train.grad_accum_steps)
    saves = {"n": 0}

    def on_checkpoint(step, path, final):              # optional Hub backup; failures are logged, never fatal
        saves["n"] += 1
        if hub and (final or saves["n"] % max(cfg.train.hub_upload_every_n_saves, 1) == 0):
            hub.upload_file(path, "checkpoints/progress.pt", f"checkpoint step {step}")
            hub.upload_file(fp_file, "data_fingerprint.txt")
            hub.upload_file(trainer.metrics_path, "metrics.jsonl")
            hub.upload_file(out / "experiment.json", "experiment.json")

    result = trainer.fit(batcher, val_batch_fn=val_fn, on_checkpoint=on_checkpoint)

    final = evaluate_lm(model, val_fn, len(val_batches), device, trainer.dtype)
    log(f"[final] step {result['final_step']}: val loss {final['loss']:.4f} | perplexity {final['perplexity']:.2f}")
    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "step": result["final_step"]},
               out / "model_final.pt")
    samples = {}
    for prompt in cfg.generation.probe_prompts:
        samples[prompt] = generate_text(model, tokenizer, prompt, cfg.data.seq_len, **_gen_kwargs(cfg))
        log(f"[sample] {prompt!r} -> {samples[prompt]!r}")
    if cfg.train.export_hf:
        try:
            diff = export_hf(model, cfg.model, tokenizer, out / "hf_model", log=log)
            result["hf_export_max_logit_diff"] = diff
            if hub:
                hub.upload_folder(out / "hf_model", "hf_model")
        except Exception as e:                         # the trained weights are already saved; report and continue
            log(f"[export] hf_model export FAILED ({type(e).__name__}: {e}); model_final.pt is still saved")
    summary = {**result, "val_loss": final["loss"], "val_perplexity": final["perplexity"], "samples": samples,
               "data_stats": data.stats}
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    if hub:
        hub.upload_file(out / "results.json", "results.json")
    return summary


def _gen_kwargs(cfg: ExperimentConfig) -> dict:
    g = cfg.generation
    return dict(max_new_tokens=g.max_new_tokens, max_new_tokens_cap=g.max_new_tokens_cap, temperature=g.temperature,
                top_k=g.top_k, repetition_penalty=g.repetition_penalty, no_repeat_ngram_size=g.no_repeat_ngram_size,
                max_sentences=g.max_sentences)


def load_trained(run_dir: str, device: str = None, extra_positions: int = 0):
    """Rebuild tokenizer + model from <run_dir>/experiment.json and load <run_dir>/model_final.pt (or a
    checkpoint). Used by evaluate / generate. `extra_positions` lengthens the RoPE table beyond the training
    context (diagnostic only: RoPE has no learned positions, so this is allowed, but those positions were never trained)."""
    cfg = load_config(str(Path(run_dir) / "experiment.json"))
    tokenizer = build_tokenizer(cfg.tokenizer)
    mcfg = resolve_model_config(cfg.model, tokenizer, cfg.data.seq_len + extra_positions)
    from .registry import MODELS
    model = MODELS.get(mcfg.name)(mcfg)
    final, ckpt = Path(run_dir) / "model_final.pt", Path(run_dir) / "checkpoints" / "progress.pt"
    path = final if final.exists() else ckpt
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    model.load_state_dict(state)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return cfg, tokenizer, model.to(device).eval(), device
