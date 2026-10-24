"""Whole pipeline on a tiny CPU model: train, checkpoint, resume exactly, evaluate, generate - config only."""
import json
from pathlib import Path

import torch

from qwenlab.pipeline import load_experiment, load_trained, run_training
from qwenlab.inference.generate import generate_text

ROOT = Path(__file__).resolve().parents[1]


def _overrides(tmp, steps):
    return [f"train.output_dir={tmp}/run", f"data.cache_dir={tmp}/cache", f"train.max_steps={steps}",
            "train.eval_every=10", "train.save_every=10", "train.log_every=5",
            f"data.source.params.pattern={ROOT}/sample_data/stories.jsonl"]


def test_train_resume_evaluate_generate(tmp_path):
    cfgs = [str(ROOT / "configs/smoke_cpu.json")]
    r1 = run_training(load_experiment(cfgs, _overrides(tmp_path, 20)), log=lambda *a: None)
    assert r1["final_step"] == 20
    assert r1["val_perplexity"] < 200          # far below the untrained ~vocab-size perplexity (~264)

    rows = [json.loads(l) for l in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    evals = [r for r in rows if "val_perplexity" in r]
    assert evals[0]["step"] == 0 and evals[-1]["val_perplexity"] < evals[0]["val_perplexity"]   # perplexity logged and falling

    r2 = run_training(load_experiment(cfgs, _overrides(tmp_path, 30)), log=lambda *a: None)   # resumes at 20
    assert r2["final_step"] == 30
    steps = [r["step"] for r in map(json.loads, (tmp_path / "run/metrics.jsonl").read_text().splitlines()) if "train_loss" in r]
    assert steps == sorted(steps) and len(steps) == len(set(steps))        # no step trained twice after resume

    cfg, tok, model, device = load_trained(str(tmp_path / "run"))
    text = generate_text(model, tok, "One morning", cfg.data.seq_len, max_new_tokens=500, max_new_tokens_cap=12)
    assert len(tok.encode(text)) <= 12 + 3
