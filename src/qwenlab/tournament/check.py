"""Pre-flight checks (need torch). Run these on Kaggle / CPU BEFORE any DGX time is spent.

  check_models : build every candidate at FULL size on CPU, count parameters, run one forward + backward pass on a few
                 tokens and verify the untrained loss is about ln(vocab). Catches wrong Hugging Face field names,
                 silently ignored settings, shape bugs and missing packages for each architecture.
  smoke_train  : train every candidate at TINY size for ~60 steps on the sample data (byte tokenizer, CPU or GPU),
                 check the loss falls, then stop / resume / finish to prove checkpointing works with that architecture."""
import copy
import json
import math
import os
import tempfile
import time
from typing import Any, Dict, List


def _apply(cfg_paths: List[str], overrides: List[str]):
    from ..config import load_config
    return load_config(cfg_paths, overrides)


def check_models(plan: Dict[str, Any], cfg_paths: List[str], vocab_override: int = 0, log=print) -> List[dict]:
    import torch
    from ..data.tokenizers import build_tokenizer
    from ..models.factory import build_model
    rows = []
    for cand in plan["candidates"]:
        row = {"id": cand["id"]}
        t0 = time.time()
        try:
            cfg = _apply(cfg_paths, plan["overrides"] + cand.get("overrides", []))
            if vocab_override:
                class _T:                                  # stand-in tokenizer of the right size (real one not ready yet)
                    eos_id, pad_id = 0, 0
                    def __len__(self): return vocab_override
                tok = _T()
            else:
                tok = build_tokenizer(cfg.tokenizer)
            model, mcfg = build_model(cfg.model, cfg.init, tok, context_len=min(plan["seq_len"], 256))
            n = sum(p.numel() for p in model.parameters())
            x = torch.randint(0, mcfg.vocab_size, (1, 64))
            _, loss = model(x, labels=torch.roll(x, -1, 1))
            loss.backward()
            bad = [k for k, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
            row.update(status="ok", params_M=round(n / 1e6, 1), loss=round(float(loss), 3),
                       ln_vocab=round(math.log(mcfg.vocab_size), 3), nonfinite_grads=len(bad),
                       seconds=round(time.time() - t0, 1))
            if abs(float(loss) - math.log(mcfg.vocab_size)) > 1.5 or bad:
                row["status"] = "SUSPICIOUS (untrained loss far from ln(vocab) or non-finite gradients)"
            del model
        except Exception as e:                             # one broken architecture must not hide the others
            row.update(status=f"FAILED: {type(e).__name__}: {str(e)[:300]}")
        log(row)
        rows.append(row)
    by_id = {r["id"]: r for r in rows}
    if "qwen2_custom" in by_id and "qwen2_hf" in by_id and by_id["qwen2_custom"].get("params_M"):
        same = by_id["qwen2_custom"].get("params_M") == by_id["qwen2_hf"].get("params_M")
        log(f"[parity] hand-written Qwen vs Hugging Face Qwen2: parameter counts {'MATCH' if same else 'DIFFER (investigate)'}")
    return rows


def smoke_train(plan: Dict[str, Any], base_smoke_cfg: str, steps: int = 60, log=print) -> List[dict]:
    from ..pipeline import load_experiment, run_training
    tiny = ["model.hidden_size=64", "model.intermediate_size=128", "model.num_layers=2", "model.num_heads=4",
            "model.num_kv_heads=2", "model.loss_chunk_tokens=0", "model.vocab_pad_multiple=8"]
    rows = []
    for cand in plan["candidates"]:
        row = {"id": cand["id"]}
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ov = [o for o in cand.get("overrides", []) if not o.startswith(("model.hidden_size", "model.intermediate_size",
                      "model.num_layers", "model.num_heads", "model.num_kv_heads"))]
                common = tiny + ov + [f"train.output_dir={tmp}/run", f"data.cache_dir={tmp}/cache", f"train.max_steps={steps}",
                                      f"train.stop_at_step={steps // 2}", f"train.save_every={steps // 2}", f"train.eval_every={steps // 2}",
                                      "train.precision=fp32", "train.time_budget_minutes=null", "data.seq_len=64"]
                cfg1 = load_experiment([base_smoke_cfg], common)
                first = run_training(cfg1, log=lambda *a: None)          # pauses at steps // 2
                cfg2 = load_experiment([base_smoke_cfg], [c for c in common if not c.startswith("train.stop_at_step")])
                final = run_training(cfg2, log=lambda *a: None)          # resumes and finishes
                lines = [json.loads(l) for l in open(f"{tmp}/run/metrics.jsonl")]
                evals = [r["val_loss"] for r in lines if "val_loss" in r]
                row.update(status="ok", paused_at=first["final_step"], finished_at=final["final_step"],
                           loss_start=round(evals[0], 3), loss_end=round(evals[-1], 3))
                if not (evals[-1] < evals[0] and final["final_step"] == steps and first["final_step"] == steps // 2):
                    row["status"] = "SUSPICIOUS (loss did not fall, or pause/resume steps are wrong)"
        except Exception as e:
            row.update(status=f"FAILED: {type(e).__name__}: {str(e)[:300]}")
        log(row)
        rows.append(row)
    return rows
