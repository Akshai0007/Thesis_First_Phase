"""GPU probe (needs torch + a GPU): for each candidate, tokens/s and peak memory at several micro-batch sizes, and the
longest context that fits with and without gradient checkpointing. Uses random token ids - it measures speed and
memory, not quality. Writes <runs_root>/<plan>/probe/<candidate>.json, which the race phase reads to pick micro-batches."""
import gc
import json
import os
import time
from typing import Any, Dict, List

import torch


def _train_steps(model, cfg, seq_len, mb, steps, device, dtype, vocab):
    from ..training.trainer import build_optimizer
    opt = build_optimizer(model, cfg.train)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    fwd = torch.compile(model) if cfg.train.compile else model
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    for i in range(steps + 3):
        if i == 3:
            torch.cuda.synchronize(); t0 = time.perf_counter()
        x = torch.randint(0, vocab, (mb, seq_len + 1), device=device)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            _, loss = fwd(x[:, :-1], labels=x[:, 1:])
        scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
        if scaler.is_enabled():
            scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if scaler.is_enabled():
            scaler.step(opt); scaler.update()
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return steps * mb * seq_len / (time.perf_counter() - t0), torch.cuda.max_memory_allocated() / 2**30


def _try(cfg, tok, seq_len, mb, grad_ckpt, steps, device, dtype):
    import dataclasses
    from ..models.factory import build_model
    mcfg_in = dataclasses.replace(cfg.model, gradient_checkpointing=grad_ckpt)
    model = None
    try:
        model, mcfg = build_model(mcfg_in, cfg.init, tok, context_len=seq_len)
        model.to(device).train()
        tok_s, peak = _train_steps(model, cfg, seq_len, mb, steps, device, dtype, mcfg.vocab_size)
        return {"seq_len": seq_len, "mb": mb, "grad_ckpt": grad_ckpt, "status": "ok", "tok_s": tok_s, "peak_gb": peak,
                "n_params": sum(p.numel() for p in model.parameters())}
    except torch.cuda.OutOfMemoryError:
        return {"seq_len": seq_len, "mb": mb, "grad_ckpt": grad_ckpt, "status": "oom"}
    except Exception as e:
        return {"seq_len": seq_len, "mb": mb, "grad_ckpt": grad_ckpt, "status": f"error: {type(e).__name__}: {str(e)[:160]}"}
    finally:
        del model
        gc.collect(); torch.cuda.empty_cache()


def probe_candidate(plan: Dict[str, Any], cand: Dict[str, Any], cfg_paths: List[str], out_dir: str, mem_cap: float = 0.9,
                    ladder=(2048, 4096, 8192, 16384, 32768), mbs=(1, 2, 4, 8, 16, 32, 64), steps: int = 5, log=print) -> dict:
    from ..config import load_config
    from ..data.tokenizers import build_tokenizer
    from ..training.trainer import resolve_dtype
    cfg = load_config(cfg_paths, plan["overrides"] + cand.get("overrides", []))
    tok = build_tokenizer(cfg.tokenizer)
    device = "cuda"
    dtype = resolve_dtype(cfg.train.precision, device)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    seq = int(plan["seq_len"])
    info: Dict[str, Any] = {"id": cand["id"], "gpu": torch.cuda.get_device_name(0), "total_gb": total_gb,
                            "dtype": str(dtype), "by_mb": [], "ladder": [], "best_micro_batch": {}}

    best, best_tok_s = None, -1.0
    for mb in mbs:                                         # micro-batch sweep at the training sequence length
        r = _try(cfg, tok, seq, mb, False, steps, device, dtype)
        info["by_mb"].append(r); log(cand["id"], r)
        if r["status"] != "ok":
            break
        if r["peak_gb"] <= mem_cap * total_gb and r["tok_s"] > best_tok_s * 1.02:     # >2 % faster than the smaller batch
            best, best_tok_s = mb, r["tok_s"]
        info["n_params"] = r["n_params"]
    if best is not None:
        info["best_micro_batch"][str(seq)] = best

    for L in ladder:                                       # longest context: plain, then with gradient checkpointing
        r = _try(cfg, tok, L, 1, False, 3, device, dtype)
        if r["status"] != "ok":
            r2 = _try(cfg, tok, L, 1, True, 3, device, dtype)
            info["ladder"] += [r, r2]; log(cand["id"], r, r2)
            if r2["status"] != "ok":
                break
        else:
            info["ladder"].append(r); log(cand["id"], r)
        ok_mb = None
        for mb in (8, 4, 2, 1):                            # micro-batch to use at this length (plain mode)
            rr = _try(cfg, tok, L, mb, False, 3, device, dtype)
            if rr["status"] == "ok" and rr["peak_gb"] <= mem_cap * total_gb:
                ok_mb = mb
                break
        if ok_mb:
            info["best_micro_batch"][str(L)] = ok_mb
    oks = [r["seq_len"] for r in info["ladder"] if r["status"] == "ok"]
    info["max_context_plain"] = max([r["seq_len"] for r in info["ladder"] if r["status"] == "ok" and not r["grad_ckpt"]], default=None)
    info["max_context_with_checkpointing"] = max(oks, default=None)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{cand['id']}.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info
