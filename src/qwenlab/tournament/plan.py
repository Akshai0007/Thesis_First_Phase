"""Tournament plan: which candidates, which budget, how each run's config overrides are derived. No torch, stdlib only.

A plan is a JSON file (see configs/tournament/*.json). Every run is an ordinary `scripts/train.py` run, so a tournament
run is resumable, produces the usual metrics.jsonl / results.json, and can be repeated by hand with the printed command."""
import itertools
import json
import math
import os
from typing import Any, Dict, List, Optional

PSEUDO_KEYS = {"seq_len", "batch_tokens", "tokens", "lr"}


def load_plan(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        plan = json.load(f)
    plan = {k: v for k, v in plan.items() if not k.startswith("_")}
    for req in ("name", "configs", "candidates"):
        if req not in plan:
            raise ValueError(f"plan {path}: missing '{req}'")
    ids = [c["id"] for c in plan["candidates"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"plan {path}: duplicate candidate ids {ids}")
    plan.setdefault("seq_len", 2048)
    plan.setdefault("batch_tokens", 524288)
    plan.setdefault("tokens", 300_000_000)
    plan.setdefault("lr", 6e-4)
    plan.setdefault("evals", 12)                       # validation points per run (equal-token comparison)
    plan.setdefault("warmup_frac", 0.03)
    plan.setdefault("decay_fraction", 0.2)
    plan.setdefault("micro_batch_size", 8)             # used when no probe result exists
    plan.setdefault("overrides", [])
    plan.setdefault("grid", {})
    plan.setdefault("seeds", [1])
    plan.setdefault("gpus", [0])
    plan.setdefault("forbidden_gpus", [7])             # DGX policy: GPU 7 belongs to another workload
    plan.setdefault("runs_root", "runs/tournament")
    plan.setdefault("max_minutes_per_job", None)
    plan.setdefault("cleanup_after_run", False)        # true on Kaggle: delete checkpoints of finished runs (20 GB output limit)
    plan.setdefault("save_every", 500)                 # local checkpoint every N steps (crash / session-loss insurance)
    plan.setdefault("hub_repo", None)                  # e.g. "user/malayalam-lm-tournament": checkpoints + results are backed up per run
    plan.setdefault("hub_upload_every_n_saves", 2)     # a full checkpoint is ~6 GB: upload every 2nd save and always at the end
    plan.setdefault("report", {})
    return plan


def check_gpus(gpus: List[int], forbidden: List[int]) -> None:
    bad = sorted(set(gpus) & set(forbidden))
    if bad:
        raise ValueError(f"GPU(s) {bad} are reserved (DGX policy: do not use GPU 7). Remove them from --gpus / the plan.")


def expand_jobs(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """candidates x grid x seeds -> list of jobs with stable ids."""
    keys = list(plan["grid"])
    combos = list(itertools.product(*[plan["grid"][k] for k in keys])) if keys else [()]
    jobs = []
    for cand in plan["candidates"]:
        for combo in combos:
            gv = dict(zip(keys, combo))
            for seed in plan["seeds"]:
                tag = "".join(f"_{k.split('.')[-1]}{str(v).replace('/', '-')}" for k, v in gv.items())
                jobs.append({"id": f"{cand['id']}{tag}_s{seed}", "cand": cand, "grid": gv, "seed": seed})
    return jobs


def choose_micro_batch(best_mb: int, batch_tokens: int, seq_len: int) -> int:
    """Largest micro-batch <= best_mb with micro_batch * seq_len dividing the global batch (grad accumulation fills the rest)."""
    mb = max(int(best_mb), 1)
    while mb >= 1:
        if batch_tokens % (mb * seq_len) == 0:
            return mb
        mb -= 1
    raise ValueError(f"batch_tokens={batch_tokens} is not a multiple of seq_len={seq_len}")


def job_overrides(plan: Dict[str, Any], job: Dict[str, Any], probe: Optional[dict] = None,
                  micro_batch: Optional[int] = None) -> List[str]:
    """The `--set` list for one run. Same tokens / batch / schedule for every candidate; only the model (and
    micro-batch, which is a memory detail: tokens per optimizer step stay identical) differs."""
    gv, cand = job["grid"], job["cand"]

    def pick(k):                                       # grid value > candidate value > plan default
        return gv.get(k, cand.get(k, plan[k]))
    seq, batch_tokens, tokens, lr = int(pick("seq_len")), int(pick("batch_tokens")), int(pick("tokens")), float(pick("lr"))
    if micro_batch is None:
        best = plan["micro_batch_size"]
        if probe and str(seq) in probe.get("best_micro_batch", {}):
            best = probe["best_micro_batch"][str(seq)]
        micro_batch = best
    mb = choose_micro_batch(micro_batch, batch_tokens, seq)
    accum = batch_tokens // (mb * seq)
    steps = max(tokens // batch_tokens, 1)
    warm = max(int(plan["warmup_frac"] * steps), 5)
    out_dir = os.path.join(plan["runs_root"], plan["name"], job["id"])
    ov = list(plan["overrides"]) + list(job["cand"].get("overrides", []))
    ov += [
        f"data.seq_len={seq}", f"train.micro_batch_size={mb}", f"train.grad_accum_steps={accum}",
        f"train.max_steps={steps}", f"train.warmup_steps={warm}", f"train.decay_fraction={plan['decay_fraction']}",
        f"train.lr={lr}", f"train.seed={job['seed']}", f"init.seed={job['seed']}",
        f"train.eval_every={max(steps // plan['evals'], 1)}", f"train.save_every={min(plan['save_every'], steps)}",
        "train.keep_last=1", f"train.output_dir={out_dir}", f"run_name={job['id']}",
        f"data.cache_dir={os.path.join(plan['runs_root'], 'data_cache')}",          # shared by every plan: the cache key already covers source/tokenizer/cleaning
    ]
    if plan.get("max_minutes_per_job"):
        ov.append(f"train.time_budget_minutes={plan['max_minutes_per_job']}")
    if plan.get("hub_repo"):                           # one repo for the whole tournament, one folder per run
        ov += [f"train.hub_repo_id={json.dumps(plan['hub_repo'])}", f"train.hub_path_prefix={json.dumps('tournament/' + plan['name'] + '/' + job['id'])}",
               f"train.hub_upload_every_n_saves={plan['hub_upload_every_n_saves']}"]
    for k, v in gv.items():
        if k not in PSEUDO_KEYS:
            ov.append(f"{k}={json.dumps(v)}")
    return ov


def suggest_tokens(tok_s: Dict[str, float], n_jobs: int, n_gpus: int, hours: float = 20.0,
                   overhead: float = 1.12) -> Dict[str, Any]:
    """Tokens per run so that ALL jobs finish within `hours` when each GPU runs one job at a time.
    Equal-token fairness means the SLOWEST candidate sets the budget. `overhead` covers evals, saves, compile."""
    waves = math.ceil(n_jobs / max(n_gpus, 1))
    per_wave_s = hours * 3600 / waves
    per_cand = {k: v * per_wave_s / overhead for k, v in tok_s.items() if v > 0}
    slowest = min(per_cand, key=per_cand.get)
    return {"waves": waves, "hours_per_wave": hours / waves, "tokens_per_run": int(per_cand[slowest]),
            "limited_by": slowest, "tokens_each_could_do": {k: int(v) for k, v in per_cand.items()}}
