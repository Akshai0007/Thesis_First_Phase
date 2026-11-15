"""Turn the runs of a tournament into a decision table. No torch. Reads each run's metrics.jsonl and results.json.

Views (a single number hides trade-offs, so there are several):
  * final quality          : validation bits-per-byte (or loss) at the end of the equal-token budget
  * equal-token curve      : quality at 25 % / 50 % / 100 % of the tokens every candidate reached
  * equal-wall-clock       : quality at the training time the SLOWEST candidate used (what you get for the same GPU hours)
  * speed / memory         : median tokens/s, MFU, peak GPU memory
  * stability              : largest gradient norm, number of loss spikes, failures
  * context use            : loss in the first vs last position bucket of the window (does the model use its context?)
Decision rule: best final quality wins; everything within `noise` of it is a tie and the FASTEST of the tied candidates is
recommended. `noise` is the seed-to-seed spread when a candidate has >= 2 seeds, otherwise plan["report"]["noise"]
(default 0.004 BPB: replace it by measuring - run the baseline with two seeds)."""
import csv
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional


def read_jsonl(path: str) -> List[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass                                   # a half-written last line after a crash
    return rows


def interp(xs: List[float], ys: List[float], x: float) -> Optional[float]:
    """Linear interpolation; None when x lies outside the measured range."""
    if not xs or x < xs[0] - 1e-9 or x > xs[-1] + 1e-9:
        return None
    for i in range(1, len(xs)):
        if x <= xs[i] + 1e-9:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return y1 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return ys[-1]


def count_spikes(losses: List[float], window: int = 10, factor: float = 1.5, skip: int = 20) -> int:
    """Training-loss values more than `factor` x the median of the previous `window` logged values (after warm-up)."""
    n = 0
    for i in range(max(window, skip), len(losses)):
        if losses[i] > factor * statistics.median(losses[i - window:i]):
            n += 1
    return n


def load_run(run_dir: str) -> Dict[str, Any]:
    rows = read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    train = [r for r in rows if "train_loss" in r]
    evals = [r for r in rows if "val_loss" in r and r.get("step", 0) > 0]
    res_path = os.path.join(run_dir, "results.json")
    results = json.load(open(res_path, encoding="utf-8")) if os.path.exists(res_path) else None
    return {"dir": run_dir, "train": train, "evals": evals, "results": results}


def summarize_run(run: Dict[str, Any]) -> Dict[str, Any]:
    train, evals, res = run["train"], run["evals"], run["results"]
    use_bpb = bool(evals) and all("val_bpb" in e for e in evals)
    key = "val_bpb" if use_bpb else "val_loss"
    out: Dict[str, Any] = {"metric": key, "finished": res is not None, "failed": res is None and not evals}
    if evals:
        out["final"] = evals[-1][key]
        out["tokens"] = evals[-1].get("tokens_seen", 0)
        out["train_time_s"] = evals[-1].get("train_time_s", 0.0)
        out["curve_tokens"] = [e.get("tokens_seen", 0) for e in evals]
        out["curve_time"] = [e.get("train_time_s", 0.0) for e in evals]
        out["curve_val"] = [e[key] for e in evals]
    if train:
        skip = max(3, len(train) // 10)                    # compile / cudnn warm-up rows are not representative
        body = train[skip:] or train
        toks = [r["tok_s"] for r in body if "tok_s" in r]
        out["tok_s"] = statistics.median(toks) if toks else None
        mfu = [r["mfu"] for r in body if "mfu" in r]
        out["mfu"] = statistics.median(mfu) if mfu else None
        mem = [r["peak_mem_gb"] for r in train if "peak_mem_gb" in r]
        out["peak_mem_gb"] = max(mem) if mem else None
        out["max_grad_norm"] = max(r["grad_norm"] for r in train)
        out["loss_spikes"] = count_spikes([r["train_loss"] for r in train])
    if res:
        out["n_params"] = res.get("n_params")
        out["stopped_early"] = res.get("stopped_early")
        pos = res.get("val_loss_by_position") or {}
        buckets = [k for k in pos if k != "overall"]
        if len(buckets) >= 2:
            out["pos_first"], out["pos_last"] = pos[buckets[0]], pos[buckets[-1]]
            out["ctx_gain"] = pos[buckets[0]] - pos[buckets[-1]]
            out["pos"] = {k: v for k, v in pos.items() if k != "overall"}
    return out


def _mean_std(vals: List[float]):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else None)


def group_key(job_id: str) -> str:
    return job_id.rsplit("_s", 1)[0]                       # drop the seed suffix


def aggregate(runs: Dict[str, Dict[str, Any]], noise_default: float = 0.004) -> Dict[str, Any]:
    """runs: {job_id: summary}. Returns {"groups": {...}, "best": ..., "recommended": ..., "milestones": ...}."""
    by_group: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for jid, s in runs.items():
        by_group[group_key(jid)].append(s)
    ok = {g: [s for s in ss if "final" in s] for g, ss in by_group.items()}
    ok = {g: ss for g, ss in ok.items() if ss}
    if not ok:
        return {"groups": {}, "best": None, "recommended": None, "note": "no run has produced an evaluation yet"}

    min_tokens = min(min(s["tokens"] for s in ss) for ss in ok.values())
    min_time = min(min(s["train_time_s"] for s in ss) for ss in ok.values())
    marks = [round(min_tokens * f) for f in (0.25, 0.5, 1.0)]

    groups = {}
    for g, ss in ok.items():
        row: Dict[str, Any] = {"n_runs": len(ss), "n_failed": len(by_group[g]) - len(ss), "metric": ss[0]["metric"]}
        row["final"], row["final_std"] = _mean_std([s["final"] for s in ss])
        for m in marks:
            vals = [interp(s["curve_tokens"], s["curve_val"], m) for s in ss]
            row[f"at_{m}"], _ = _mean_std(vals)
        row["at_equal_time"], _ = _mean_std([interp(s["curve_time"], s["curve_val"], min_time) for s in ss])
        for k in ("tok_s", "mfu", "peak_mem_gb", "max_grad_norm", "loss_spikes", "n_params", "ctx_gain", "pos_first", "pos_last"):
            row[k], _ = _mean_std([s.get(k) for s in ss])
        pos_dicts = [s["pos"] for s in ss if s.get("pos")]
        if pos_dicts:                                     # mean loss per position bucket across seeds
            row["pos"] = {k: statistics.mean(d[k] for d in pos_dicts if k in d) for k in pos_dicts[0]}
        row["tokens"] = min(s["tokens"] for s in ss)
        row["stopped_early"] = any(s.get("stopped_early") for s in ss)
        groups[g] = row

    best = min(groups, key=lambda g: groups[g]["final"])
    stds = [r["final_std"] for r in groups.values() if r["final_std"] is not None]
    noise = max(stds) if stds else noise_default
    tied = [g for g in groups if groups[g]["final"] - groups[best]["final"] <= noise]
    speeds = {g: groups[g]["tok_s"] or 0.0 for g in tied}
    recommended = max(tied, key=lambda g: speeds[g])
    return {"groups": groups, "best": best, "tied": tied, "recommended": recommended, "noise": noise,
            "noise_source": "seed spread" if stds else "default (measure it: run the baseline with 2 seeds)",
            "milestones": {"tokens": marks, "equal_time_s": min_time}}


def _f(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def render_markdown(agg: Dict[str, Any]) -> str:
    if not agg.get("groups"):
        return "No finished evaluations yet.\n"
    g = agg["groups"]
    order = sorted(g, key=lambda k: g[k]["final"])
    metric = g[order[0]]["metric"]
    marks = agg["milestones"]["tokens"]
    lines = [f"## Tournament result (metric: {metric}, lower is better)\n",
             f"Equal-token marks: {', '.join(f'{m/1e6:.0f}M' for m in marks)} tokens; equal wall-clock mark: "
             f"{agg['milestones']['equal_time_s']/3600:.2f} h of training.\n",
             f"Noise used for ties: {agg['noise']:.4f} ({agg['noise_source']}).\n",
             "| rank | candidate | params (M) | final | " + " | ".join(f"@{m/1e6:.0f}M" for m in marks[:-1]) +
             " | @equal time | tok/s | MFU | peak GB | max grad-norm | spikes | ctx gain | runs |",
             "|---" * (11 + len(marks) - 1) + "|"]
    for i, k in enumerate(order, 1):
        r = g[k]
        mark = " **(best)**" if k == agg["best"] else ""
        mid = " | ".join(_f(r.get(f"at_{m}")) for m in marks[:-1])
        lines.append(
            f"| {i} | {k}{mark} | {_f((r['n_params'] or 0)/1e6, 1)} | {_f(r['final'])} | {mid} | {_f(r['at_equal_time'])} | "
            f"{_f(r['tok_s'], 0)} | {_f(r['mfu'], 3)} | {_f(r['peak_mem_gb'], 1)} | {_f(r['max_grad_norm'], 2)} | "
            f"{_f(r['loss_spikes'], 0)} | {_f(r['ctx_gain'], 3)} | {r['n_runs']}" + (f" (+{r['n_failed']} failed)" if r["n_failed"] else "") + " |")
    lines += ["",
              f"**Best final quality:** {agg['best']}.  **Tied within noise:** {', '.join(agg['tied'])}.",
              f"**Recommendation (fastest among the tied):** {agg['recommended']}.",
              "",
              "Read it as: quality at equal tokens says which design learns more per token; quality at equal wall-clock says which is "
              "better per GPU hour; peak GB and max context (probe phase) say what fits; spikes / grad-norm warn about instability "
              "at the learning rate used (a design that only wins at a lower LR is not a win)."]
    return "\n".join(lines) + "\n"


def render_position_table(agg: Dict[str, Any]) -> str:
    """Loss by position bucket per candidate (context-window study). Compare buckets that EXIST for every run:
    the first buckets are the same positions in every window length, so a longer-context model should show a lower loss
    in the late buckets of the shorter one (e.g. 1024-2048) if the extra context helps."""
    g = {k: v for k, v in agg.get("groups", {}).items() if v.get("pos")}
    if not g:
        return ""
    keys = []
    for v in g.values():
        for b in v["pos"]:
            if b not in keys:
                keys.append(b)
    keys.sort(key=lambda b: int(b.split("-")[0]))
    lines = ["\n## Loss by position inside the window (nats/token)\n",
             "| candidate | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
    for name in sorted(g):
        lines.append(f"| {name} | " + " | ".join(_f(g[name]["pos"].get(b), 3) for b in keys) + " |")
    return "\n".join(lines) + "\n"


def write_csv(agg: Dict[str, Any], path: str) -> None:
    if not agg.get("groups"):
        return
    rows = [{"candidate": k, **v} for k, v in agg["groups"].items()]
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def build_report(plan_name: str, runs_root: str, job_ids: List[str], noise_default: float = 0.004) -> Dict[str, Any]:
    summaries = {}
    for jid in job_ids:
        d = os.path.join(runs_root, plan_name, jid)
        if os.path.isdir(d):
            summaries[jid] = summarize_run(load_run(d))
    return aggregate(summaries, noise_default)
