"""Model tournament: find the best architecture for Malayalam on your hardware. Phases (run in this order):

  list      show the candidates in a plan
  check     [CPU ok]   build every candidate at full size, count parameters, 1 forward/backward  -> catches config/package errors
  smoke     [CPU ok]   train every candidate at tiny size, check the loss falls, stop/resume     -> proves the pipeline works
  prepare   [CPU]      build the tokenised data cache ONCE (all runs share it)
  probe     [1 GPU]    tokens/s, peak memory, best micro-batch and longest context for every candidate
  budget    [-]        how many tokens per run fit in --hours given the probe speeds
  race      [GPUs]     one run per candidate per GPU, equal tokens / batch / schedule, resumable
  report    [-]        ranking table: quality at equal tokens and equal wall-clock, speed, memory, stability

  python scripts/tournament.py --plan configs/tournament/smoke_kaggle.json --phase check
  python scripts/tournament.py --plan configs/tournament/race1_screen.json --phase race --gpus 0,1,2,3,4,5
GPU 7 is refused (DGX policy). Re-running a phase skips what is already finished.
"""
import argparse
import json
import os
import sys

import _bootstrap  # noqa: F401
from qwenlab.tournament.plan import check_gpus, expand_jobs, job_overrides, load_plan, suggest_tokens
from qwenlab.tournament.report import build_report, render_markdown, render_position_table, write_csv
from qwenlab.tournament.runner import run_race

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--phase", required=True, choices=["list", "check", "smoke", "prepare", "probe", "budget", "race", "report"])
ap.add_argument("--gpus", default=None, help="comma-separated GPU ids for probe/race (default: the plan's list)")
ap.add_argument("--hours", type=float, default=20.0, help="budget phase: total wall-clock hours available")
ap.add_argument("--dry-run", action="store_true", help="race: print the commands only")
ap.add_argument("--smoke-config", default=os.path.join(HERE, "..", "configs", "smoke_cpu.json"))
ap.add_argument("--vocab", type=int, default=0, help="check: pretend the tokenizer has this many tokens (e.g. 32000) when the real one is not available yet")
a = ap.parse_args()

plan = load_plan(a.plan)
gpus = [int(x) for x in a.gpus.split(",")] if a.gpus else plan["gpus"]
if a.phase in ("probe", "race"):
    check_gpus(gpus, plan["forbidden_gpus"])
if a.phase == "probe":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])          # before torch is imported

root = os.path.join(plan["runs_root"], plan["name"])
os.makedirs(root, exist_ok=True)

if a.phase == "list":
    for j in expand_jobs(plan):
        print(j["id"], "|", " ".join(job_overrides(plan, j)[:len(plan["overrides"]) + len(j["cand"].get("overrides", []))]))

elif a.phase == "check":
    from qwenlab.tournament.check import check_models
    rows = check_models(plan, plan["configs"], a.vocab)
    json.dump(rows, open(os.path.join(root, "check.json"), "w"), indent=1)
    bad = [r["id"] for r in rows if r["status"] != "ok"]
    print("\nALL CANDIDATES OK" if not bad else f"\nPROBLEMS: {bad}")
    sys.exit(1 if bad else 0)

elif a.phase == "smoke":
    from qwenlab.tournament.check import smoke_train
    rows = smoke_train(plan, a.smoke_config)
    json.dump(rows, open(os.path.join(root, "smoke.json"), "w"), indent=1)
    bad = [r["id"] for r in rows if r["status"] != "ok"]
    print("\nSMOKE OK" if not bad else f"\nPROBLEMS: {bad}")
    sys.exit(1 if bad else 0)

elif a.phase == "prepare":
    from qwenlab.config import load_config
    from qwenlab.data.pretrain_data import build_pretrain_data
    from qwenlab.data.tokenizers import build_tokenizer
    cfg = load_config(plan["configs"], job_overrides(plan, expand_jobs(plan)[0]))
    data = build_pretrain_data(cfg.data, cfg.tokenizer, build_tokenizer(cfg.tokenizer))
    print(json.dumps(data.stats, indent=1))
    n = data.stats["train_tokens"]
    print(f"\ntrain tokens available: {n:,}. A run of {plan['tokens']:,} tokens is {plan['tokens'] / n:.2f} epoch(s).")

elif a.phase == "probe":
    from qwenlab.tournament.probe import probe_candidate
    for cand in plan["candidates"]:
        probe_candidate(plan, cand, plan["configs"], os.path.join(root, "probe"))
    print("probe results in", os.path.join(root, "probe"))

elif a.phase == "budget":
    speeds = {}
    for cand in plan["candidates"]:
        pj = os.path.join(root, "probe", f"{cand['id']}.json")
        if not os.path.exists(pj):
            print("missing probe for", cand["id"]); continue
        info = json.load(open(pj))
        mb = info["best_micro_batch"].get(str(plan["seq_len"]))
        row = next((r for r in info["by_mb"] if r["mb"] == mb and r["status"] == "ok"), None)
        if row:
            speeds[cand["id"]] = row["tok_s"]
            print(f"{cand['id']:18s} micro-batch {mb:3d}  {row['tok_s']:>10,.0f} tok/s  peak {row['peak_gb']:.1f} GB  max context {info.get('max_context_plain')}/{info.get('max_context_with_checkpointing')} (plain/ckpt)")
    if speeds:
        n_jobs = len(expand_jobs(plan))
        s = suggest_tokens(speeds, n_jobs, len(gpus), a.hours)
        print(f"\n{n_jobs} jobs on {len(gpus)} GPUs = {s['waves']} wave(s) of {s['hours_per_wave']:.1f} h. "
              f"Tokens per run that fit: {s['tokens_per_run']:,} (limited by {s['limited_by']}).")
        print("Set \"tokens\" in the plan to at most this value (a multiple of batch_tokens).")

elif a.phase == "race":
    st = run_race(plan, gpus, os.path.join(HERE, "train.py"), a.dry_run)
    json.dump(st, open(os.path.join(root, "race_status.json"), "w"), indent=1)
    print(json.dumps(st, indent=1))

elif a.phase == "report":
    agg = build_report(plan["name"], plan["runs_root"], [j["id"] for j in expand_jobs(plan)],
                       plan["report"].get("noise", 0.004))
    md = render_markdown(agg) + render_position_table(agg)
    open(os.path.join(root, "report.md"), "w", encoding="utf-8").write(md)
    write_csv(agg, os.path.join(root, "report.csv"))
    print(md)
