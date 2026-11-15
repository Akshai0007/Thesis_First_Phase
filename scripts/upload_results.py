"""Upload the FINDINGS of a tournament (report, probe results, every run's metrics/results/config/log, plan, configs) to the Hub.
Checkpoints are not included; they are backed up per run by `hub_repo` in the plan (see docs/TOURNAMENT_RUNBOOK.md).

  python scripts/upload_results.py --plan configs/tournament/race1_screen.json --repo Akzhai/malayalam-lm-tournament
  python scripts/upload_results.py --plan ... --repo ... --subdir account_B      # second Kaggle account / second machine
Token: Kaggle secret HF_TOKEN or the HF_TOKEN environment variable. Safe to re-run: it uploads a fresh snapshot each time.
"""
import argparse
import os

import _bootstrap  # noqa: F401
from qwenlab.tournament.hub_results import upload_results
from qwenlab.tournament.plan import load_plan

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--repo", required=True, help="e.g. Akzhai/malayalam-lm-tournament")
ap.add_argument("--subdir", default="", help='folder inside the repo, e.g. "account_A" / "account_B" / "dgx"')
ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
ap.add_argument("--stage", default="hub_stage")
a = ap.parse_args()
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ok = upload_results(load_plan(a.plan), a.plan, a.repo, a.subdir, not a.public, a.stage, root)
raise SystemExit(0 if ok else 1)
