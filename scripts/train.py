"""Train from scratch on continuous text.

    python scripts/train.py --config configs/base.json configs/english.json
    python scripts/train.py --config configs/base.json configs/malayalam.json
    python scripts/train.py --config configs/base.json configs/english.json --set train.max_steps=2000

Later config files override earlier ones. Re-running the same command resumes from the last checkpoint.
"""
import argparse

import _bootstrap  # noqa: F401
from qwenlab.pipeline import load_experiment, run_training

ap = argparse.ArgumentParser()
ap.add_argument("--config", nargs="+", required=True, help="one or more JSON configs (later ones override earlier ones)")
ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.max_steps=500 data.seq_len=256")
a = ap.parse_args()
run_training(load_experiment(a.config, a.set))
