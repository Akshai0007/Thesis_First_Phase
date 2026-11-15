"""Collect a tournament's FINDINGS into one folder and upload it to the Hugging Face Hub (checkpoints are NOT included:
they are huge and are backed up separately per run by train.hub_repo_id). stdlib only; huggingface_hub is imported lazily.

Layout in the repo (under SUBDIR, e.g. "account_A", if given):
    tournament/<plan>/plan.json  report.md  report.csv  check.json  smoke.json  race_status.json  README.md
    tournament/<plan>/probe/<candidate>.json                       speed / memory / max-context measurements
    tournament/<plan>/runs/<job>/metrics.jsonl results.json experiment.json stdout.log data_fingerprint.txt
    configs/                                                         the exact config files used
"""
import glob
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

SKIP_DIRS = {"checkpoints", "hf_model", "__pycache__", ".hub_tmp"}
SKIP_SUFFIXES = (".pt", ".tmp", ".safetensors", ".npy")


def _ignore(_dir, names):
    return [n for n in names if n in SKIP_DIRS or n.endswith(SKIP_SUFFIXES)]


def _git_commit(src: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", src, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown (not a git checkout)"


def _gpu_info() -> str:
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True,
                                       stderr=subprocess.DEVNULL).strip().replace("\n", "; ")
    except Exception:
        return "no nvidia-smi"


def stage_results(plan: Dict[str, Any], plan_path: str, stage_dir: str, src_root: str = ".", configs_dir: Optional[str] = None) -> Dict[str, Any]:
    """Copy the findings of one plan into stage_dir. Returns a small summary (files, MB)."""
    stage = Path(stage_dir)
    shutil.rmtree(stage, ignore_errors=True)
    base = stage / "tournament" / plan["name"]
    base.mkdir(parents=True)
    root = Path(plan["runs_root"]) / plan["name"]
    shutil.copy(plan_path, base / "plan.json")
    for f in ("report.md", "report.csv", "check.json", "smoke.json", "race_status.json"):
        if (root / f).exists():
            shutil.copy(root / f, base / f)
    if (root / "probe").exists():
        shutil.copytree(root / "probe", base / "probe")
    runs = 0
    for d in sorted(root.iterdir()) if root.exists() else []:
        if d.is_dir() and d.name != "probe":
            shutil.copytree(d, base / "runs" / d.name, ignore=_ignore)
            runs += 1
    cdir = Path(configs_dir or os.path.join(src_root, "configs"))
    if cdir.exists():
        shutil.copytree(cdir, stage / "configs")
    commit = _git_commit(src_root)
    (base / "README.md").write_text(
        f"# Tournament `{plan['name']}`\n\n- uploaded: {time.ctime()}\n- code commit: {commit}\n- GPUs: {_gpu_info()}\n"
        f"- candidates: {', '.join(c['id'] for c in plan['candidates'])}\n- tokens per run: {plan['tokens']:,} | batch tokens: {plan['batch_tokens']:,} | "
        f"seq_len: {plan['seq_len']} | seeds: {plan['seeds']}\n\n"
        "- `report.md` / `report.csv`: the ranking table\n- `probe/`: tokens/s, memory and max context per candidate\n"
        "- `runs/<job>/`: metrics.jsonl (curves), results.json (final numbers, loss by position), experiment.json (exact config), stdout.log\n"
        "- checkpoints are not here: they live under `tournament/<plan>/<job>/` only when `hub_repo` was set in the plan\n", encoding="utf-8")
    files = [p for p in stage.rglob("*") if p.is_file()]
    return {"runs": runs, "files": len(files), "mb": sum(p.stat().st_size for p in files) / 1e6, "commit": commit}


def get_token() -> Optional[str]:
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return os.environ.get("HF_TOKEN")


def upload_results(plan: Dict[str, Any], plan_path: str, repo_id: str, subdir: str = "", private: bool = True,
                   stage_dir: str = "hub_stage", src_root: str = ".", log=print) -> bool:
    token = get_token()
    if not token:
        log("No HF_TOKEN found: on Kaggle add it under Add-ons > Secrets and attach it to the notebook; elsewhere `export HF_TOKEN=...`.")
        return False
    summary = stage_results(plan, plan_path, stage_dir, src_root)
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    log(f"uploading {summary['mb']:.1f} MB ({summary['files']} files, {summary['runs']} runs) to {repo_id}")
    api.upload_folder(folder_path=stage_dir, repo_id=repo_id, repo_type="model", path_in_repo=subdir or None,
                      commit_message=f"tournament {plan['name']} {time.strftime('%Y-%m-%d %H:%M')} ({summary['commit'][:8]})")
    log(f"done: https://huggingface.co/{repo_id}" + (f"/tree/main/{subdir}" if subdir else ""))
    return True
