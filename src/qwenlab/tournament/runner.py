"""Launch tournament runs: one `scripts/train.py` process per GPU, never on a reserved GPU, resumable, OOM-aware.
stdlib only (the training itself happens in the child processes)."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

from .plan import check_gpus, expand_jobs, job_overrides


def _probe_info(plan: Dict[str, Any], cand_id: str) -> Optional[dict]:
    p = os.path.join(plan["runs_root"], plan["name"], "probe", f"{cand_id}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def command_for(plan: Dict[str, Any], job: Dict[str, Any], train_script: str, micro_batch: Optional[int] = None) -> List[str]:
    ov = job_overrides(plan, job, _probe_info(plan, job["cand"]["id"]), micro_batch)
    return [sys.executable, train_script, "--config", *plan["configs"], "--set", *ov]


def cleanup_run_dir(out_dir: str) -> int:
    """Delete the heavy files of a FINISHED run (checkpoints, final weights, exported model); keep metrics/results/config/logs.
    Returns the bytes freed. Used when plan["cleanup_after_run"] is true (Kaggle: its output folder is limited to 20 GB)."""
    freed = 0
    for name in ("checkpoints", "hf_model", "model_final.pt"):
        path = os.path.join(out_dir, name)
        if os.path.isdir(path):
            freed += sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            freed += os.path.getsize(path)
            os.remove(path)
    return freed


def _oom(log_path: str) -> bool:
    try:
        tail = open(log_path, encoding="utf-8", errors="ignore").read()[-6000:].lower()
    except OSError:
        return False
    return "out of memory" in tail


def run_race(plan: Dict[str, Any], gpus: List[int], train_script: str, dry_run: bool = False, log=print) -> Dict[str, str]:
    """Returns {job_id: 'done' | 'skipped' | 'failed' | 'dry-run'}."""
    check_gpus(gpus, plan["forbidden_gpus"])
    jobs = expand_jobs(plan)
    q: Queue = Queue()
    for j in jobs:
        q.put(j)
    status: Dict[str, str] = {}
    lock = threading.Lock()

    def run_one(job: Dict[str, Any], gpu: int) -> str:
        out_dir = os.path.join(plan["runs_root"], plan["name"], job["id"])
        if os.path.exists(os.path.join(out_dir, "results.json")):
            return "skipped"
        os.makedirs(out_dir, exist_ok=True)
        info = _probe_info(plan, job["cand"]["id"]) or {}
        seq = int(job["grid"].get("seq_len", plan["seq_len"]))
        mb = info.get("best_micro_batch", {}).get(str(seq))
        for attempt in range(2):                           # second attempt: half the micro-batch (same tokens per step)
            cmd = command_for(plan, job, train_script, mb)
            log(f"[race] GPU {gpu}: {job['id']}" + (f" (retry, micro-batch {mb})" if attempt else ""))
            log("       " + " ".join(cmd))
            if dry_run:
                return "dry-run"
            log_path = os.path.join(out_dir, "stdout.log")
            with open(log_path, "a", encoding="utf-8") as lf:
                rc = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}, stdout=lf, stderr=subprocess.STDOUT).returncode
            if rc == 0:
                if plan.get("cleanup_after_run") and os.path.exists(os.path.join(out_dir, "results.json")):
                    log(f"[race] {job['id']}: freed {cleanup_run_dir(out_dir) / 2**30:.1f} GB of checkpoints")
                return "done"
            if _oom(log_path) and attempt == 0:
                cur = mb or plan["micro_batch_size"]
                if cur > 1:
                    mb = max(cur // 2, 1)
                    continue
            return "failed"
        return "failed"

    def worker(gpu: int):
        while True:
            try:
                job = q.get_nowait()
            except Empty:
                return
            t0 = time.time()
            res = run_one(job, gpu)
            with lock:
                status[job["id"]] = res
            log(f"[race] {job['id']}: {res} ({(time.time() - t0) / 60:.1f} min)")

    threads = [threading.Thread(target=worker, args=(g,), daemon=True) for g in gpus]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return status
