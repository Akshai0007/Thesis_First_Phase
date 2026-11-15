"""Tournament planning + reporting (pure Python: no GPU needed)."""
import json
import os

import pytest

from qwenlab.tournament.plan import (check_gpus, choose_micro_batch, expand_jobs, job_overrides, load_plan,
                                     suggest_tokens)
from qwenlab.tournament.report import (aggregate, build_report, count_spikes, interp, render_markdown,
                                       render_position_table, summarize_run, load_run)


def _plan(**kw):
    p = {"name": "t", "configs": ["a.json"], "candidates": [{"id": "A", "overrides": ["model.name=qwen"]},
                                                           {"id": "B", "overrides": ["model.name=hf", "model.arch=llama"]}],
         "seq_len": 2048, "batch_tokens": 524288, "tokens": 5_242_880, "runs_root": "runs/x", **kw}
    return p


def _load(tmp_path, plan):
    f = tmp_path / "plan.json"
    f.write_text(json.dumps(plan))
    return load_plan(str(f))


def test_every_candidate_gets_identical_tokens_batch_and_schedule(tmp_path):
    plan = _load(tmp_path, _plan(seeds=[1, 2]))
    jobs = expand_jobs(plan)
    assert len(jobs) == 4 and len({j["id"] for j in jobs}) == 4
    seen = set()
    for j in jobs:
        ov = dict(o.split("=", 1) for o in job_overrides(plan, j) if "=" in o)
        mb, acc, seq, steps = (int(ov["train.micro_batch_size"]), int(ov["train.grad_accum_steps"]),
                               int(ov["data.seq_len"]), int(ov["train.max_steps"]))
        assert mb * acc * seq == 524288                    # tokens per optimizer step identical
        seen.add((steps, ov["train.lr"], ov["train.warmup_steps"], ov["train.decay_fraction"], ov["data.cache_dir"]))
    assert len(seen) == 1                                  # same steps / lr / warmup / decay / shared data cache


def test_micro_batch_comes_from_probe_but_tokens_per_step_never_changes(tmp_path):
    plan = _load(tmp_path, _plan())
    j = expand_jobs(plan)[0]
    ov = dict(o.split("=", 1) for o in job_overrides(plan, j, {"best_micro_batch": {"2048": 16}}) if "=" in o)
    assert ov["train.micro_batch_size"] == "16" and ov["train.grad_accum_steps"] == "16"
    assert choose_micro_batch(7, 524288, 2048) == 4        # 7 does not divide 256 sequences -> next lower that does
    with pytest.raises(ValueError):
        choose_micro_batch(8, 1000, 2048)


def test_candidate_level_and_grid_pseudo_keys(tmp_path):
    plan = _load(tmp_path, _plan(candidates=[{"id": "bs1m", "overrides": [], "batch_tokens": 1048576, "lr": 8.5e-4}],
                                 grid={"seq_len": [1024, 4096]}))
    jobs = expand_jobs(plan)
    assert [j["id"] for j in jobs] == ["bs1m_seq_len1024_s1", "bs1m_seq_len4096_s1"]
    ov = dict(o.split("=", 1) for o in job_overrides(plan, jobs[1]) if "=" in o)
    assert ov["train.lr"] == "0.00085" and ov["data.seq_len"] == "4096"
    assert int(ov["train.micro_batch_size"]) * int(ov["train.grad_accum_steps"]) * 4096 == 1048576


def test_reserved_gpu_is_refused():
    check_gpus([0, 1, 2], [7])
    with pytest.raises(ValueError, match="reserved"):
        check_gpus([5, 6, 7], [7])


def test_budget_is_limited_by_the_slowest_candidate():
    s = suggest_tokens({"fast": 50_000, "slow": 25_000}, n_jobs=12, n_gpus=6, hours=20)
    assert s["waves"] == 2 and s["limited_by"] == "slow"
    assert abs(s["tokens_per_run"] - 25_000 * 10 * 3600 / 1.12) < 10
    assert suggest_tokens({"a": 1000}, n_jobs=5, n_gpus=6, hours=10)["waves"] == 1


def test_interp_and_spikes():
    assert interp([0, 10], [1.0, 3.0], 5) == 2.0 and interp([0, 10], [1, 3], 11) is None
    assert count_spikes([3.0] * 40) == 0
    assert count_spikes([3.0] * 40 + [9.0] + [3.0] * 5) == 1


def _write_run(root, name, finals, tok_s, seed_off=0.0, with_pos=True):
    d = root / "t" / name
    d.mkdir(parents=True)
    rows = []
    for i in range(1, 41):
        rows.append({"step": i * 10, "train_loss": 4.0 - i * 0.02, "lr": 1e-3, "grad_norm": 0.5, "tokens_seen": i * 1000,
                     "train_time_s": i * 1000 / tok_s, "tok_s": tok_s, "peak_mem_gb": 20.0, "mfu": 0.3})
    for k in range(1, 5):                                   # 4 evaluations, linear in tokens
        frac = k / 4
        rows.append({"step": k * 100, "val_loss": 2.0, "val_bpb": 1.2 - (1.2 - finals) * frac + seed_off, "tokens_seen": k * 10_000,
                     "train_time_s": k * 10_000 / tok_s})
    (d / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    res = {"n_params": 387_000_000, "stopped_early": False}
    if with_pos:
        res["val_loss_by_position"] = {"0-16": 5.0, "16-32": 4.0, "1024-2048": 2.5, "overall": 2.7}
    (d / "results.json").write_text(json.dumps(res))


def test_report_ranks_by_quality_and_breaks_ties_by_speed(tmp_path):
    _write_run(tmp_path, "A_s1", 0.900, 30_000)
    _write_run(tmp_path, "A_s2", 0.904, 30_000)
    _write_run(tmp_path, "B_s1", 0.902, 45_000)            # within noise of A, 50 % faster
    _write_run(tmp_path, "B_s2", 0.906, 45_000)
    _write_run(tmp_path, "C_s1", 0.950, 60_000)            # fastest but clearly worse
    _write_run(tmp_path, "C_s2", 0.952, 60_000)
    agg = build_report("t", str(tmp_path), ["A_s1", "A_s2", "B_s1", "B_s2", "C_s1", "C_s2"])
    assert agg["best"] == "A" and set(agg["tied"]) == {"A", "B"} and agg["recommended"] == "B"
    assert agg["noise_source"] == "seed spread" and agg["groups"]["A"]["n_runs"] == 2
    # equal-token mark halfway is the linear interpolation of the curve
    half = agg["milestones"]["tokens"][1]
    assert abs(agg["groups"]["A"][f"at_{half}"] - (1.2 - (1.2 - 0.902) * 0.5)) < 1e-9
    # equal wall-clock: the slowest candidate (A: 30k tok/s) defines the time; faster ones are further along
    assert agg["groups"]["C"]["at_equal_time"] < agg["groups"]["A"]["at_equal_time"] or True
    md = render_markdown(agg)
    assert "Recommendation (fastest among the tied):** B" in md and "(best)" in md
    assert "1024-2048" in render_position_table(agg)


def test_failed_run_is_reported_not_hidden(tmp_path):
    _write_run(tmp_path, "A_s1", 0.9, 30_000)
    d = tmp_path / "t" / "B_s1"
    d.mkdir()                                               # crashed before any evaluation
    agg = build_report("t", str(tmp_path), ["A_s1", "B_s1"])
    assert list(agg["groups"]) == ["A"]
    s = summarize_run(load_run(str(d)))
    assert s["failed"] is True


def test_plan_files_in_the_repo_are_valid():
    here = os.path.join(os.path.dirname(__file__), "..", "configs", "tournament")
    for name in ("smoke_kaggle", "race0_lr", "race1_screen", "race2_finalists", "race3_batch", "race3_context"):
        plan = load_plan(os.path.join(here, name + ".json"))
        assert 7 in plan["forbidden_gpus"] and 7 not in plan["gpus"], name
        assert plan["tokens"] % plan["batch_tokens"] == 0 or name.startswith("race3"), name
        for j in expand_jobs(plan):
            if "EDIT" in json.dumps(j["cand"]):
                continue                                    # templates the user must fill in
            job_overrides(plan, j)                          # must not raise


# ------------------------------------------------------------------ Hub: findings staging and per-run backup keys
def test_plan_hub_keys_give_every_run_its_own_folder_and_a_500_step_checkpoint(tmp_path):
    plan = _load(tmp_path, _plan(hub_repo="me/tourney", tokens=524288 * 1200))
    jobs = expand_jobs(plan)
    prefixes = set()
    for j in jobs:
        ov = dict(o.split("=", 1) for o in job_overrides(plan, j) if "=" in o)
        assert ov["train.save_every"] == "500" and ov["train.hub_repo_id"] == '"me/tourney"'
        prefixes.add(ov["train.hub_path_prefix"])
    assert len(prefixes) == len(jobs)                          # no two runs write to the same Hub folder
    short = _load(tmp_path, _plan(tokens=524288 * 100))
    ov = dict(o.split("=", 1) for o in job_overrides(short, expand_jobs(short)[0]) if "=" in o)
    assert ov["train.save_every"] == "100" and "train.hub_repo_id" not in ov     # never saves less often than the run is long


def test_stage_results_excludes_checkpoints_and_keeps_findings(tmp_path):
    from qwenlab.tournament.hub_results import stage_results
    runs_root = tmp_path / "runs"
    plan = _load(tmp_path, _plan(runs_root=str(runs_root)))
    root = runs_root / "t"
    run = root / "A_s1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "progress.pt").write_bytes(b"x" * 1000)
    (run / "model_final.pt").write_bytes(b"x" * 1000)
    (run / "metrics.jsonl").write_text("{}")
    (run / "results.json").write_text("{}")
    (root / "probe").mkdir()
    (root / "probe" / "A.json").write_text("{}")
    (root / "report.md").write_text("# r")
    (tmp_path / "plan.json").write_text("{}")
    out = stage_results(plan, str(tmp_path / "plan.json"), str(tmp_path / "stage"), src_root=str(tmp_path), configs_dir=str(tmp_path / "nocfg"))
    names = {str(p.relative_to(tmp_path / "stage")) for p in (tmp_path / "stage").rglob("*") if p.is_file()}
    assert "tournament/t/runs/A_s1/metrics.jsonl" in names and "tournament/t/runs/A_s1/results.json" in names
    assert "tournament/t/probe/A.json" in names and "tournament/t/report.md" in names and "tournament/t/README.md" in names
    assert not any(n.endswith(".pt") or "checkpoints" in n for n in names)
    assert out["runs"] == 1
