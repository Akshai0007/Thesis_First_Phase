# Model tournament runbook (Kaggle first, then the DGX A100 80 GB)

Goal: pick the best architecture for Malayalam on your hardware by measuring **quality (validation bits-per-byte), speed (tokens/s, MFU),
memory (peak GB, longest context), and stability** under identical data, tokens, batch and schedule - within about one day of DGX time.

Everything is driven by plan files in `configs/tournament/`. Every run is an ordinary `scripts/train.py` run: resumable, same
`metrics.jsonl` / `results.json`. Re-running a phase skips what is already finished. GPU 7 is refused by the launcher.

## 0. Rules that keep the comparison fair

* Same data cache, same tokens, same tokens per optimizer step, same warm-up/decay, same LR for all candidates (tournament plans override only the model).
* Micro-batch differs per model (memory detail); gradient accumulation fills the gap, so tokens per step is identical.
* The learning rate is checked on the baseline FIRST (race0): a wrong LR makes every architecture comparison unfair.
* Two seeds for the finalists give you the real noise level; ties within the noise are resolved in favour of the faster model.

## A. Kaggle (2 x T4, 16 GB) - prove it works, ~1.5 h. Do NOT skip: a bug found here costs minutes, on the DGX it costs a night.

```bash
pip install -e ".[hf-archs]"                      # transformers >= 4.53 for the Qwen3 / SmolLM3 / OLMo-2 / Gemma-3 classes
python -m pytest tests -q                         # your 86 existing tests + the new ones; ALL must pass (the new torch tests have never run)

PLAN=configs/tournament/smoke_kaggle.json
python scripts/tournament.py --plan $PLAN --phase check --vocab 32000    # every candidate at FULL size on CPU: params, 1 fwd/bwd
python scripts/tournament.py --plan $PLAN --phase smoke                   # every candidate TINY: loss falls, stop + resume works
```
Expected `check` output: one line per candidate with `status: ok`, `params_M` near 387-395, and `loss` close to `ln_vocab` (about 10.4 for 32k).
Also `[parity] ... parameter counts MATCH` for hand-written Qwen vs HF Qwen2. A line `FAILED: ...` names the exact field or package to fix
(e.g. a transformers version without Gemma-3). Fix or delete that candidate from the plan; do not take a broken one to the DGX.

```bash
python scripts/tournament.py --plan $PLAN --phase prepare                 # builds the tokenised cache once (20M tokens for the smoke)
python scripts/tournament.py --plan $PLAN --phase probe --gpus 0          # tokens/s, memory, max context per candidate (T4: expect small numbers)
python scripts/tournament.py --plan $PLAN --phase race --gpus 0,1         # 9 candidates x ~3M tokens, fp16 on T4 (auto)
python scripts/tournament.py --plan $PLAN --phase report                  # ranking table: only checks the machinery, NOT a scientific result
```
Smoke passes when: no candidate FAILED, every run produced `results.json`, the report table lists all candidates, and losses fell.
Then copy `runs/` (small files only) somewhere safe; do not copy checkpoints.

## B. DGX setup (30 min)

1. JupyterHub terminal. `nvidia-smi`: note which of GPUs 0-6 are free. **Never GPU 7.** Shared ID: coordinate with co-users.
2. Create a venv in your home directory, `pip install -e ".[hf-archs]"`, run `python -m pytest tests -q` again (bf16 + real CUDA paths).
3. Put your tokenizer files in place; set `configs/malayalam.json` to your real dataset and tokenizer.
4. Policy: the cache for 1.5B tokens in uint16 is ~3 GB (fine). Checkpoints are ~6 GB per run: after each race `rm -rf runs/tournament/<plan>/*/checkpoints` once `results.json` exists; back up `metrics.jsonl`, `results.json`, `report.*` to the Hub / your laptop.
5. Run long jobs from the terminal with `nohup ... &` (a closed notebook tab kills a notebook kernel). At the very end: kernel-shutdown snippet, then File > Hub Control Panel > Stop My Server.

## C. DGX race program (~16 h of GPU time; fits in a day)

| # | Command (plan in `configs/tournament/`) | Time | What it decides |
|---|---|---|---|
| 1 | `prepare` with race1_screen | 1-2 h (tokenising ~1.5B tokens, single process) | builds the shared cache; prints epochs per run |
| 2 | `probe` with race1_screen, `--gpus 0` | ~1 h | micro-batch per model, tokens/s, max context plain / with checkpointing |
| 3 | `budget --hours 5 --gpus 0,1,2,3,4,5` | seconds | tokens per run that fit; edit `tokens` in the plan (multiple of `batch_tokens`) |
| 4 | `race` with race0_lr, `--gpus 0,1,2` | ~45 min | LR for the baseline (3e-4 / 6e-4 / 1.2e-3). Put the winner into `lr` of the other plans |
| 5 | `race` with race1_screen, `--gpus 0,1,2,3,4,5` | ~5 h (2 waves) | screening: 9 candidates x 1 seed x ~300M tokens |
| 6 | `report`, then edit race2_finalists (baseline + best 2) | minutes | pick finalists; check the table for spikes/instability |
| 7 | `race` with race2_finalists | ~10 h (1 wave of 6) | final: 3 models x 2 seeds x ~1.2B tokens; gives the decision |
| 8 | `report` | seconds | recommendation (rule below) |

Equal-token budget means the SLOWEST candidate sets how many tokens fit; `budget` computes it from the probe. If time is short, lower `tokens`
in race2 (800M still separates clear differences; it will not separate differences smaller than the noise).

Commands (replace PLAN):
```bash
python scripts/tournament.py --plan configs/tournament/race1_screen.json --phase probe --gpus 0
python scripts/tournament.py --plan configs/tournament/race1_screen.json --phase budget --hours 5 --gpus 0,1,2,3,4,5
nohup python scripts/tournament.py --plan configs/tournament/race1_screen.json --phase race --gpus 0,1,2,3,4,5 > race1.log 2>&1 &
python scripts/tournament.py --plan configs/tournament/race1_screen.json --phase report
```

## D. Reading the report

* **Quality**: `final` (val bits-per-byte, lower is better), plus the curve at 25 / 50 / 100 % of the tokens every candidate reached.
* **Per-GPU-hour quality**: `@equal time` = quality after the training time of the slowest candidate: a faster model gets further in the same hours.
* **Speed/memory**: tok/s (end-to-end, data loading included), MFU (A100 peak 312 TFLOPs), peak GB.
* **Stability**: max grad-norm and loss spikes. A model that only works at a low LR is not a win; if one candidate spikes, re-run it at half LR before discarding it.
* **Context use**: `ctx gain` and the position table: loss in early vs late positions of the window.
* **Rule**: best final quality wins; everything within the noise of the best is tied; recommend the FASTEST of the tied. The noise is the seed spread from race2; before that it is a default (0.004 BPB) you should treat as a guess.
* Do not trust a difference smaller than the noise, and do not generalise a 300M-token ranking to 8B tokens without the finals: screening is for elimination only.

## E. After the winner: batch size and context window

1. Put the winner's overrides into `race3_batch.json` and `race3_context.json` (replace the `EDIT:` lines), run `check` then `probe` for each plan.
2. `race3_batch`: 256k / 512k / 1M tokens per step at equal tokens (LR scaled ~sqrt). Compare quality at equal TOKENS.
3. `race3_context`: 1k / 2k / 4k / 8k / 16k windows at equal tokens. Compare the loss in the SHARED position buckets (the position table), not the overall loss: longer windows have lower overall loss simply because late positions are easier.
4. Choose the training window as the longest one whose late-bucket gain is real AND whose throughput cost you can afford; extend later with the staged approach in PLAN.md if you need more.

## F. Adopting the winner in your pipeline

* Winner is the hand-written `qwen`: nothing changes.
* Winner is an HF architecture: in `configs/malayalam.json` set `"model": {"name": "hf", "arch": "<winner>", ...shape...}` and `"init": {"scheme": "gpt2_scaled"}` (or `hf_default` for hybrids). Training, evaluation, generation, resume and Hub backup work unchanged; `export_hf` writes the model with `save_pretrained`.
* Then train the final model for the full budget (8B tokens on one GPU is ~58 h; see CHANGES_v04.md "Next improvements" for multi-GPU).

## G. Hugging Face: crash insurance and findings

**Token.** Kaggle: Add-ons > Secrets > `HF_TOKEN`, attach it to the notebook. DGX: `export HF_TOKEN=...` (needs internet from the DGX; if it has none, copy `runs/` off the machine by hand, as the policy asks).

**1. Checkpoints that survive crashes (per run).** Every run saves a local checkpoint every **500 steps** (`save_every` in the plan, default 500; a run shorter than 500 steps saves at its end).
To also back them up, set in the plan file: `"hub_repo": "Akzhai/malayalam-lm-tournament"`. Each run then uploads to its own folder
`tournament/<plan>/<job>/` in that one repo (checkpoint every `hub_upload_every_n_saves` = 2nd save and at the end, plus metrics.jsonl, experiment.json, results.json).
A full checkpoint is ~6 GB and every upload adds a new copy to your Hub storage: use `hub_repo` for the long runs (race2 finals), not for 2-hour screening runs; check your storage quota first; `HubBackup.prune_stale_lfs(dry_run=True)` (existing code) shows old copies you can free.

**2. Continuing after a crash / lost session.** Run the SAME command again (`--phase race ...`). Finished runs (`results.json` present) are skipped. An unfinished run resumes from its local
`checkpoints/progress.pt`; if the output folder is gone (new Kaggle session) and `hub_repo` was set, the checkpoint is downloaded from the Hub before training starts (existing `run_training` behaviour, now per-run via `train.hub_path_prefix`).
If a whole machine/session was lost, re-running also "re-finishes" completed runs cheaply: they restore their final checkpoint, evaluate, and rewrite `results.json`.
The data cache is not on the Hub: re-run `--phase prepare` first on a new machine (the cache key guarantees the same data).

**3. Uploading the findings** (report, probe results, every run's metrics/results/config/log, the plan, configs; no checkpoints), after any phase:
```bash
python scripts/upload_results.py --plan configs/tournament/race1_screen.json --repo Akzhai/malayalam-lm-tournament --subdir dgx
# second Kaggle account / second machine: use --subdir account_B so uploads never overwrite each other
```
Kaggle notebook cell: `!python scripts/upload_results.py --plan configs/tournament/smoke_kaggle.json --repo Akzhai/malayalam-lm-tournament --subdir kaggle_smoke`.
Result in the repo: `<subdir>/tournament/<plan>/{report.md, report.csv, probe/, runs/<job>/..., README.md}` and `<subdir>/configs/`. The README records the code commit and GPU model.
