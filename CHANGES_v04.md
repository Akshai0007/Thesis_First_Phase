# qwenlab v0.4: what changed, why, and what is still unproven

## 1. Audit verdict on v0.3 (from reading every file)

**Already good and untouched:** strict typed config with overrides; registries for tokenizer / source / normalizer / init / schedule / task;
a hand-written Qwen2 that matches Hugging Face logits exactly; deterministic resumable batching; atomic checkpoints; data-cache fingerprint
checks; time budget for Kaggle sessions; Hub backup; perplexity evaluation; guarded generation; 86 tests.

**Plug-in-ness, honestly:**

| Part | v0.3 | Verdict |
|---|---|---|
| tokenizer, data source, normalizer, init, schedule, task mix | registry + config | plug-in |
| model | `MODELS` registry exists, **but** `ModelConfig` is Qwen-shaped and every architecture needs a hand-written class that honours the same interface | partly: swapping Qwen -> Qwen3/Llama/OLMo/Gemma/SmolLM3 meant writing code |
| optimizer | hard-wired AdamW | not plug-in |
| trainer | single GPU, no DDP | fine for a tournament (one run per GPU), a limit for the final long run |
| evaluation | validation loss / perplexity only | no bits-per-byte, no context diagnostics |

**Dead or misleading items found:** `train.compile` and `model.gradient_checkpointing` were config fields nothing read; the chunked loss
did not reduce backward memory (every chunk's logits stayed alive for autograd) and called `.item()` once per chunk;
`train.precision=auto` picked bf16 on a T4 (emulated, slow); throughput and peak memory were never logged, so "training speed" could not be compared;
validation text was the first documents of the stream; the token cache is uint32 (twice the size needed for a 32k vocabulary); no duplicate removal;
no byte count, so perplexity cannot be compared across tokenizers.

## 2. What changed (everything new is opt-in; defaults reproduce v0.3 behaviour)

| File | Change | Default |
|---|---|---|
| `models/hf_adapter.py` (new), `models/arch_presets.py` (new) | `model.name="hf"` + `model.arch` builds ANY Hugging Face causal LM from the SAME shape fields, behind the exact interface `forward(input_ids, labels, attn_mask, loss_mask) -> (logits, loss)`. Presets: qwen2, qwen3, llama, olmo2, smollm3, gemma3 (+ experimental hybrids falcon_h1, qwen3_5). Config fields silently dropped by a transformers version raise an error. | not used unless `name="hf"` |
| `config.py` | new fields: `model.arch/hf_config/attn_impl/loss_chunk_checkpoint`, `train.optimizer/optimizer_params/fused_adamw`, `data.token_dtype/dedup_exact`, `eval.position_loss` | all default to v0.3 behaviour except `eval.position_loss=true` (extra diagnostic in the final evaluation only; failure-proof) |
| `training/optimizers.py` (new), `registry.py` | `OPTIMIZERS` registry: `adamw` (identical to before) and `muon_adamw` (new, untested at scale) | `adamw` |
| `training/trainer.py` | logs `tokens_seen`, `train_time_s`, `tok_s` (end-to-end), `s_per_step`, `mfu`, `peak_mem_gb`, `val_bpb`; `train.compile` now works; GradScaler handles optimizer bundles; `auto` precision = bf16 on Ampere+, fp16 on older GPUs; tokens and time survive resume | extra log fields only |
| `models/qwen.py` | `gradient_checkpointing` now works; chunked loss no longer syncs per chunk and can recompute chunk logits (`loss_chunk_checkpoint`) | off |
| `data/pretrain_data.py` | UTF-8 byte counts (`train_bytes`, `val_bytes`) -> bits-per-byte; `token_dtype="uint16"`; `dedup_exact` | cache keys unchanged unless you change these |
| `evaluation/position_loss.py` (new), `pipeline.py` | final loss by position bucket; `val_bpb`, `n_params`, `tokens_seen`, `train_time_s` in `results.json`; MFU accounting | on |
| `export.py` | exports HF-adapter models with `save_pretrained` + reload check | n/a |
| `init.py` | `hf_default` scheme keeps the architecture's own initialisation (hybrids) | n/a |
| `tournament/` (new), `scripts/tournament.py`, `configs/tournament/*.json` | the model-comparison harness: check, smoke, prepare, probe, budget, race, report | n/a |
| `hub.py` users: `pipeline.py`, `config.py` | `train.hub_path_prefix`: many runs share one Hub repo, one folder each; plans set it automatically when `hub_repo` is given | empty = old layout |
| `tournament/hub_results.py`, `scripts/upload_results.py` | upload a tournament's findings (no checkpoints) to the Hub, patterned on your tokenizer-results upload cell | n/a |
| tests | `test_tournament_plan_report.py` (pure Python), `test_v04_models_and_trainer.py`, `test_v04_data.py` (need torch/transformers) | n/a |

Backward compatibility is tested, not assumed: `test_default_fingerprint_is_exactly_the_old_one` proves v0.3 data caches are still found;
the default optimizer test proves the same two AdamW groups; the existing loss-path test still compares chunked and full loss.

## 3. Verification status (read this before trusting anything)

* **Executed in the authoring sandbox (no torch available):** the tournament planning and report logic (9 tests pass: equal tokens per step for every
  candidate, micro-batch/accumulation arithmetic, GPU 7 refusal, budget fit, report ranking, tie-break by speed, equal-time interpolation, failed-run handling),
  syntax of every file, and the validity of every plan file.
* **NOT executed anywhere yet:** the Hugging Face adapter, presets for non-Qwen architectures, Muon, the trainer changes, uint16/dedup/bytes, loss-by-position,
  probe, check, smoke, race launcher. They are written against the code I read, and tests for them exist, but **the first `pytest` run on Kaggle is the real
  test**. Expect small fixes, mostly transformers-version field names for Gemma-3 / OLMo-2 / SmolLM3 (the `check` phase is built to find exactly those).
* Hybrid presets (Falcon-H1, Qwen3.5) need extra CUDA kernels and a recent transformers: treat as phase-2 experiments, not part of the one-day race.

## 4. Next improvements (not done, in priority order)

1. **Multi-GPU (DDP) for the final run**: not needed for the tournament, but one A100 at ~35k tok/s needs ~58 h for 8B tokens; 5-6 GPUs bring it to ~11-14 h. Do it after the winner is known (about 60 lines in the trainer + a rank-aware batcher).
2. **Validation set**: first-documents-of-the-stream is biased for web data; use a hash split (like `lmlab`'s `val_fraction`) and keep a separate news/wiki validation file for domain breakdowns.
3. **Parallel tokenisation**: encoding is one document at a time; `encode_batch` / multiprocessing makes the 1.5B-token cache build 5-10x faster.
4. **Downstream evaluation**: Belebele-Malayalam, MILU, minimal pairs by log-likelihood, with a task-qualification step (see `PLAN.md` from the earlier answer) - validation BPB alone does not tell you the model is useful.
5. **Contamination check** between benchmark text and the training cache before reporting benchmark numbers.
6. Remove `tasks/` or wire it in; it is currently unused by `pipeline.py`.
