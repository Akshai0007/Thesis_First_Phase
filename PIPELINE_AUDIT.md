# Pipeline audit against the professor's feedback

Audited: `qwenlab_v2.zip` (package + tests + the old smoke-test notebook). **Not seen:** `qwenlab_scratch_v2.ipynb`,
which, per your project record, holds the real data loading, tokenization, task building, Hub upload and generation probes.

## A. Was the pipeline swappable? (before)
| Question | Finding in qwenlab_v2.zip |
|---|---|
| Swap the **model** by config? | Partly. `MODELS` registry + `ModelConfig` exist, but `vocab_size` was a hand-typed number, never taken from a tokenizer. The smoke test even edited the shared `QWEN2_5_0_5B` preset in place (`cfg.vocab_size = ...`), which silently changes it for the rest of the process. |
| Swap the **tokenizer**? | No. `config.py` has `TokenizerConfig` and a `TOKENIZERS` registry, but nothing registers a tokenizer and nothing reads the config. No tokenizer code in the zip. |
| Swap the **dataset** (e.g. story telling)? | No. `SourceConfig`, `PretrainDataConfig` and a `SOURCES` registry exist, but no source, loader, cleaning or caching code is in the zip. Dataset code lived in the notebook. |
| Config-driven run? | No. `load_config` / `ExperimentConfig` were never called by any script; there were no config files and no `train.py`. |
| Task registry uniform? | No. `causal_lm` is built with `(seq_len, eos_id)`, `summarize/qa/correction` with `(tokenizer, seq_len, im_start, im_end)`, `fim` with token ids. They cannot be built generically from config. (Left as is: not needed for the pure causal-LM stage.) |
| `import qwenlab` registers everything? | Docstring said so; code did not (fixed). |

Verdict: the **model, init, trainer, schedules, resume and batch-mixing code are solid and modular**; the **data/tokenizer/config/evaluation/inference layers were missing from the package**, so a dataset or tokenizer swap meant editing notebook code.

## B. Professor's points -> what was done
| Professor | Status before | Change |
|---|---|---|
| Continuous text for next-word prediction, strip Q&A | Stage 2 mixed in SQuAD QA, Alpaca, correction, summarise, FIM (50% of the last 20% of steps) | New `data/pretrain_data.py` + `preprocess.py`: plain-text windows only; documents with chat/FIM markers or `Question:/Answer:` layout are **dropped and counted**. `scripts/train.py` refuses `train.stage != "pretrain"`. Task code is kept, unused, for a later stage. |
| Repetition = exceeding context window | `max_seq_len` was 32768 while training used 512; `model.generate` cropped to 32768, so nothing enforced 512; no length cap, no greedy mode, no stop rule | One source of truth: `data.seq_len` = training context. Factory sets `model.max_seq_len` to it. `inference/generate.py` clamps prompt + new tokens inside it, hard-caps new tokens (64), stops at ~2 sentences / EOS, optional repetition penalty and n-gram block. `generate.py --repetition-check` measures looping inside vs beyond the window. |
| Perplexity, not only loss | Trainer returned a plain mean loss; perplexity existed only in notebook code | `evaluation/perplexity.py` (token-weighted loss, `exp(loss)`), used by the trainer; every eval prints and writes `val_loss` + `val_perplexity` to `metrics.jsonl`; step-0 evaluation of the untrained model (perplexity should be about the vocabulary size). |
| Config module | Not wired | `configs/base.json` + `english.json` / `storytelling.json` / `malayalam.json`; later files override earlier ones; `--set a.b=c` overrides; typos in keys raise an error. Unused/misleading fields removed (`fim_rate`, `shard_tokens`, HellaSwag/ARC lists that no code implements). |
| Tokenizer as a standalone function | Missing | `data/tokenizers.py`: `hf`, `sentencepiece`, `bytes` (offline tests), one interface. |
| Fresh Qwen2.5 weights, embedding resizes to tokenizer | Init existed; resize did not | `models/factory.py`: vocab = len(tokenizer) rounded up to a multiple of 128; special ids checked; preset is copied, never edited. |
| Checkpoint every N steps, resume | Already good (atomic writes, resume) | Unchanged; now the data position resumes from the step number inside the same script (tested). |
| Inference script with guard-rails | Missing | `scripts/generate.py`. |

## C. Verified here (CPU, tiny model, offline)
- Your original tests: all still pass; plus 9 new tests (cleaning, factory, perplexity, generation guard, config round-trip, end-to-end train -> resume -> evaluate -> generate). Total 65 passed, 1 skipped.
- The skipped test is the Hugging Face equivalence test: `transformers` is not installed in this sandbox. `models/qwen.py` was not edited, but **re-run the suite on Kaggle**.
- Config-only swaps tried on the tiny model: dataset (`stories.jsonl` -> `facts.txt`: 41 Q&A/chat documents dropped), tokenizer (bytes -> SentencePiece, vocab 257 -> 304), model size (2 -> 3 layers, 64 -> 96 hidden). Resume tested (150 -> 200 steps, no repeated step).

## D. NOT verified (be aware)
- No GPU, no fp16, no real Qwen tokenizer, no FineWeb-Edu / TinyStories / Sangraha download (no Hugging Face access from the sandbox). Dataset ids and `text_field` values in the configs are from memory; check them on Kaggle. `configs/malayalam.json` is a template.
- Not included in the package pipeline (still notebook-only): Hugging Face upload/backup, the time-budget auto-fit, Hugging Face-format export, the SFT task streams, the three-way comparison.
- `QwenForCausalLM.generate` still exists (sampling only, no cap); the scripts use `inference/generate.py` instead.
- The tiny-model samples in the smoke run are gibberish by design (91k parameters, 150 steps).

## E. Second check: English -> Malayalam by config only
Tried on the tiny model (SentencePiece trained on a toy Malayalam sample, `configs/smoke_cpu.json` + overrides): runs end to end, vocabulary and embedding resized from the tokenizer, perplexity starts at about the vocabulary size and falls, Malayalam prompt generated inside the context cap.

Gaps found and fixed:
- **Cleaning was English-shaped and fixed in code.** A Malayalam tokenizer trained on text with old-style chillu letters converted to atomic ones got byte-fallback pieces when the model data was not converted the same way (5 tokens became 12 for one phrase; 9% more tokens on the toy corpus). New `data.normalizer` setting with a `malayalam` normaliser (written from your record's rules) plus an optional script-share filter.
- **`init_health` assumed a Qwen layout** (`model.layers[0].self_attn.o_proj`), which would crash another registered model. Now skipped when absent.
- **Config paths were Kaggle-only** (`/kaggle/working/...`), so your professor could not run them. Now relative; the Kaggle notebook passes overrides.

Still true after the swap (not code problems, but not automatic either):
- `base.json` carries the English values (lr 2e-4, warmup 500, micro-batch 1, 20,000 steps). Your Malayalam experiments point to different settings; they are config edits.
- A different **architecture** (not a size change) needs a new class registered in `MODELS` with the same interface (`forward(input_ids, labels, loss_mask) -> (logits, loss)`), and `ModelConfig` fields are Qwen-style (kv heads, rope theta).
- Perplexity is not comparable across tokenizers (English vs Malayalam); use bits-per-byte for that.

## F. Outside suggestions: checked against the code
| # | Suggestion | Verdict |
|---|---|---|
| Universal sources | csv / parquet / custom jsonl columns / joined columns | **Correct, done.** My own version (shared helper, fails on a wrong column name with the list of real ones; csv/parquet read locally without needing `datasets`). |
| 1.1 | Small dataset puts all text in validation | **Correct, fixed.** Validation now takes at most `val_tokens` and at most `max_val_fraction` (10%) of the text. |
| 1.3 | Chillu normalisation; configurable Q&A drop | Chillu: **already done** in the previous update (`data.normalizer`). `data.drop_qa_format` switch: **added** (default on, as the professor wants). The claim that medical `A:` notes would be dropped is overstated: the filter needs a `Question:`/`Q:` line followed by an `Answer:`/`A:` line. |
| 2.1 | Qwen3 (qk-norm, decoupled head_dim) | **Not done on purpose.** Your own earlier design removed Qwen3 ("no longer in scope"), the professor asked for Qwen2.5-0.5B, and adding it changes the model verified against Hugging Face and the weight bridge. Possible later as a separate, tested step. |
| 2.2 | Auto-extend RoPE | **Rejected.** It would let the model silently run on positions it never trained on, which is the opposite of the professor's guard-rail. Instead a too-long input now raises a clear error naming the context window. |
| 3.1 | eos fallback for non-Qwen tokenizers | **Correct, fixed, and worse than described:** `convert_tokens_to_ids` returns the `<unk>` id for a missing token, so the old code could silently use `<unk>` as end-of-text. Now it checks, falls back to the tokenizer's own eos, or errors. |
| 3.2 | Vocabulary 151,680 vs 151,936 | **Correct and important.** My factory gave the Qwen tokenizer 151,680 (493,803,392 parameters), not the 151,936 / 494,032,768 of your trained model and official Qwen2.5. New `model.auto_vocab` switch; `english.json` sets it off with 151,936. |
| 4.1 | `stop_at_step` | **Correct, done.** Tested: pause at step 20 and resume gives the same learning-rate curve as one uninterrupted run. A "re-warmup" mode for a run that already decayed was **not** added (not requested by the professor). |
| 5.1 | Urdu / Hindi sentence marks | **Correct, done** (۔ ؟ ॥ added; one shared definition). |
| 5.2 | Out-of-domain evaluation | **Correct, done** (`evaluate.py --source-json`, any source and format). |
| 6.1 | Hub backup/restore and `hf_model/` export | **Correct, done**, with limits: tested against a fake local Hub only (no network here). Old Hub copies are not deleted by code. |
| 6.2 | pip install in read-only `/kaggle/input` | **Correct, fixed:** the notebook copies the package to `/kaggle/working` first. |

Also added: a data fingerprint check on resume (a rebuilt dataset that differs from the one the run started on stops the run).

Test status after this update: 86 passed, 0 skipped (`transformers` is now installed here, so the Hugging Face equivalence test ran and passed with the edited model file).
Still not run: a real GPU / fp16 run, the real Qwen tokenizer, any real Hub upload, real dataset downloads.

## G. Review of the three Kaggle notebooks (Run 1 baseline, Run 2 training, Run 3 comparison)
Problems found in the supplied code and fixed in the delivered notebooks:
| # | Problem | Fix |
|---|---|---|
| 1 | **Hub clean-up compared `f.oid` with the current file's SHA-256.** In huggingface_hub the SHA-256 is `file_oid`; `oid` is a different id, so every object looked stale and the current checkpoint would have been deleted (the same mistake that deleted the model before). Present in both the Run 2 helper and the training notebook. | `HubBackup.prune_stale_lfs()`: compares `file_oid`, refuses to delete unless the current file is positively matched, never deletes everything, prints the list. Tested with a fake Hub whose `oid` differs from `file_oid`. |
| 2 | Continuing the old 20k run with a 30k plan: the old run had decayed to 2e-5; a plain 30k WSD plan jumps back to 2e-4 at step 20,001. `stop_at_step` does not fix that when the old run was planned for 20k. | `train.schedule_start_step`: the schedule restarts at the checkpoint step (warm-up 200, peak 1e-4, decay over the remaining 10k). |
| 3 | The seeded run would resume at window 320,000 of a shuffled new data pool: ~60% of the 10k extra steps would repeat text the old run had seen. | `skip_docs=150000` (the same boundary your Run 1 uses for unseen FineWeb text) and the data position is counted from the restart step. |
| 4 | The old repo's `training_history.json` (different format) was loaded; the final table would crash with a KeyError on the first record. | History, plan and metrics are restored only from your own checkpoint repo, never from the old repo. |
| 5 | A model stopped at any step <= 20,000 (e.g. a time-out at 12,000) would have been exported into the "20k" repo. | Export only at exactly 20,000 or 30,000 (or an explicit `EXPORT_ANYWAY_TO`). |
| 6 | Repo names differed between documents (`Thesis_Modular_v3_*` vs `Thesis_Modular_v2_*`). | One set, matching the training notebook: `Thesis_Modular_v2_Ckpt / _20k / _30k`. |
| 7 | A resumed session did not check that the rebuilt dataset equals the one the run started on. | Data fingerprint stored in the run plan; a mismatch stops the run. |
| 8 | The checkpoint was uploaded twice at a planned stop (in-loop save and "final" save), 6 GB each. | The trainer no longer saves the same step twice; the notebook never uploads a step twice. |
| 9 | Monkey-patching `Trainer.save_checkpoint`; the NaN check then ran after the previous file was already replaced. | `on_checkpoint` callback; the trainer's own NaN guard runs before anything is written. |
| 10 | `dtype=` vs `torch_dtype=` differs between transformers versions. | `load_hf` tries one, falls back to the other. |
| 11 | Evaluation notebooks wrote results into your model repo `Thesis_Scratch_1`. | Results go to a separate small repo `Akzhai/Thesis_Eval`. |
| 12 | The "unguarded 120 tokens" repetition figure cut the text at the first blank line, so it measured shorter, different text. | Raw text (only stop tokens removed). Run 3 adds the real test of the professor's hypothesis for every model: one long generation, repetition inside vs beyond the 512-token window. |

Tested here (CPU, tiny models, fake local Hub and fake datasets): the real cells of the training notebook across simulated session breaks in both modes, Hub down for a whole run, missing old repo, changed data, old repo left byte-identical; the real cells of the other two notebooks end to end. Not tested: a real GPU/fp16 run, real Hugging Face traffic, the real datasets and models.
