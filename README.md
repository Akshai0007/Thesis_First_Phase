# qwenlab — modular from-scratch causal-LM pipeline (Qwen2.5-0.5B architecture)

Trains a **randomly initialised** Qwen2.5-0.5B-architecture model on **continuous text** (next-word prediction),
logs **loss and perplexity** on held-out text, checkpoints every N steps (resumable), and generates text with
hard length guard-rails. Every setting lives in `configs/*.json`.

## Run
```bash
pip install -e ".[dev]"               # installs the package + pytest (the scripts also run without installing: they add ./src to the path)
python -m pytest tests -q             # CPU tests, about 1-2 minutes

# offline smoke test, CPU, ~1 minute, no downloads:
python scripts/train.py --config configs/smoke_cpu.json

# real run (Kaggle T4):  later config files override earlier ones; paths in configs are relative,
# on Kaggle add:  --set train.output_dir=/kaggle/working/runs/english data.cache_dir=/kaggle/working/data_cache/english
# no GPU?  add:  --set train.precision=fp32
python scripts/train.py --config configs/base.json configs/english.json
python scripts/evaluate.py --run-dir /kaggle/working/runs/english
python scripts/generate.py --run-dir /kaggle/working/runs/english --prompt "The capital of France is"
# does repetition come from running past the training context window? (long generation, inside vs beyond the window)
python scripts/generate.py --run-dir /kaggle/working/runs/english --prompt "Once upon a time" --repetition-check

# small run with the real Qwen tokenizer and FineWeb-Edu (needs internet; minutes on a GPU) - perplexity should fall from ~vocabulary size:
python scripts/train.py --config configs/base.json configs/english_demo.json
```
Re-running the same `train.py` command resumes from the last checkpoint (exact data position, no repeats).
Outputs in the run folder: `experiment.json` (resolved config), `metrics.jsonl` (train loss, lr, val loss, val
perplexity), `checkpoints/progress.pt`, `model_final.pt`, `results.json` (final metrics + sample texts).

## Swap one thing = edit config only
| To change | Edit | Nothing else changes |
|---|---|---|
| dataset (e.g. story telling) | `data.source.kind` / `data.source.params` (`configs/storytelling.json`) | tokenizer, model, trainer |
| tokenizer (e.g. Malayalam SentencePiece) | `tokenizer.name` / `tokenizer.path` (`configs/malayalam.json`) | the embedding table is resized automatically from the tokenizer |
| model size | `model.hidden_size`, `num_layers`, ... | data, tokenizer, trainer |
| context window | `data.seq_len` | the model's position table and the generation limit follow it |
| language | `--config configs/base.json configs/<language>.json` (sets tokenizer, dataset **and** text normaliser together) | |
| text cleaning | `data.normalizer` (`default`, `malayalam`), `data.script`, `data.min_script_ratio` | add one with `@NORMALIZERS.register(...)` in `data/normalizers.py` |

Local data in any common format, any column names (`kind: text_files`, glob pattern): `.txt`, `.jsonl`, `.csv`,
`.parquet`; `"text_field": "content"` for one column or `["title","body"]` to join several. Hugging Face datasets
take the same `text_field`. A wrong column name fails at once and lists the real ones.

New dataset type: add a function with `@SOURCES.register("name")` in `data/sources.py` (yields strings).
New tokenizer type: add a class with `@TOKENIZERS.register("name")` in `data/tokenizers.py`
(`encode`, `decode`, `len()`, `eos_id`, `pad_id`).

## Structure (matches the professor's five modules)
| Module | Files |
|---|---|
| 1. Configuration | `src/qwenlab/config.py`, `configs/` |
| 2. Data + tokenization | `data/tokenizers.py`, `data/sources.py`, `data/preprocess.py`, `data/pretrain_data.py` |
| 3. Model | `models/qwen.py`, `models/factory.py`, `init.py` |
| 4. Training + evaluation | `training/trainer.py`, `training/schedules.py`, `evaluation/perplexity.py` |
| 5. Inference | `inference/generate.py`, `scripts/generate.py` |

Data rules: documents are cleaned, joined with an end-of-text token and cut into windows of `seq_len`+1 tokens.
Documents containing chat markers, fill-in-the-middle markers or `Question:/Answer:` layouts are dropped and
counted (see `data_stats` in `results.json`). Validation documents are taken from the stream (at most `data.val_tokens`
and at most `data.max_val_fraction` of the text) and are never trained on.

## English -> Malayalam checklist
1. Put the trained Unigram-32k `.model` at `tokenizers/malayalam_unigram32k.model` (or set `tokenizer.path`).
2. Check the dataset id / `data_dir` in `configs/malayalam.json`.
3. Use the SAME cleaning for tokenizer and model text: `data.normalizer=malayalam` (rules from the project record;
   register your own `malayalam_norm.py` function under that name if you want exact parity).
4. Review the training numbers (learning rate, warmup, micro-batch) - `base.json` holds the English values.
5. `python scripts/train.py --config configs/base.json configs/malayalam.json`

## Multi-session runs, pausing, backup
- **Pause at a step without changing the learning-rate curve:** keep `train.max_steps` at the planned total and set
  `train.stop_at_step=20000`; later run again without it. (Changing `max_steps` itself between sessions changes the
  schedule - do not.) The run also stops cleanly at `train.time_budget_minutes` and resumes.
- **Hub backup (optional):** `--set train.hub_repo_id=user/repo` and an `HF_TOKEN` (Kaggle secret). The latest full
  checkpoint is uploaded every `train.hub_upload_every_n_saves` saves and at the end; on a fresh session the same command
  restores it first. Upload failures never stop training and nothing is ever deleted from the Hub. Each full upload
  (~6 GB) adds storage that the Hub does not free by itself - remove old versions by hand (Settings -> Storage).
- **Data check on resume:** the run stores a fingerprint of its training tokens and refuses to resume on different data.
- **Hugging Face export:** `--set train.export_hf=true` writes `hf_model/` (safetensors; reloaded and compared with the
  trained model). Load with `AutoModelForCausalLM.from_pretrained(repo, subfolder="hf_model")`.
- **Out-of-domain perplexity:** `scripts/evaluate.py --run-dir ... --name wikitext --source-json '{...}'` (see the file).
- **Vocabulary:** `model.auto_vocab=true` (default) sizes the embedding from the tokenizer (rounded up to a multiple of 128).
  Set `"auto_vocab": false, "vocab_size": 151936` to match official Qwen2.5 / earlier checkpoints
  (`configs/english.json` does this: 494,032,768 parameters).

## Added for the Kaggle notebooks (v0.3)
- `train.schedule_start_step=N`: the learning-rate schedule restarts at step N (new warm-up / stable / decay over the
  remaining steps). Use it to continue a run that already decayed: a plain longer plan would jump from the decayed rate
  back to the old peak.
- `data.source.params.skip_docs=N`: ignore the first N documents (text an earlier run already trained on).
- `HubBackup.prune_stale_lfs()`: frees old overwritten checkpoint copies on the Hub. It compares `file_oid` (never `oid`),
  deletes nothing unless the current file is positively identified, and prints what it removes.
- The trainer refuses to save a checkpoint whose weights contain NaN/inf, and does not save the same step twice.

## Results (Kaggle T4, measured)
All four models use the same Qwen tokenizer, so their perplexities are comparable. Text: 200 windows of 512 tokens,
FineWeb-Edu documents taken after the first 300,000 (unseen by every from-scratch run) and the WikiText-103 test set.
Lower is better.

| Model | FineWeb-Edu perplexity | WikiText-103 perplexity |
|---|---|---|
| v1 scratch (old pipeline, 20,000 steps; its last stage mixed in task data) | 71.48 | 138.54 |
| **v2 modular, 30,000 steps** (the v1 model + 10,000 steps of clean continuous text) | **41.97** | **93.54** |
| fine-tuned Qwen2.5-0.5B (pretrained start) | 15.69 | 18.79 |
| pretrained Qwen2.5-0.5B | 16.52 | 16.85 |

During the 10,000-step run (held-out text inside the training notebook, a different set of windows): FineWeb-Edu
perplexity 66.73 -> 46.95 and WikiText-103 131.74 -> 90.88 (loss 4.20 -> 3.85). About 246 million training tokens in total.

**Repetition.** One greedy generation of 600 tokens from "Once upon a time": share of repeated 4-grams in the first 512
tokens (inside the training context window) and after them (beyond it):

| Model | inside the window | beyond the window |
|---|---|---|
| v1 scratch | 95.8% | 82.0% |
| v2 modular 30k | 94.3% | 83.1% |
| fine-tuned | 96.2% | 88.8% |
| pretrained | 73.7% | 37.1% |

Repetition is as high inside the window as beyond it for every model, so running past the context window does not
explain the looping (one prompt, greedy decoding). Generation is still held inside the window by the guard-rails.
`generate.py` also offers `repetition_penalty` and `no_repeat_ngram_size`; their effect has not been measured yet.

**How to read this.** The 41% (FineWeb-Edu) and 32.5% (WikiText) perplexity gains come from 10,000 more steps on clean text together; the two causes
are not separated. The model is still about 2.5x (FineWeb-Edu) and 5.5x (WikiText) behind the pretrained model, as
expected from a few hundred million training tokens against trillions.

## Where things are
- `kaggle_run/`: the notebooks as executed on a Kaggle T4, with their outputs: `Run1_Baseline_Evaluation` (the three existing
  models), `Run2_Train_20k_to_30k_HF_Backups` (the 10,000-step run with Hugging Face checkpoint backups) and
  `Run3_Final_Comparison` (the table above). They call the package functions directly.
- `kaggle/run_pipeline.ipynb`: a runner for `scripts/train.py`; not executed on Kaggle yet.
- Models are in private Hugging Face repositories (`Thesis_Scratch_1`, `Thesis_Modular_v2_30k`); access can be shared on request.
