# qwenlab: how the pipeline meets the professor's requirements

## Mapping
| Requirement | Where | How it is checked |
|---|---|---|
| Next-word prediction on continuous text, no Q&A pairs | `data/pretrain_data.py`, `data/preprocess.py` | Documents with chat / fill-in-the-middle markers or `Question:/Answer:` layout are dropped and counted (`data_stats` in `results.json`); `scripts/train.py` refuses any stage except `pretrain`. Tests: `test_modular_pipeline.py`. |
| Repetition: stay inside the context window | `inference/generate.py`, `scripts/generate.py` | `data.seq_len` is the single context length; the model's position table follows it; generation clamps prompt + new tokens inside it, caps new tokens (64) and stops after about two sentences. A longer input raises a clear error. `--repetition-check` compares repetition inside vs beyond the window. Tests: `test_modular_pipeline.py`, `test_universal_data_and_resume.py`. |
| Perplexity, not only loss | `evaluation/perplexity.py`, `training/trainer.py` | Token-weighted loss and `exp(loss)` on a held-out set at regular intervals and at step 0; written to `metrics.jsonl`. |
| Config module | `config.py`, `configs/*.json` | One JSON per language/dataset; later files override earlier ones; typos in keys raise an error. |
| Swappable tokenizer and dataset | `data/tokenizers.py`, `data/sources.py`, `data/normalizers.py` | Registries; hf / SentencePiece / byte tokenizers; Hugging Face datasets and local .txt / .jsonl / .csv / .parquet with any column names. |
| Fresh Qwen2.5-0.5B weights, embedding follows the tokenizer | `models/qwen.py`, `models/factory.py`, `init.py` | Random initialisation (normal, depth-scaled residual projections) with checks; vocabulary from the tokenizer or set explicitly (151,936 = official Qwen2.5, 494,032,768 parameters). Logits equal the Hugging Face `Qwen2ForCausalLM` (test: `test_hf_equivalence.py`). |
| Checkpoints every N steps, safe resume | `training/trainer.py`, `pipeline.py`, `hub.py` | Atomic checkpoint writes, exact data position on resume, data fingerprint check, refuses to save NaN weights, optional Hugging Face backup/restore. |
| Inference script | `scripts/generate.py` | See above. |

## What was run
- `python -m pytest tests -q`: 92 passed on CPU and on the Kaggle T4 (about 47 s), including the Hugging Face equivalence test.
- Offline smoke run (`configs/smoke_cpu.json`): training, evaluation, generation and resume on a tiny model; perplexity starts near the vocabulary size and falls.
- Swaps done by config only on the tiny model: dataset, tokenizer (bytes -> SentencePiece), model size, English -> Malayalam sample.
- On a Kaggle T4 with the real data, tokenizer and Hugging Face Hub (through the notebooks in `kaggle_run/`, which call the package's trainer, data builder, evaluation and export):
  - baseline evaluation of the three existing models;
  - a dry run, then the main run from step 21,000 to 30,000 in fp16 (about 3.3 s/step; this session resumed from the run's own checkpoint at step 21,000, written by an earlier session): 18 checkpoint uploads to the Hub, each followed by deletion of exactly one old copy, resume from the uploaded checkpoint, export to `hf_model/` (logit difference 7.8e-05);
  - final comparison of four models (see README, Results).

## What has not been run yet
- `scripts/train.py` itself with `configs/english.json` on a GPU (the Kaggle runs used the same package functions through notebooks). `configs/english_demo.json` is the quick way to check the script end to end with the real tokenizer and dataset (needs internet).
- A from-scratch 0 -> 30,000 run of the new clean-text pipeline: the 30k model started from the earlier 20,000-step checkpoint.
- The effect of `repetition_penalty` / `no_repeat_ngram_size` on repetition.

## Known limits
- `configs/malayalam.json` is a template: it needs the trained tokenizer file and a check of the dataset id.
- `base.json` holds the English training values; Malayalam values still have to be chosen.
- The question-answering / summarisation / fill-in-the-middle task code is kept for a later stage and is not used by `scripts/train.py`.
