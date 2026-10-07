# qwenlab

A modular, from-scratch pretraining and multitask SFT pipeline for Qwen-style decoder language models, built around the **Qwen2.5-0.5B** architecture (494,032,768 parameters, matching the published figure).

Every swappable component (model, init scheme, tokenizer, data source, task, LR schedule, eval task) registers itself under a string name, and configs refer to components only by that name. Changing the model, dataset or tokenizer is a config edit, not a code edit.

## Features

- **Qwen2.5 architecture from scratch** (GQA with 14 heads / 2 KV heads, RoPE, RMSNorm, SwiGLU MLP, QKV bias, tied embeddings), with a bridge to Hugging Face and a test that checks equivalence against the real HF Qwen2 implementation.
- **Typed, validated config**: JSON in, dataclasses out, `--set a.b.c=value` overrides. Unknown keys raise immediately, so typos never silently fall back to defaults.
- **Multiple training tasks**: causal LM, prompt/response (SFT), and fill-in-the-middle (FIM).
- **Weight init schemes** (e.g. GPT-2 scaled init) selectable by name.
- **LR schedules** registered by name.
- **Trainer** with chunked LM-head loss (the 151k-vocab logits are huge), optional gradient checkpointing, dtype resolution, time-budgeted runs and resumable checkpoints, built for Kaggle's 12-hour session limit.

## Project structure

```
qwenlab/
├── pyproject.toml
├── requirements.txt
├── src/qwenlab/
│   ├── config.py            # typed configs (ModelConfig, InitConfig, TrainConfig, ...)
│   ├── registry.py          # generic component registries
│   ├── init.py              # weight init schemes
│   ├── models/
│   │   ├── qwen.py          # Qwen2.5-style decoder LM
│   │   └── hf_bridge.py     # Hugging Face interoperability
│   ├── tasks/               # causal_lm, prompt_response, fim
│   ├── training/            # trainer.py, schedules.py
│   ├── data/                # data sources
│   ├── tokenization/        # tokenizers
│   └── evaluation/          # eval tasks
├── tests/                   # pytest suite
└── kaggle/
    ├── smoke_test.py            # GPU feasibility test
    └── qwenlab_kaggle.ipynb     # Kaggle runner notebook
```

## Installation

```bash
git clone https://github.com/<your-username>/qwenlab.git
cd qwenlab
pip install -e .
```

Requires Python 3.10+ and PyTorch 2.3+.

## Running the tests

```bash
pytest
```

The suite covers architecture equivalence against Hugging Face Qwen2, weight init, LR schedules, tasks and the trainer.

## Running on Kaggle

1. Upload this repo (or the `qwenlab` folder) as a Kaggle Dataset and attach it to a new notebook.
2. Set the accelerator to **GPU T4 x2** or **P100**.
3. Open `kaggle/qwenlab_kaggle.ipynb` and run it. It installs the package from the attached dataset, runs the test suite, restores a previous checkpoint if one is attached, runs the GPU smoke test, and shows where the checkpoint is saved.

### Resuming across sessions

A Kaggle session lasts at most 12 hours, so long runs span several sessions. At the end of a session, commit the notebook, create a Kaggle Dataset from `qwenlab_run/checkpoints/progress.pt`, and attach it as an extra input in the next session. The notebook finds and restores it automatically.

## Results

> TODO: add your training curves, evaluation scores, and a link to the trained checkpoint (host weights on Hugging Face or Kaggle; do not commit them to this repo).

| Experiment | Model | Data | Steps | Final loss | Eval |
|---|---|---|---|---|---|
| _fill in_ | | | | | |

## Status

Core architecture, init, tasks, trainer and tests are in place. Real tokenizer/data wiring and the full multi-session pretraining run are the next phase.

## License

MIT, see [LICENSE](LICENSE).
