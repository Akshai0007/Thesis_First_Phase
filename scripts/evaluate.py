"""Held-out loss and perplexity of a trained run, optionally on another (out-of-domain) text source.

    python scripts/evaluate.py --run-dir runs/english
    python scripts/evaluate.py --run-dir runs/english --name wikitext \
        --source-json '{"kind":"hf_dataset","params":{"path":"Salesforce/wikitext","name":"wikitext-103-raw-v1","split":"test"}}'
    python scripts/evaluate.py --run-dir runs/english --name mytext \
        --source-json '{"kind":"text_files","params":{"pattern":"other/*.csv","text_field":["title","body"]}}'

The extra source is cleaned and tokenized exactly like the training text. Perplexity depends on the tokenizer,
so only compare numbers made with the same tokenizer.
"""
import argparse
import json

import _bootstrap  # noqa: F401
from qwenlab.config import SourceConfig, from_dict
from qwenlab.data.mixing import make_cycler
from qwenlab.data.pretrain_data import build_eval_windows, build_pretrain_data, lm_make_batch
from qwenlab.evaluation.perplexity import evaluate_lm
from qwenlab.pipeline import load_trained

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--source-json", default=None, help="evaluate on this other source instead of the run's own validation set")
ap.add_argument("--name", default="validation")
ap.add_argument("--max-tokens", type=int, default=200_000, help="amount of text to read from --source-json")
a = ap.parse_args()

cfg, tok, model, device = load_trained(a.run_dir)
if a.source_json:
    windows = build_eval_windows(cfg.data, tok, from_dict(SourceConfig, json.loads(a.source_json)), a.max_tokens)
else:
    windows = build_pretrain_data(cfg.data, cfg.tokenizer, tok).val          # cached; same validation text as training
bs = min(a.batch_size, len(windows))
batches = [lm_make_batch([windows[i + j] for j in range(bs)]) for i in range(0, (len(windows) // bs) * bs, bs)]
res = evaluate_lm(model, make_cycler(batches), len(batches), device)
res.update(set=a.name, vocab_size=cfg.model.vocab_size, uniform_guess_perplexity=cfg.model.vocab_size)
print(json.dumps(res, indent=2))
