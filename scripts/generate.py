"""Generate text from a trained run. Length is ALWAYS held inside the training context window.

    python scripts/generate.py --run-dir runs/english --prompt "The capital of France is"
    python scripts/generate.py --run-dir runs/english --prompt "Once upon a time" --repetition-check

--repetition-check generates far past the usual cap and reports looping (repeated 4-grams) inside vs beyond the
context window, to show whether repetition comes from the window or from the model itself.
"""
import argparse

import _bootstrap  # noqa: F401
from qwenlab.evaluation.repetition import repeated_ngram_fraction
from qwenlab.inference.generate import generate_ids, generate_text
from qwenlab.pipeline import _gen_kwargs, load_trained

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True)
ap.add_argument("--prompt", required=True)
ap.add_argument("--max-new-tokens", type=int, default=None)
ap.add_argument("--repetition-check", action="store_true")
a = ap.parse_args()

cfg, tok, model, device = load_trained(a.run_dir)
ctx = cfg.data.seq_len
kw = _gen_kwargs(cfg)
if a.max_new_tokens:
    kw["max_new_tokens"] = a.max_new_tokens
print(f"context window = {ctx} tokens; max new tokens allowed = {min(kw['max_new_tokens'], kw['max_new_tokens_cap'])}")
print(generate_text(model, tok, a.prompt, ctx, **kw))

if a.repetition_check:
    # Diagnostic only: lengthen the position table by 150 and let the text run past the training window.
    cfg2, tok, model2, device = load_trained(a.run_dir, extra_positions=150)
    prompt_ids = tok.encode(a.prompt)
    total = ctx + 100
    new = generate_ids(model2, prompt_ids, context_len=ctx + 150, max_new_tokens=total, max_new_tokens_cap=10**9,
                       eos_id=None, device=device)
    cut = max(ctx - len(prompt_ids), 8)               # position where the text leaves the trained window
    print(f"repeated 4-grams: first {cut} generated tokens (inside the window) = {repeated_ngram_fraction(new[:cut]):.2f}; "
          f"tokens after it (beyond the window) = {repeated_ngram_fraction(new[cut:]):.2f}")
