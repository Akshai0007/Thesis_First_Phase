"""Text generation with hard guard-rails.

The model is trained on windows of `context_len` tokens (data.seq_len). Positions beyond that were never seen in
training, so generation must stay inside it:  prompt tokens + new tokens  <=  context_len.
This module enforces that no matter what the caller asks for:
  * max_new_tokens is clamped to `max_new_tokens_cap` (default 64, roughly two sentences) and to the room left;
  * a prompt that is too long is cut from the LEFT (the most recent text is kept);
  * the sliding window is never longer than context_len.
Optional repetition controls (repetition_penalty, no_repeat_ngram_size) and sentence/EOS stopping are provided.
"""
import re
from typing import List, Optional

import torch

# sentence-ending marks: Latin . ! ?   Devanagari danda । ॥   Urdu/Arabic full stop ۔ and question mark ؟
_ENDS = ".!?\u0964\u0965\u06d4\u061f"
_SENT_END = re.compile("[" + _ENDS + "]\\s*$")
_SENT_ANY = re.compile("[" + _ENDS + "](?:\\s|$)")


def _apply_repetition_penalty(logits, generated: List[int], penalty: float):
    if penalty == 1.0 or not generated:
        return logits
    idx = torch.tensor(sorted(set(generated)), device=logits.device)
    vals = logits[0, idx]
    logits[0, idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def _ban_repeated_ngrams(logits, ids: List[int], n: int):
    if n <= 1 or len(ids) < n:
        return logits
    prefix = tuple(ids[-(n - 1):])
    banned = [ids[i + n - 1] for i in range(len(ids) - n + 1) if tuple(ids[i:i + n - 1]) == prefix]
    if banned:
        logits[0, banned] = -float("inf")
    return logits


@torch.no_grad()
def generate_ids(model, prompt_ids: List[int], context_len: int, max_new_tokens: int = 48,
                 max_new_tokens_cap: int = 64, temperature: float = 0.0, top_k: Optional[int] = None,
                 repetition_penalty: float = 1.0, no_repeat_ngram_size: int = 0,
                 eos_id: Optional[int] = None, max_sentences: Optional[int] = None,
                 decode=None, device: Optional[str] = None) -> List[int]:
    """Returns ONLY the newly generated ids. temperature=0 -> greedy."""
    assert len(prompt_ids) > 0, "empty prompt"
    device = device or next(model.parameters()).device
    max_new = max(1, min(int(max_new_tokens), int(max_new_tokens_cap), context_len - 1))
    prompt_ids = list(prompt_ids)[-(context_len - max_new):]      # keep the most recent text; room for max_new
    ids = list(prompt_ids)
    was_training = model.training
    model.eval()
    new: List[int] = []
    for _ in range(max_new):
        window = torch.tensor([ids[-context_len:]], dtype=torch.long, device=device)   # never longer than context_len
        logits, _ = model(window)
        logits = logits[:, -1, :].float()
        logits = _apply_repetition_penalty(logits, ids[-context_len:], repetition_penalty)
        logits = _ban_repeated_ngrams(logits, ids, no_repeat_ngram_size)
        if temperature and temperature > 0:
            logits = logits / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        else:
            nxt = int(torch.argmax(logits, dim=-1))
        if eos_id is not None and nxt == eos_id:
            break
        ids.append(nxt)
        new.append(nxt)
        if max_sentences and decode is not None:
            text = decode(new)
            if len(_SENT_ANY.findall(text)) >= max_sentences and _SENT_END.search(text):
                break
    if was_training:
        model.train()
    return new


def generate_text(model, tokenizer, prompt: str, context_len: int, **kw) -> str:
    new = generate_ids(model, tokenizer.encode(prompt), context_len, eos_id=tokenizer.eos_id,
                       decode=tokenizer.decode, **kw)
    return tokenizer.decode(new)
