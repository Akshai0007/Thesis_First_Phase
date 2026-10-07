"""causal_lm: the core pretraining task -- next-token prediction over a
packed stream of raw text, no prompt/response structure at all.

This is the ONE task every decoder-only causal architecture is directly,
natively built for: at every position, predict the next token from
everything before it. No architecture change needed, no loss masking needed
(every position contributes to the loss) -- this is what "pretraining" means
for a model shaped like Qwen.
"""
import torch

from ..registry import TASKS


def pack_tokens(token_ids: list, seq_len: int, eos_id: int):
    """Concatenate documents (each followed by EOS) into one long stream,
    then cut into non-overlapping seq_len+1 windows (input = window[:-1],
    target = window[1:]). Standard pretraining packing -- maximizes how much
    of the data budget is actually spent on next-token predictions instead
    of padding."""
    buf = []
    for ids in token_ids:
        buf.extend(ids)
        buf.append(eos_id)
    n_windows = (len(buf) - 1) // seq_len
    windows = []
    for i in range(n_windows):
        chunk = buf[i * seq_len: i * seq_len + seq_len + 1]
        windows.append(chunk)
    return windows


@TASKS.register("causal_lm")
class CausalLMTask:
    """Produces (input_ids, labels, loss_mask=None) batches from a packed
    token stream. loss_mask is None here specifically -- every position is a
    real target, unlike the SFT-style tasks in prompt_response.py where a
    prompt span must be excluded from the loss."""

    def __init__(self, seq_len: int, eos_id: int):
        self.seq_len = seq_len
        self.eos_id = eos_id

    def build_windows(self, documents: list) -> list:
        return pack_tokens(documents, self.seq_len, self.eos_id)

    def make_batch(self, windows: list, device: str = "cpu"):
        arr = torch.tensor(windows, dtype=torch.long)
        input_ids, labels = arr[:, :-1].to(device), arr[:, 1:].to(device)
        return input_ids, labels, None
