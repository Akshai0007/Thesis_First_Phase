"""prompt_response: the shared mechanism behind summarize / qa / correction.

Why these three (and not a true bidirectional "masked" task) are the right
choice for THIS architecture:

Qwen, like every model built in this project's earlier phase, is
decoder-only and strictly causal (see model.py: attention is always
is_causal in the packed-pretraining sense; there is no bidirectional
attention mode here at all, unlike the earlier from-scratch project where a
separate bidirectional variant was built specifically to support masked-word
prediction). That is a genuine architectural constraint, not a decision made
per-task -- a causal model cannot look at future tokens, full stop.

Summarization and question-answering, however, do NOT need bidirectional
attention: both are naturally expressible as "read a prompt, generate a
response" -- exactly what a causal decoder already does natively. The only
thing needed is data formatting (wrap prompt+response in a chat template)
and a loss mask (only the response tokens should count toward the loss --
the model shouldn't be graded on predicting the prompt it was just given).
Correction (built in the earlier project too) is the same pattern.

True masked-word prediction is deliberately NOT offered as a task in this
pipeline -- it would need a second, bidirectional model, which defeats the
point of "train and later compare against the real (causal) pretrained
Qwen". FIM (fim.py) is the decoder-only-compatible substitute: Qwen's own
tokenizer already reserves fim_prefix/fim_middle/fim_suffix special tokens
(confirmed in tokenizer config research), meaning real Qwen was very likely
trained with infilling capability too -- FIM is what "masked-style" training
looks like for an architecture that can only ever look backward.
"""
from dataclasses import dataclass
from typing import List, Tuple

import torch

from ..registry import TASKS


@dataclass
class ChatExample:
    instruction: str    # the fixed task framing, e.g. "Summarize the following text."
    user_content: str   # the actual input (article to summarize, question to answer, etc.)
    response: str        # the target output


def render_chatml(example: ChatExample, im_start: str, im_end: str) -> Tuple[str, str]:
    """Returns (prompt_text, response_text) already ChatML-wrapped, split at
    the exact point where the loss mask should turn on."""
    prompt = (
        f"{im_start}system\n{example.instruction}{im_end}\n"
        f"{im_start}user\n{example.user_content}{im_end}\n"
        f"{im_start}assistant\n"
    )
    response = f"{example.response}{im_end}\n"
    return prompt, response


class PromptResponseTask:
    """Shared by summarize/qa/correction -- only the instruction text and
    the data source differ between them; the masking mechanism is identical,
    so it lives here once rather than being copy-pasted three times."""

    def __init__(self, tokenizer, seq_len: int, im_start: str, im_end: str):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.im_start = im_start
        self.im_end = im_end

    def encode_example(self, example: ChatExample):
        prompt, response = render_chatml(example, self.im_start, self.im_end)
        prompt_ids = self.tokenizer.encode(prompt)
        response_ids = self.tokenizer.encode(response)
        ids = prompt_ids + response_ids
        loss_mask = [False] * len(prompt_ids) + [True] * len(response_ids)
        return ids, loss_mask

    def make_batch(self, examples: List[ChatExample], pad_id: int, device: str = "cpu"):
        encoded = [self.encode_example(e) for e in examples]
        max_len = min(max(len(ids) for ids, _ in encoded), self.seq_len + 1)

        input_batch, label_batch, mask_batch = [], [], []
        for i, (ids, mask) in enumerate(encoded):
            ids, mask = ids[:max_len], mask[:max_len]
            pad_n = max_len - len(ids)
            ids = ids + [pad_id] * pad_n
            mask = mask + [False] * pad_n

            input_batch.append(ids[:-1])
            label_batch.append(ids[1:])
            # a target position's loss counts if the TARGET token (one ahead
            # of the input position) falls in the response span
            shifted_mask = mask[1:]
            if not any(shifted_mask):
                # Caught during testing: if seq_len truncates away the entire
                # response span (prompt alone already exceeds seq_len), this
                # example silently contributes zero loss -- wasted compute
                # with no visible sign anything is wrong. Same failure class
                # as the correction-task pre-flight guard from the earlier
                # project; a print here (not a hard assert) since a single
                # bad example in a large batch shouldn't crash a whole run,
                # but it must never be silent.
                print(f"WARNING: example {i}'s response was entirely truncated away "
                      f"(prompt length >= seq_len={self.seq_len}) -- contributes zero loss. "
                      f"Increase seq_len or shorten this example's prompt.")
            mask_batch.append(shifted_mask)

        x = torch.tensor(input_batch, dtype=torch.long, device=device)
        y = torch.tensor(label_batch, dtype=torch.long, device=device)
        m = torch.tensor(mask_batch, dtype=torch.bool, device=device)
        return x, y, m


@TASKS.register("summarize")
def make_summarize_task(tokenizer, seq_len, im_start, im_end, **_):
    return PromptResponseTask(tokenizer, seq_len, im_start, im_end)


@TASKS.register("qa")
def make_qa_task(tokenizer, seq_len, im_start, im_end, **_):
    return PromptResponseTask(tokenizer, seq_len, im_start, im_end)


@TASKS.register("correction")
def make_correction_task(tokenizer, seq_len, im_start, im_end, **_):
    return PromptResponseTask(tokenizer, seq_len, im_start, im_end)


# Default instruction framings -- data-source code supplies (user_content,
# response) pairs; the task registry entry above just needs the shared
# mechanism, these strings are used by the data-loading side when building
# ChatExample objects for each task.
DEFAULT_INSTRUCTIONS = {
    "summarize": "Summarize the following text.",
    "qa": "Answer the following question based on the given context.",
    "correction": "Correct the grammar and spelling errors in the following sentence.",
}
