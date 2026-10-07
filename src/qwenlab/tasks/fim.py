"""fim: Fill-In-the-Middle, reformats a contiguous span of text as a
prefix/middle/suffix triple the model must predict causally.

This is the standard trick (used by Codex, StarCoder, and confirmed as part
of Qwen's own tokenizer design via its reserved fim_prefix/fim_middle/
fim_suffix special tokens) for getting "fill in what's missing" behavior out
of a model that can only ever attend backward. Two orderings are used, each
with its own document layout:

  PSM (prefix-suffix-middle): <fim_prefix>{pre}<fim_suffix>{suf}<fim_middle>{mid}
  SPM (suffix-prefix-middle): <fim_suffix>{suf}<fim_prefix>{pre}<fim_middle>{mid}

In both, the model sees prefix and suffix (in some order) BEFORE being asked
to generate the middle -- so at generation time it has already attended to
both sides of the gap, achieving the same practical goal as bidirectional
masked prediction (use context from both directions) without ever violating
causal attention. Real training mixes a configurable fraction of windows
into this format and leaves the rest as plain causal_lm -- controlled by
PretrainDataConfig.fim_rate, so this is an additive variation on the
pretraining stream, not a separate task most of the time.
"""
import random

from ..registry import TASKS


def split_for_fim(token_ids: list, rng: random.Random):
    """Pick two cut points inside a token sequence, splitting it into
    prefix / middle / suffix. The middle is what the model must generate."""
    n = len(token_ids)
    if n < 3:
        return token_ids, [], []
    a, b = sorted(rng.sample(range(1, n), 2)) if n > 2 else (1, n - 1)
    return token_ids[:a], token_ids[a:b], token_ids[b:]


def build_fim_example(token_ids: list, fim_prefix_id: int, fim_middle_id: int,
                       fim_suffix_id: int, spm_rate: float, rng: random.Random):
    prefix, middle, suffix = split_for_fim(token_ids, rng)
    if rng.random() < spm_rate:
        ids = [fim_suffix_id] + suffix + [fim_prefix_id] + prefix + [fim_middle_id] + middle
    else:
        ids = [fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle
    # prompt = everything up to and INCLUDING the fim_middle_id marker itself
    # (3 markers total: whichever ordering of prefix/suffix markers, plus the
    # middle marker) -- only the actual `middle` content after it is the real
    # target. Caught by a test: forgetting to count the middle marker itself
    # here silently loss-masked it as if it were predictable content, and
    # produced an ids/mask length mismatch. Deriving prompt_len from len(ids)
    # directly (rather than manually recomputing marker counts) makes this
    # robust by construction -- it can't drift out of sync again.
    prompt_len = len(ids) - len(middle)
    loss_mask = [False] * prompt_len + [True] * len(middle)
    return ids, loss_mask


@TASKS.register("fim")
class FIMTask:
    def __init__(self, fim_prefix_id: int, fim_middle_id: int, fim_suffix_id: int,
                 spm_rate: float = 0.5, seed: int = 0):
        self.fim_prefix_id = fim_prefix_id
        self.fim_middle_id = fim_middle_id
        self.fim_suffix_id = fim_suffix_id
        self.spm_rate = spm_rate
        self.rng = random.Random(seed)

    def transform_window(self, token_ids: list):
        """Takes one plain packed causal_lm window and reformats it as a FIM
        example -- this is what fim_rate-controlled mixing calls per-window
        on the data-loading side."""
        return build_fim_example(token_ids, self.fim_prefix_id, self.fim_middle_id,
                                  self.fim_suffix_id, self.spm_rate, self.rng)
