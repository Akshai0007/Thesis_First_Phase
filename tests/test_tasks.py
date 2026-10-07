import random
import pytest

from qwenlab.tasks.causal_lm import pack_tokens, CausalLMTask
from qwenlab.tasks.prompt_response import ChatExample, render_chatml, PromptResponseTask
from qwenlab.tasks.fim import split_for_fim, build_fim_example, FIMTask


# ----------------------------------------------------------------------- causal_lm
def test_pack_tokens_shifts_correctly_for_next_token_prediction():
    docs = [[1, 2, 3, 4, 5], [6, 7, 8]]
    windows = pack_tokens(docs, seq_len=4, eos_id=0)
    # stream = 1 2 3 4 5 0 6 7 8 0  (len 10) -> (10-1)//4 = 2 windows of len 5.
    # Windows stride by seq_len (4), each window is seq_len+1 (5) long, so
    # consecutive windows overlap by exactly 1 token -- standard pretraining
    # packing (same scheme nanoGPT uses): that shared token is what gives the
    # last position of window i a valid next-token target without wasting data.
    assert len(windows) == 2
    assert windows[0] == [1, 2, 3, 4, 5]
    assert windows[1] == [5, 0, 6, 7, 8]


def test_causal_lm_task_make_batch_input_is_target_shifted_by_one():
    task = CausalLMTask(seq_len=4, eos_id=0)
    windows = [[1, 2, 3, 4, 5]]
    x, y, mask = task.make_batch(windows)
    assert x.tolist() == [[1, 2, 3, 4]]
    assert y.tolist() == [[2, 3, 4, 5]]
    assert mask is None  # every position is a real target for pure pretraining


# ----------------------------------------------------------------------- prompt_response
class _CharTok:
    """Minimal tokenizer stand-in: one integer per character, for testing
    packing/masking MECHANICS without needing the real (internet-only) Qwen
    tokenizer. Deliberately not a real subword tokenizer."""
    def __init__(self):
        self._next = 0
        self._map = {}

    def encode(self, text):
        out = []
        for ch in text:
            if ch not in self._map:
                self._map[ch] = self._next
                self._next += 1
            out.append(self._map[ch])
        return out


def test_render_chatml_splits_prompt_and_response_at_the_right_point():
    ex = ChatExample(instruction="Do X.", user_content="input", response="output")
    prompt, response = render_chatml(ex, im_start="<|im_start|>", im_end="<|im_end|>")
    assert prompt.endswith("<|im_start|>assistant\n")
    assert response.startswith("output")
    assert "input" in prompt and "input" not in response


def test_prompt_response_loss_mask_only_covers_response_tokens():
    tok = _CharTok()
    task = PromptResponseTask(tok, seq_len=200, im_start="<S>", im_end="<E>")
    ex = ChatExample(instruction="I", user_content="U", response="RESP")
    ids, mask = task.encode_example(ex)
    assert len(ids) == len(mask)
    # the response text "RESP<E>\n" must be exactly where mask is True
    n_true = sum(mask)
    assert n_true == len("RESP<E>\n")  # response + its closing tag + newline
    # and those True positions must be a contiguous SUFFIX (response comes last)
    first_true = mask.index(True)
    assert all(mask[first_true:]), "loss mask has gaps or isn't a contiguous suffix"
    assert not any(mask[:first_true]), "loss mask is True somewhere before the response starts"


def test_prompt_response_make_batch_respects_padding_and_masking():
    tok = _CharTok()
    task = PromptResponseTask(tok, seq_len=150, im_start="<S>", im_end="<E>")
    examples = [
        ChatExample("I", "short", "ok"),
        ChatExample("I", "a much longer piece of input content here", "a longer response too"),
    ]
    x, y, mask = task.make_batch(examples, pad_id=999)
    assert x.shape == y.shape == mask.shape
    assert x.shape[0] == 2
    # every example must have SOME real (non-padding) loss-masked position
    assert mask.any(dim=1).all(), "an example ended up with zero loss-counted positions"


def test_prompt_response_warns_when_truncation_eats_entire_response(capsys):
    """The failure mode the fix above guards against: seq_len too small for
    the prompt alone must print a visible warning, not fail silently."""
    tok = _CharTok()
    task = PromptResponseTask(tok, seq_len=5, im_start="<S>", im_end="<E>")  # deliberately tiny
    examples = [ChatExample("instruction", "some real content here", "response")]
    task.make_batch(examples, pad_id=999)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out and "truncated away" in captured.out


# ----------------------------------------------------------------------- fim
def test_split_for_fim_reconstructs_original_sequence():
    rng = random.Random(0)
    ids = list(range(20))
    prefix, middle, suffix = split_for_fim(ids, rng)
    assert prefix + middle + suffix == ids


@pytest.mark.parametrize("spm_rate", [0.0, 1.0])
def test_build_fim_example_mask_length_and_middle_coverage(spm_rate):
    rng = random.Random(1)
    ids = list(range(30))
    out_ids, out_mask = build_fim_example(ids, fim_prefix_id=-1, fim_middle_id=-2,
                                           fim_suffix_id=-3, spm_rate=spm_rate, rng=rng)
    assert len(out_ids) == len(out_mask)
    n_true = sum(out_mask)
    # exactly one contiguous True suffix (the middle span) at the end
    first_true = out_mask.index(True) if n_true else len(out_mask)
    assert all(out_mask[first_true:])
    assert not any(out_mask[:first_true])
    # the special tokens themselves must never be loss-masked (they're
    # prompt structure, not something to predict)
    for marker in (-1, -2, -3):
        idxs = [i for i, t in enumerate(out_ids) if t == marker]
        assert all(not out_mask[i] for i in idxs), f"special token {marker} incorrectly loss-masked"


def test_fim_task_both_orderings_produce_valid_examples():
    task_spm = FIMTask(fim_prefix_id=-1, fim_middle_id=-2, fim_suffix_id=-3, spm_rate=1.0, seed=0)
    task_psm = FIMTask(fim_prefix_id=-1, fim_middle_id=-2, fim_suffix_id=-3, spm_rate=0.0, seed=0)
    window = list(range(50))

    ids_spm, mask_spm = task_spm.transform_window(window)
    ids_psm, mask_psm = task_psm.transform_window(window)

    # SPM starts with fim_suffix marker, PSM starts with fim_prefix marker
    assert ids_spm[0] == -3
    assert ids_psm[0] == -1
    assert len(ids_spm) == len(mask_spm)
    assert len(ids_psm) == len(mask_psm)
