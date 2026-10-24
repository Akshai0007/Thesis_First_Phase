"""Tests for the swappable-pipeline layer: tokenizer, source, cleaning, factory, perplexity, guarded generation, config."""
import json
import math

import numpy as np
import pytest
import torch

from qwenlab.config import ExperimentConfig, QWEN2_5_0_5B, InitConfig, from_dict, load_config, save_config, to_dict
from qwenlab.data.preprocess import clean_document, is_task_formatted
from qwenlab.data.tokenizers import ByteTokenizer
from qwenlab.evaluation.perplexity import evaluate_lm, perplexity_from_loss
from qwenlab.evaluation.repetition import repeated_ngram_fraction
from qwenlab.inference.generate import generate_ids
from qwenlab.models.factory import build_model, resolve_model_config

TINY = dict(hidden_size=32, intermediate_size=64, num_layers=2, num_heads=4, num_kv_heads=2, loss_chunk_tokens=0)


def tiny_cfg(**kw):
    from qwenlab.config import ModelConfig
    return ModelConfig(**{**TINY, **kw})


# ---------------------------------------------------------------- cleaning
def test_qa_and_chat_documents_are_detected_and_dropped():
    prose = "The river runs down to the sea. " * 20
    assert clean_document(prose)[0] is not None
    qa = "Question: Who wrote the book?\nAnswer: A person who lived long ago and wrote about many things. " * 3
    assert clean_document(qa) == (None, "qa_format")
    assert is_task_formatted("<|im_start|>user\nhi<|im_end|>")
    assert clean_document("too short") == (None, "short")


# ---------------------------------------------------------------- factory
def test_embedding_follows_tokenizer_and_preset_is_not_mutated():
    before = QWEN2_5_0_5B.vocab_size
    tok = ByteTokenizer()                                  # 257 tokens
    cfg = resolve_model_config(tiny_cfg(vocab_pad_multiple=64), tok, context_len=128)
    assert cfg.vocab_size == 320 and cfg.max_seq_len == 128 and cfg.head_dim == 8
    model, cfg = build_model(tiny_cfg(vocab_pad_multiple=64), InitConfig(seed=1), tok, context_len=128)
    assert model.embed_tokens.weight.shape == (320, 32) and model.lm_head.weight is model.embed_tokens.weight
    assert QWEN2_5_0_5B.vocab_size == before               # the shared preset stays untouched


def test_two_tokenizer_sizes_same_code():
    class Tok:                                             # stands in for a Malayalam 32k tokenizer
        eos_id, pad_id = 2, 0
        def __len__(self): return 32000
    m, c = build_model(tiny_cfg(), InitConfig(seed=1), Tok(), 64)
    assert c.vocab_size == 32000 and m.embed_tokens.num_embeddings == 32000


# ---------------------------------------------------------------- perplexity
def test_perplexity_is_exp_of_token_weighted_loss():
    model, cfg = build_model(tiny_cfg(), InitConfig(seed=0), ByteTokenizer(), 16)
    x = torch.randint(0, 256, (4, 16)); y = torch.randint(0, 256, (4, 16))
    full = (x, y, None)
    res = evaluate_lm(model, lambda: full, 2, "cpu")
    assert res["tokens"] == 2 * 4 * 16
    assert res["perplexity"] == pytest.approx(math.exp(res["loss"]))
    # untrained model ~ uniform guessing: perplexity close to the vocabulary size
    assert res["perplexity"] == pytest.approx(cfg.vocab_size, rel=0.25)
    assert perplexity_from_loss(0.0) == 1.0


# ---------------------------------------------------------------- generation guard-rails
def test_generation_never_exceeds_context_or_cap():
    model, cfg = build_model(tiny_cfg(), InitConfig(seed=0), ByteTokenizer(), 32)   # RoPE table is only 32 long
    seen = []
    orig = model.forward

    def spy(ids, *a, **k):
        seen.append(ids.shape[1])
        return orig(ids, *a, **k)
    model.forward = spy
    prompt = list(range(100))                               # prompt much longer than the context
    out = generate_ids(model, prompt, context_len=32, max_new_tokens=10_000, max_new_tokens_cap=20)
    assert len(out) <= 20                                    # cap wins over the request
    assert max(seen) <= 32                                   # the model never saw more than context_len tokens


def test_greedy_is_deterministic_and_ngram_ban_works():
    model, _ = build_model(tiny_cfg(), InitConfig(seed=3), ByteTokenizer(), 32)
    a = generate_ids(model, [1, 2, 3], 32, max_new_tokens=15)
    b = generate_ids(model, [1, 2, 3], 32, max_new_tokens=15)
    assert a == b
    c = generate_ids(model, [1, 2, 3], 32, max_new_tokens=25, no_repeat_ngram_size=3)
    assert repeated_ngram_fraction([1, 2, 3] + c, 3) == 0.0


def test_repeated_ngram_fraction():
    assert repeated_ngram_fraction([1, 2, 3, 4, 5, 6], 4) == 0.0
    assert repeated_ngram_fraction([7] * 20, 4) > 0.9


# ---------------------------------------------------------------- config round trip
def test_config_round_trip_and_comment_keys(tmp_path):
    cfg = from_dict(ExperimentConfig, {"model": TINY, "data": {"seq_len": 128}})
    p = tmp_path / "c.json"
    save_config(cfg, str(p))
    again = load_config(str(p))                              # head_dim is derived, must not break reloading
    assert again.model.head_dim == 8 and again.data.seq_len == 128
    q = tmp_path / "d.json"
    q.write_text(json.dumps({"_note": "comment", "data": {"seq_len": 64}}))
    assert load_config([str(p), str(q)]).data.seq_len == 64
    with pytest.raises(ValueError):
        from_dict(ExperimentConfig, {"data": {"seq_length": 1}})     # typos still fail loudly


# ---------------------------------------------------------------- normalizers (language swap)
def test_malayalam_normalizer_converts_old_chillu_and_keeps_joiners():
    from qwenlab.data.normalizers import malayalam, default, script_ratio
    old = "അവ\u0d28\u0d4d\u200d"                     # consonant + virama + ZWJ  (old-style chillu n)
    assert malayalam(old) == "അവ\u0d7b"               # -> atomic chillu letter
    assert default(old) == old                       # the language-neutral normaliser leaves it alone
    assert malayalam("ക\u0d4d\u200cഷ") == "ക\u0d4d\u200cഷ"          # ZWNJ is meaningful and kept
    assert malayalam("a\u200bb\ufeffc\u00add") == "abcd"           # zero-width space, BOM, soft hyphen removed
    # vowel signs / virama are combining marks, not letters: "മലയാളം" has 4 letters (മ ല യ ള) + 3 Latin = 4/7
    assert script_ratio("മലയാളം abc", "malayalam") == pytest.approx(4 / 7, abs=0.01)


def test_script_filter_drops_wrong_language():
    text = "This is a long English paragraph about rivers and hills. " * 8
    assert clean_document(text, 100, "malayalam", "malayalam", 0.5) == (None, "wrong_script")
    assert clean_document(text, 100)[0] is not None


def test_english_to_malayalam_swap_is_config_only(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    from pathlib import Path
    from qwenlab.pipeline import load_experiment, run_training
    root = Path(__file__).resolve().parents[1]
    lines = [json.loads(l)["text"] for l in open(root / "sample_data/malayalam_sample.jsonl", encoding="utf-8")]
    from qwenlab.data.normalizers import malayalam
    (tmp_path / "t.txt").write_text("\n".join(malayalam(t) for t in lines), encoding="utf-8")
    spm.SentencePieceTrainer.train(input=str(tmp_path / "t.txt"), model_prefix=str(tmp_path / "ml"), vocab_size=600,
        model_type="unigram", pad_id=0, unk_id=1, eos_id=2, bos_id=-1, byte_fallback=True, character_coverage=1.0,
        normalization_rule_name="identity", minloglevel=2)
    ov = [f"tokenizer.name=sentencepiece", f"tokenizer.path={tmp_path}/ml.model",
          f"data.source.params.pattern={root}/sample_data/malayalam_sample.jsonl", "data.min_chars=100",
          "data.normalizer=malayalam", "data.script=malayalam", "data.min_script_ratio=0.5",
          f"train.output_dir={tmp_path}/run", f"data.cache_dir={tmp_path}/cache", "train.max_steps=20",
          "train.eval_every=10", "train.save_every=10", "generation.probe_prompts=[]"]
    r = run_training(load_experiment([str(root / "configs/smoke_cpu.json")], ov), log=lambda *a: None)
    assert r["val_perplexity"] < 600 and r["data_stats"]["dropped_wrong_script"] == 0
