"""Synthetic correction data."""
import random

from qwenlab.data.synthetic import corrupt, make_correction_pairs, split_sentences

S = ["The committee approved the new budget after a long discussion on Tuesday.",
     "Researchers found that regular exercise improves memory in older adults.",
     "She walked to the market and bought fresh vegetables for the family dinner."]


def test_corrupt_changes_the_text_and_keeps_most_of_it():
    rng = random.Random(0)
    for s in S * 20:
        bad = corrupt(s, rng)
        assert bad is not None and bad != s
        common = set(s.split()) & set(bad.split())
        assert len(common) >= len(s.split()) - 6          # a few words damaged, not a rewrite


def test_corrupt_is_deterministic_given_the_generator():
    assert [corrupt(s, random.Random(5)) for s in S] == [corrupt(s, random.Random(5)) for s in S]


def test_corrupt_refuses_text_it_cannot_damage():
    assert corrupt("a b c", random.Random(0)) is None
    assert corrupt("1234 5678 9012", random.Random(0)) is None


def test_split_sentences_keeps_only_clean_sentences():
    text = "Too short. " + S[0] + " ok. " + S[1] + "\n" + S[2] + " " + ("word " * 60) + "."
    got = split_sentences(text)
    assert S[0] in got and S[1] in got and S[2] in got
    assert all(8 <= len(g.split()) <= 40 for g in got)


def test_pairs_are_deterministic_and_the_target_is_the_clean_sentence():
    a = make_correction_pairs(S, 50, seed=3)
    assert a == make_correction_pairs(S, 50, seed=3) and len(a) == 50
    assert all(clean in S and bad != clean for bad, clean in a)
    assert a != make_correction_pairs(S, 50, seed=4)
