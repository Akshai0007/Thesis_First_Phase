"""Synthetic 'correction' data: clean sentences + automatically damaged copies.

The model is trained to restore the clean sentence from the damaged one. This needs no extra dataset
(no licence or availability risk) and the damage types are exactly the mistakes people make:
typos (swapped / dropped / doubled letters), a missing small word, a doubled word, a lower-case start.
Everything is deterministic given the random generator, so data is reproducible and resume-safe.
"""
import random
import re
from typing import List, Optional, Tuple

_SMALL_WORDS = {"the", "a", "an", "of", "to", "in", "and", "is", "are", "was", "were", "that", "for", "on", "with", "as", "by"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _typo(word: str, rng: random.Random) -> str:
    """damage one word (needs >= 4 letters); the result always differs from the input"""
    for _ in range(6):
        i = rng.randrange(1, len(word) - 1)            # interior position, first letter stays
        kind = rng.choice(("swap", "drop", "double"))
        if kind == "swap":
            new = word[:i] + word[i + 1] + word[i] + word[i + 2:]
        elif kind == "drop":
            new = word[:i] + word[i + 1:]
        else:
            new = word[:i] + word[i] + word[i:]
        if new != word:
            return new
    return word[:-1]                                    # last resort: drop the final letter


def corrupt(text: str, rng: random.Random, max_typos: int = 3) -> Optional[str]:
    """Return a damaged copy of `text`, or None if the text is too short/odd to damage meaningfully."""
    words = text.split(" ")
    cand = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    if not cand:
        return None
    for i in rng.sample(cand, min(rng.randint(1, max_typos), len(cand))):
        words[i] = _typo(words[i], rng)
    if rng.random() < 0.30:                             # forget a small word
        idx = [i for i, w in enumerate(words) if w.lower() in _SMALL_WORDS]
        if idx and len(words) > 6:
            del words[rng.choice(idx)]
    if rng.random() < 0.20:                             # accidentally type a word twice
        i = rng.randrange(len(words))
        words.insert(i, words[i])
    out = " ".join(words)
    if rng.random() < 0.15 and out[:1].isupper():       # forget the capital letter
        out = out[0].lower() + out[1:]
    return out if out != text else None


def split_sentences(text: str, min_words: int = 8, max_words: int = 40) -> List[str]:
    """clean, self-contained sentences of a sensible length"""
    out = []
    for s in _SENT_SPLIT.split(text.replace("\n", " ")):
        s = s.strip()
        n = len(s.split())
        if min_words <= n <= max_words and s[-1:] in ".!?" and s[:1].isupper():
            out.append(s)
    return out


def make_correction_pairs(sentences: List[str], n: int, seed: int = 0) -> List[Tuple[str, str]]:
    """(damaged, clean) pairs, deterministic for a given seed. Sentences are visited in order, cycling if needed."""
    rng = random.Random(seed)
    pairs, i, tried = [], 0, 0
    while len(pairs) < n and sentences and tried < 20 * n + 100:
        clean = sentences[i % len(sentences)]
        i += 1
        tried += 1
        bad = corrupt(clean, rng)
        if bad is not None:
            pairs.append((bad, clean))
    return pairs
