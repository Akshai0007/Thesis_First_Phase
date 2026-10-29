"""Swappable tokenizers. The rest of the pipeline only ever sees the small interface below:

    tok.encode(text) -> list[int]      tok.decode(ids) -> str
    len(tok)  (vocabulary size)        tok.eos_id, tok.pad_id

To use a different tokenizer (e.g. a Malayalam SentencePiece model), change `tokenizer.name` and
`tokenizer.path` in the config. Nothing else changes: the model's embedding table is sized from len(tok)
by models/factory.py.
"""
from typing import List

from ..registry import TOKENIZERS


class HFTokenizer:
    """Any Hugging Face tokenizer (e.g. Qwen/Qwen2.5-0.5B), by hub id or local folder."""

    def __init__(self, path: str, eos_token: str = "<|endoftext|>", pad_token=None, **_):
        from transformers import AutoTokenizer  # imported lazily: only needed if this tokenizer is chosen
        self.tok = AutoTokenizer.from_pretrained(path)
        self.eos_id = self._find(eos_token, self.tok.eos_token_id, "end-of-text")
        pad_fallback = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.eos_id
        self.pad_id = self._find(pad_token, pad_fallback, "padding") if pad_token else pad_fallback

    def _find(self, token, fallback_id, what):
        """Id of `token` if the tokenizer really has it; otherwise the tokenizer's own id (e.g. </s>, <eos>).
        (convert_tokens_to_ids answers with the <unk> id for a missing token, so that is checked explicitly.)"""
        tid = self.tok.convert_tokens_to_ids(token) if token else None
        missing = tid is None or (self.tok.unk_token_id is not None and tid == self.tok.unk_token_id
                                  and token != self.tok.unk_token)
        if not missing:
            return tid
        if fallback_id is None:
            raise ValueError(f"{what} token {token!r} is not in this tokenizer and it defines no default: "
                             f"set tokenizer.eos_token to a token it has")
        print(f"[tokenizer] {token!r} not in vocabulary; using the tokenizer's own {what} id {fallback_id}")
        return fallback_id

    def __len__(self) -> int:
        return len(self.tok)          # includes added special tokens

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.tok.decode(list(ids))


class SentencePieceTokenizer:
    """A SentencePiece .model file, e.g. the Malayalam Unigram-32k tokenizer.
    Ids follow the SentencePiece model itself (pad/unk/eos are read from it)."""

    def __init__(self, path: str, **_):
        import sentencepiece as spm
        self.path = path
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self.eos_id = self.sp.eos_id()
        pad = self.sp.pad_id()
        self.pad_id = pad if pad >= 0 else self.eos_id
        assert self.eos_id >= 0, "this SentencePiece model has no end-of-document id (train it with eos_id=2)"

    def __len__(self) -> int:
        return self.sp.get_piece_size()

    def encode(self, text: str) -> List[int]:
        return self.sp.encode(text)

    def decode(self, ids) -> str:
        return self.sp.decode(list(ids))


class ByteTokenizer:
    """No-download fallback: one token per UTF-8 byte (ids 0-255) + end-of-text (256).
    Used by the unit tests and for a quick offline check of the whole pipeline."""

    def __init__(self, **_):
        self.eos_id = 256
        self.pad_id = 256

    def __len__(self) -> int:
        return 257

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


TOKENIZERS.register("hf")(HFTokenizer)
TOKENIZERS.register("sentencepiece")(SentencePieceTokenizer)
TOKENIZERS.register("bytes")(ByteTokenizer)


def build_tokenizer(tok_cfg):
    cls = TOKENIZERS.get(tok_cfg.name)
    return cls(path=tok_cfg.path, eos_token=tok_cfg.eos_token, pad_token=tok_cfg.pad_token)
