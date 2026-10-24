"""Swappable raw-text sources. A source is a function that yields documents (plain strings).
Switching dataset = changing `data.source.kind` / `data.source.params` in the config; nothing downstream
knows where the text came from. Column names are parameters:

    "text_field": "content"                  one column
    "text_field": ["title", "body"]          several columns, joined with a newline

Local files (kind "text_files", glob pattern): .txt  .jsonl  .csv  .parquet
"""
import csv
import glob
import json
import sys
from typing import Iterator, List, Union

from ..registry import SOURCES

TextField = Union[str, List[str]]


def _fields(text_field: TextField) -> List[str]:
    return list(text_field) if isinstance(text_field, (list, tuple)) else [text_field]


def _row_text(row: dict, text_field: TextField) -> str:
    parts = [str(row[f]) for f in _fields(text_field) if row.get(f) not in (None, "")]
    return "\n".join(parts)


def _check_columns(row: dict, text_field: TextField, where: str):
    """Fail on the first row with the list of real column names, instead of silently yielding nothing."""
    missing = [f for f in _fields(text_field) if f not in row]
    if missing:
        raise KeyError(f"{where}: column(s) {missing} not found. Available columns: {sorted(row)}. "
                       f"Set data.source.params.text_field accordingly.")


def _limit_reached(n: int, max_docs) -> bool:
    return max_docs is not None and n >= max_docs


@SOURCES.register("hf_dataset")
def hf_dataset(path: str, text_field: TextField = "text", name: str = None, split: str = "train",
               data_dir: str = None, streaming: bool = True, max_docs: int = None, skip_docs: int = 0,
               **_) -> Iterator[str]:
    """Any Hugging Face dataset. Streams, so nothing large is downloaded up front.
    skip_docs: ignore the first N documents (e.g. text an earlier run already trained on)."""
    from datasets import load_dataset
    ds = load_dataset(path, name=name, split=split, data_dir=data_dir, streaming=streaming)
    if skip_docs:
        ds = ds.skip(skip_docs)
    n = 0
    for row in ds:
        if n == 0:
            _check_columns(row, text_field, f"dataset {path}")
        text = _row_text(row, text_field)
        if text.strip():
            yield text
            n += 1
            if _limit_reached(n, max_docs):
                return


def _rows_csv(fp: str):
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    with open(fp, encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def _rows_parquet(fp: str, text_field: TextField):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(fp)
    cols = [c for c in _fields(text_field) if c in pf.schema_arrow.names] or None
    for batch in pf.iter_batches(batch_size=1024, columns=cols):
        yield from batch.to_pylist()


def _rows_jsonl(fp: str):
    with open(fp, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


@SOURCES.register("text_files")
def text_files(pattern: str, text_field: TextField = "text", split_docs_on_blank_line: bool = False,
               max_docs: int = None, skip_docs: int = 0, **_) -> Iterator[str]:
    """Local files matched by a glob pattern: .jsonl / .csv / .parquet (columns named by text_field)
    and plain .txt (one document per file, or split on blank lines)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no files match {pattern!r}")
    n, skipped = 0, 0
    for fp in files:
        low = fp.lower()
        if low.endswith((".jsonl", ".csv", ".parquet")):
            rows = _rows_jsonl(fp) if low.endswith(".jsonl") else _rows_csv(fp) if low.endswith(".csv") \
                else _rows_parquet(fp, text_field)
            first = True
            for row in rows:
                if first:
                    _check_columns(row, text_field, fp)
                    first = False
                text = _row_text(row, text_field)
                if text.strip():
                    if skipped < skip_docs:
                        skipped += 1
                        continue
                    yield text
                    n += 1
                    if _limit_reached(n, max_docs):
                        return
            continue
        with open(fp, encoding="utf-8") as f:
            raw = f.read()
        for d in (raw.split("\n\n") if split_docs_on_blank_line else [raw]):
            if d.strip():
                if skipped < skip_docs:
                    skipped += 1
                    continue
                yield d
                n += 1
                if _limit_reached(n, max_docs):
                    return


def build_source(source_cfg) -> Iterator[str]:
    return SOURCES.get(source_cfg.kind)(**source_cfg.params)
