"""Universal sources, small-dataset split, tokenizer eos fallback, explicit vocab, stop_at_step, Hub backup,
HF export, out-of-domain evaluation, sentence markers."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from qwenlab.config import QWEN2_5_0_5B, InitConfig, PretrainDataConfig, SourceConfig, TokenizerConfig, ModelConfig
from qwenlab.data.pretrain_data import build_eval_windows, build_pretrain_data
from qwenlab.data.sources import build_source
from qwenlab.data.tokenizers import ByteTokenizer
from qwenlab.hub import HubBackup
from qwenlab.models.factory import build_model, resolve_model_config
from qwenlab.pipeline import load_experiment, run_training

ROOT = Path(__file__).resolve().parents[1]
LONG = "The river runs down from the hills to the sea and the people nearby use it every day. "


def src(kind, **params):
    return SourceConfig(kind=kind, params=params)


# ------------------------------------------------------------------ sources: formats and columns
def test_csv_jsonl_parquet_custom_and_multi_columns(tmp_path):
    rows = [{"title": f"T{i}", "body": f"B{i} " + LONG, "sentence": f"S{i}"} for i in range(3)]
    with open(tmp_path / "a.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (tmp_path / "a.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pq = pytest.importorskip("pyarrow.parquet"); import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "a.parquet")
    for ext in ("csv", "jsonl", "parquet"):
        one = list(build_source(src("text_files", pattern=str(tmp_path / f"a.{ext}"), text_field="sentence")))
        assert one == ["S0", "S1", "S2"]
        both = list(build_source(src("text_files", pattern=str(tmp_path / f"a.{ext}"), text_field=["title", "body"])))
        assert both[0].startswith("T0\nB0") and len(both) == 3


def test_wrong_column_name_lists_the_real_ones(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps({"content": "x"}), encoding="utf-8")
    with pytest.raises(KeyError, match="content"):
        list(build_source(src("text_files", pattern=str(tmp_path / "a.jsonl"), text_field="text")))
    with pytest.raises(FileNotFoundError):
        list(build_source(src("text_files", pattern=str(tmp_path / "none*.csv"))))


# ------------------------------------------------------------------ small datasets
def test_small_dataset_smaller_than_val_tokens_still_trains(tmp_path):
    docs = [LONG * 4 + f"item {i}" for i in range(40)]                       # ~ 12k byte-tokens in total
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps({"text": d}) for d in docs), encoding="utf-8")
    cfg = PretrainDataConfig(source=src("text_files", pattern=str(tmp_path / "d.jsonl")), cache_dir=str(tmp_path / "c"),
                             seq_len=64, val_tokens=1_000_000, min_chars=50)   # val_tokens far above the data size
    data = build_pretrain_data(cfg, TokenizerConfig(name="bytes"), ByteTokenizer(), log=lambda *a: None)
    total = data.stats["train_tokens"] + data.stats["val_tokens"]
    assert data.stats["train_tokens"] > 0.8 * total and len(data.val) >= 1


# ------------------------------------------------------------------ tokenizer eos fallback
def test_hf_tokenizer_falls_back_to_its_own_eos_and_never_uses_unk(tmp_path):
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from qwenlab.data.tokenizers import HFTokenizer
    vocab = {"[UNK]": 0, "</s>": 1, "hello": 2, "world": 3}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]")); t.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=t, unk_token="[UNK]", eos_token="</s>").save_pretrained(tmp_path)
    tok = HFTokenizer(path=str(tmp_path), eos_token="<|endoftext|>")         # Qwen's default is not in this vocabulary
    assert tok.eos_id == 1 and tok.eos_id != 0                              # its own </s>, not the <unk> id


# ------------------------------------------------------------------ vocabulary
def test_vocab_auto_vs_explicit():
    class Tok:
        eos_id, pad_id = 151643, 151643
        def __len__(self): return 151665                                    # Qwen2.5 tokenizer
    auto = resolve_model_config(QWEN2_5_0_5B, Tok(), 512)
    import dataclasses
    fixed = resolve_model_config(dataclasses.replace(QWEN2_5_0_5B, auto_vocab=False), Tok(), 512)
    assert auto.vocab_size == 151680 and fixed.vocab_size == 151936
    assert fixed.num_params() == 494_032_768                                 # the documented parameter count
    assert QWEN2_5_0_5B.vocab_size == 151936                                 # preset untouched


# ------------------------------------------------------------------ model guard
def test_too_long_input_gives_a_clear_error():
    m, _ = build_model(ModelConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_heads=4, num_kv_heads=2,
                                   loss_chunk_tokens=0), InitConfig(seed=0), ByteTokenizer(), 16)
    with pytest.raises(ValueError, match="context window"):
        m(torch.zeros(1, 17, dtype=torch.long))


# ------------------------------------------------------------------ sentence markers
def test_urdu_hindi_sentence_marks_stop_generation():
    from qwenlab.inference.generate import _SENT_ANY, generate_ids
    assert len(_SENT_ANY.findall("یہ ایک جملہ ہے۔ کیا یہ دوسرا ہے؟ تیسرا")) == 2
    assert len(_SENT_ANY.findall("यह वाक्य है। यह दूसरा॥ ")) == 2
    script = "ab۔ cd۔ efgh"                                                     # a model that "writes" this text

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, ids, labels=None):
            n = ids.shape[1] - 1                                             # prompt is 1 token
            out = torch.full((1, ids.shape[1], 2000), -1e9); out[0, -1, ord(script[min(n, len(script) - 1)])] = 0
            return out, None
    ids = generate_ids(Stub(), [1], context_len=64, max_new_tokens=12, max_sentences=2, decode=lambda x: "".join(map(chr, x)))
    assert "".join(map(chr, ids)) == "ab۔ cd۔"                              # stopped right after the 2nd sentence


# ------------------------------------------------------------------ stop_at_step keeps the schedule
def _ov(tmp, steps, extra=()):
    return [f"train.output_dir={tmp}/run", f"data.cache_dir={tmp}/cache", f"train.max_steps={steps}",
            "train.eval_every=10", "train.save_every=10", "train.log_every=5",
            f"data.source.params.pattern={ROOT}/sample_data/stories.jsonl", *extra]


def _lrs(path):
    return {r["step"]: r["lr"] for r in map(json.loads, Path(path).read_text().splitlines()) if "lr" in r}


def test_stop_at_step_pauses_without_changing_the_schedule(tmp_path):
    cfgs = [str(ROOT / "configs/smoke_cpu.json")]
    r1 = run_training(load_experiment(cfgs, _ov(tmp_path / "split", 40, ["train.stop_at_step=20"])), log=lambda *a: None)
    assert r1["final_step"] == 20 and r1["stopped_early"]
    r2 = run_training(load_experiment(cfgs, _ov(tmp_path / "split", 40)), log=lambda *a: None)       # resume, run to the end
    run_training(load_experiment(cfgs, _ov(tmp_path / "whole", 40)), log=lambda *a: None)
    assert r2["final_step"] == 40 and not r2["stopped_early"]
    a, b = _lrs(tmp_path / "split/run/metrics.jsonl"), _lrs(tmp_path / "whole/run/metrics.jsonl")
    assert a == pytest.approx(b)                                                                    # identical learning-rate curve


# ------------------------------------------------------------------ data changed between sessions
def test_resume_refuses_changed_data(tmp_path):
    cfgs = [str(ROOT / "configs/smoke_cpu.json")]
    run_training(load_experiment(cfgs, _ov(tmp_path, 10)), log=lambda *a: None)
    other = _ov(tmp_path, 20)
    other = [o for o in other if not o.startswith("data.source.params.pattern")] + \
            [f"data.source.params.pattern={ROOT}/sample_data/malayalam_sample.jsonl", f"data.cache_dir={tmp_path}/cache2"]
    with pytest.raises(RuntimeError, match="data changed"):
        run_training(load_experiment(cfgs, other), log=lambda *a: None)


# ------------------------------------------------------------------ Hub backup + restore (fake Hub = local folder)
class FakeApi:
    def __init__(self, root, fail_first=0):
        self.root, self.fails, self.calls = Path(root), fail_first, []
    def create_repo(self, repo_id, private, exist_ok): self.calls.append("create")
    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, commit_message):
        if self.fails > 0:
            self.fails -= 1; raise OSError("network down")
        dst = self.root / path_in_repo; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(path_or_fileobj, dst)
        self.calls.append(path_in_repo)
    def upload_folder(self, folder_path, path_in_repo, repo_id, commit_message):
        shutil.copytree(folder_path, self.root / path_in_repo, dirs_exist_ok=True); self.calls.append(path_in_repo + "/")
    def hf_hub_download(self, repo_id, filename, local_dir):
        src = self.root / filename
        if not src.exists():
            raise FileNotFoundError(filename)
        dst = Path(local_dir) / filename; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy(src, dst); return str(dst)


def test_hub_retries_and_never_raises(tmp_path):
    f = tmp_path / "x.bin"; f.write_bytes(b"1")
    hub = HubBackup("u/r", api=FakeApi(tmp_path / "hub", fail_first=2), wait=0, log=lambda *a: None)
    assert hub.upload_file(f, "x.bin") is True                              # succeeded on the 3rd try
    hub = HubBackup("u/r", api=FakeApi(tmp_path / "hub2", fail_first=99), wait=0, log=lambda *a: None)
    assert hub.upload_file(f, "x.bin") is False                             # gave up, did not raise
    assert hub.restore_file("missing.pt", tmp_path / "out" / "missing.pt") is False


def test_training_survives_a_lost_output_folder_via_hub(tmp_path):
    cfgs = [str(ROOT / "configs/smoke_cpu.json")]
    hubdir = tmp_path / "hub"
    mk = lambda: HubBackup("u/r", api=FakeApi(hubdir), wait=0, log=lambda *a: None)
    run_training(load_experiment(cfgs, _ov(tmp_path, 20, ["train.hub_upload_every_n_saves=1"])), log=lambda *a: None, hub=mk())
    assert (hubdir / "checkpoints/progress.pt").exists() and (hubdir / "data_fingerprint.txt").exists()
    shutil.rmtree(tmp_path / "run")                                          # the Kaggle session's disk is gone
    r = run_training(load_experiment(cfgs, _ov(tmp_path, 30, ["train.hub_upload_every_n_saves=1"])), log=lambda *a: None, hub=mk())
    steps = [x["step"] for x in map(json.loads, (tmp_path / "run/metrics.jsonl").read_text().splitlines()) if "train_loss" in x]
    assert r["final_step"] == 30 and min(steps) > 20                          # continued from the restored step 20


# ------------------------------------------------------------------ HF export + out-of-domain evaluation
def test_export_hf_round_trip(tmp_path):
    pytest.importorskip("transformers")
    from qwenlab.export import export_hf
    tok = ByteTokenizer()
    m, cfg = build_model(ModelConfig(hidden_size=32, intermediate_size=64, num_layers=2, num_heads=4, num_kv_heads=2,
                                     loss_chunk_tokens=0), InitConfig(seed=0), tok, 32)
    diff = export_hf(m, cfg, tok, tmp_path / "hf_model", log=lambda *a: None)
    assert diff < 1e-4 and (tmp_path / "hf_model/model.safetensors").exists()


def test_out_of_domain_windows_use_same_cleaning(tmp_path):
    (tmp_path / "o.csv").write_text("title,body\n" + "\n".join(f"T{i},{LONG * 3}" for i in range(10)), encoding="utf-8")
    cfg = PretrainDataConfig(seq_len=64, min_chars=50)
    w = build_eval_windows(cfg, ByteTokenizer(), src("text_files", pattern=str(tmp_path / "o.csv"), text_field=["title", "body"]), 5000)
    assert len(w) >= 1 and len(w[0]) == 65


# ------------------------------------------------------------------ safe LFS pruning (the oid vs file_oid trap)
class _Lfs:
    def __init__(self, file_oid, size=6_000_000_000, filename="progress.pt", oid=None):
        self.file_oid, self.size, self.filename, self.pushed_at = file_oid, size, filename, "t"
        self.oid = oid or ("pointer-" + file_oid)             # a DIFFERENT identifier, like the real thing


class _Tree:
    def __init__(self, sha): self.lfs = type("L", (), {"sha256": sha})()


class PruneApi:
    def __init__(self, head, stored):
        self.head, self.stored, self.deleted = head, stored, None
    def list_repo_tree(self, repo_id, recursive): return [_Tree(h) for h in self.head] + [type("D", (), {})()]
    def list_lfs_files(self, repo_id): return list(self.stored)
    def permanently_delete_lfs_files(self, repo_id, files): self.deleted = [f.file_oid for f in files]


def test_prune_deletes_only_old_copies_and_never_the_current_file():
    api = PruneApi(head=["cur"], stored=[_Lfs("old1"), _Lfs("cur"), _Lfs("old2")])
    msg = HubBackup("u/r", api=api, log=lambda *a: None).prune_stale_lfs()
    assert api.deleted == ["old1", "old2"] and "2 old copies" in msg


def test_prune_refuses_when_current_file_cannot_be_identified():
    # head sha matches nothing stored (what happens if the wrong identifier is compared) -> delete NOTHING
    api = PruneApi(head=["cur"], stored=[_Lfs("a"), _Lfs("b")])
    msg = HubBackup("u/r", api=api, log=lambda *a: None).prune_stale_lfs()
    assert api.deleted is None and "deleting nothing" in msg
    api2 = PruneApi(head=[], stored=[_Lfs("a")])
    assert HubBackup("u/r", api=api2, log=lambda *a: None).prune_stale_lfs().startswith("skipped") and api2.deleted is None


# ------------------------------------------------------------------ nan guard, skip_docs, schedule restart
def test_nan_weights_are_never_saved(tmp_path):
    from qwenlab.config import TrainConfig
    from qwenlab.training.trainer import Trainer
    m, _ = build_model(ModelConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_heads=4, num_kv_heads=2,
                                   loss_chunk_tokens=0), InitConfig(seed=0), ByteTokenizer(), 16)
    t = Trainer(m, TrainConfig(output_dir=str(tmp_path), precision="fp32"), device="cpu")
    t.save_checkpoint(1)
    good = (tmp_path / "checkpoints/progress.pt").read_bytes()
    with torch.no_grad():
        m.embed_tokens.weight[0, 0] = float("nan")
    with pytest.raises(FloatingPointError):
        t.save_checkpoint(2)
    assert (tmp_path / "checkpoints/progress.pt").read_bytes() == good


def test_skip_docs(tmp_path):
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps({"text": f"doc {i} " + LONG}) for i in range(10)), encoding="utf-8")
    docs = list(build_source(src("text_files", pattern=str(tmp_path / "d.jsonl"), skip_docs=7)))
    assert len(docs) == 3 and docs[0].startswith("doc 7")


def test_schedule_restart_has_no_jump():
    from qwenlab.config import TrainConfig
    from qwenlab.training.schedules import lr_multiplier
    plain = TrainConfig(max_steps=30000, warmup_steps=500, schedule="wsd", decay_fraction=0.2)
    assert lr_multiplier(20000, plain) == 1.0                       # why a plain 30k plan cannot continue a decayed run
    cont = TrainConfig(max_steps=30000, warmup_steps=200, schedule="wsd", decay_fraction=0.2, schedule_start_step=20000)
    vals = [lr_multiplier(s, cont) for s in (20000, 20199, 20200, 27999, 28000, 29999)]
    assert vals[0] < 0.01 and vals[1] == 1.0 and vals[2] == 1.0 and vals[3] == 1.0   # warms up from ~0, then flat
    assert vals[4] == 1.0 and 0.1 <= vals[5] < 0.2                  # decays over the last 20% of the remaining steps
    same = TrainConfig(max_steps=100, warmup_steps=10, schedule="wsd", decay_fraction=0.2)
    assert [lr_multiplier(s, same) for s in range(100)] == [lr_multiplier(s, TrainConfig(**{**same.__dict__, "schedule_start_step": 0})) for s in range(100)]


def test_no_duplicate_save_at_the_final_step(tmp_path):
    from qwenlab.config import TrainConfig
    from qwenlab.training.trainer import Trainer
    m, _ = build_model(ModelConfig(hidden_size=32, intermediate_size=64, num_layers=1, num_heads=4, num_kv_heads=2,
                                   loss_chunk_tokens=0), InitConfig(seed=0), ByteTokenizer(), 16)
    t = Trainer(m, TrainConfig(output_dir=str(tmp_path), precision="fp32", max_steps=6, save_every=3, log_every=3,
                               warmup_steps=1, micro_batch_size=2, grad_accum_steps=1, schedule="constant"), device="cpu")
    saves = []
    orig = t.save_checkpoint
    t.save_checkpoint = lambda step, extra=None, final=False: (saves.append(step), orig(step, extra, final))[1]
    calls = []
    x = torch.randint(0, 256, (2, 16))
    t.fit(lambda: (x, x, None), on_checkpoint=lambda step, path, final: calls.append((step, final)))
    assert saves == [3, 6]                          # step 6 is saved once, not twice
    assert calls == [(3, False), (6, False), (6, True)]   # but the callback still learns that the run is finished
