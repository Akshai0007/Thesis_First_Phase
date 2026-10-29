"""Optional Hugging Face Hub backup/restore so a Kaggle session that ends (or a lost output folder) never loses
training. Enabled by `train.hub_repo_id`; needs a token in the HF_TOKEN environment variable.

Safety rules:
  * a Hub failure NEVER stops training: every call is retried 3 times, then logged and skipped;
  * the only deletion is prune_stale_lfs(), which removes ONLY objects no current file refers to, with guards;
  * each full checkpoint upload adds a new ~6 GB copy to the repository's storage (old copies are not freed
    automatically), so uploads are rare (train.hub_upload_every_n_saves) - clean old versions by hand in the
    Hub settings (Storage -> List LFS files) after checking exactly what will go.
"""
import os
import shutil
import time
from pathlib import Path


class HubBackup:
    def __init__(self, repo_id: str, api=None, token: str = None, retries: int = 3, wait: float = 5.0, log=print):
        self.repo_id, self.retries, self.wait, self.log = repo_id, retries, wait, log
        if api is None:
            from huggingface_hub import HfApi            # imported lazily: only needed when the Hub is used
            api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        self.api = api
        self._ready = False

    def _try(self, what: str, fn):
        for attempt in range(1, self.retries + 1):
            try:
                return True, fn()
            except Exception as e:                        # network, auth, storage full ... never fatal
                self.log(f"[hub] {what} failed (attempt {attempt}/{self.retries}): {type(e).__name__}: {str(e)[:200]}")
                if attempt < self.retries:
                    time.sleep(self.wait)
        return False, None

    def _ensure_repo(self) -> bool:
        if not self._ready:
            ok, _ = self._try("create/check repo", lambda: self.api.create_repo(self.repo_id, private=True, exist_ok=True))
            self._ready = ok
        return self._ready

    def upload_file(self, local_path, path_in_repo: str, message: str = "backup") -> bool:
        local_path = Path(local_path)
        if not local_path.exists() or not self._ensure_repo():
            return False
        size = local_path.stat().st_size / 1e9
        ok, _ = self._try(f"upload {path_in_repo} ({size:.2f} GB)", lambda: self.api.upload_file(
            path_or_fileobj=str(local_path), path_in_repo=path_in_repo, repo_id=self.repo_id, commit_message=message))
        self.log(f"[hub] {'uploaded' if ok else 'SKIPPED'} {path_in_repo}")
        return ok

    def upload_folder(self, local_dir, path_in_repo: str, message: str = "export") -> bool:
        if not Path(local_dir).exists() or not self._ensure_repo():
            return False
        ok, _ = self._try(f"upload folder {path_in_repo}", lambda: self.api.upload_folder(
            folder_path=str(local_dir), path_in_repo=path_in_repo, repo_id=self.repo_id, commit_message=message))
        self.log(f"[hub] {'uploaded' if ok else 'SKIPPED'} folder {path_in_repo}")
        return ok

    def prune_stale_lfs(self, dry_run: bool = False) -> str:
        """Free Hub storage taken by OLD versions of files that were overwritten (e.g. earlier progress.pt copies).

        An LFS object is kept if its SHA-256 (`file_oid`) belongs to ANY file at the current head of the repo.
        (`oid` is a different identifier: comparing it with the head's SHA-256 would make every object look
        stale.) Safety rules, all enforced here:
          * the head must contain LFS files, and at least one stored object must match one of them,
            otherwise nothing is deleted;
          * if every stored object would be deleted, nothing is deleted;
          * the list is printed before anything is removed; failures never raise.
        Returns a one-line summary."""
        try:
            head = {f.lfs.sha256 for f in self.api.list_repo_tree(self.repo_id, recursive=True)
                    if getattr(f, "lfs", None)}
            stored = list(self.api.list_lfs_files(self.repo_id))
        except Exception as e:
            return f"skipped (could not list: {type(e).__name__})"
        matched = [f for f in stored if f.file_oid in head]
        if not head or not matched:
            return "skipped (current files could not be matched to stored objects - deleting nothing)"
        stale = [f for f in stored if f.file_oid not in head]
        if not stale:
            return "0 old copies"
        if len(stale) >= len(stored):
            return "skipped (everything looked stale - deleting nothing)"
        gb = sum(f.size for f in stale) / 1e9
        for f in stale:
            self.log(f"[hub]   stale: {f.filename} {f.size / 1e9:.2f} GB pushed {f.pushed_at}")
        if dry_run:
            return f"dry run: would delete {len(stale)} old copies ({gb:.1f} GB)"
        ok, _ = self._try("delete stale LFS objects", lambda: self.api.permanently_delete_lfs_files(self.repo_id, stale))
        return f"deleted {len(stale)} old copies ({gb:.1f} GB)" if ok else "delete failed (nothing lost)"

    def restore_file(self, path_in_repo: str, local_path) -> bool:
        """Download path_in_repo to local_path if local_path does not exist yet. False if absent / failed."""
        local_path = Path(local_path)
        if local_path.exists():
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)

        def fetch():
            got = self.api.hf_hub_download(repo_id=self.repo_id, filename=path_in_repo, local_dir=str(local_path.parent / ".hub_tmp"))
            shutil.move(got, local_path)
        ok, _ = self._try(f"restore {path_in_repo}", fetch)
        self.log(f"[hub] {'restored' if ok else 'nothing restored for'} {path_in_repo}")
        shutil.rmtree(local_path.parent / ".hub_tmp", ignore_errors=True)
        return ok
