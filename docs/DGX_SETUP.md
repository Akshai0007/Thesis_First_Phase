# DGX A100 setup, Hugging Face connection, and first run (step by step)

Do the Kaggle proof first (`kaggle/tournament_kaggle.ipynb`, all phases passing). This guide gets the same code running on the DGX (80 GB A100s, GPUs 0-6 only).
Time: about 1 hour of setup, then the runs in `TOURNAMENT_RUNBOOK.md`.

## Step 1. Log in and look around (5 min)

1. Open the JupyterHub address you were given and log in. **File > New > Terminal** opens a terminal (use it for everything below; if there is no Terminal option, tell me: the notebook `dgx/tournament_dgx.ipynb` still works).
2. Run, and keep the output:
```bash
nvidia-smi                                   # which of GPUs 0-6 are free; GPU 7 is NOT yours
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
df -h ~                                      # free disk in your home directory
free -g | head -2                            # RAM
python3 --version
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # may fail: that is fine
curl -sI --max-time 10 https://huggingface.co | head -1      # "HTTP/2 200" = the DGX can reach Hugging Face
curl -sI --max-time 10 https://pypi.org | head -1            # can it install packages?
```
You need about **10 GB for the data cache + 6 GB per run that is still keeping a checkpoint**. If `df` shows less than ~60 GB free, ask the admin for a quota (through your supervisor; the policy says requests go supervisor -> Head -> sys.admin, cc cto@duk.ac.in and dgx@duk.ac.in).
If either `curl` prints nothing, the DGX has no internet: see "No internet" at the end before continuing.

## Step 2. Put the code on the DGX (5 min)

Option A (works without internet): in the JupyterHub file browser click **Upload**, choose `qwenlab_v04.zip`, then in the terminal:
```bash
cd ~ && unzip -q qwenlab_v04.zip && cd qwenlab && ls          # you should see pyproject.toml, src, scripts, configs, docs ...
```
Option B (internet): `git clone <your GitHub repo> ~/qwenlab` after committing v0.4 (see `CHANGES_v04.md`).

Upload your tokenizer too: `~/qwenlab/tokenizers/malayalam_unigram32k.model` (or any path; the notebook asks for it).

## Step 3. Python environment (15 min)

Keep everything inside your home directory (nothing system-wide, nothing that affects other users).
```bash
# 3a. does the system Python already have a CUDA build of torch? (output of step 1)
#     yes -> reuse it:   python3 -m venv --system-site-packages ~/venvs/qwenlab
#     no  -> isolated:   python3 -m venv ~/venvs/qwenlab
python3 -m venv --system-site-packages ~/venvs/qwenlab
source ~/venvs/qwenlab/bin/activate
pip install -U pip
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
If that prints `False` or fails, install torch for the CUDA version `nvidia-smi` shows (top right), for example CUDA 12.x:
`pip install torch --index-url https://download.pytorch.org/whl/cu124` (use the cu number closest to yours; if unsure, send me the `nvidia-smi` header).
Then:
```bash
cd ~/qwenlab
pip install -e ".[hf-archs]" sentencepiece pyarrow ipykernel
python -c "import torch,transformers; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), transformers.__version__)"   # True, A100-SXM4-80GB, >=4.53
python -m ipykernel install --user --name qwenlab --display-name "qwenlab (venv)"             # makes this environment selectable in notebooks
```
In a notebook choose **Kernel > Change kernel > "qwenlab (venv)"**. (Skipping this is the most common reason a notebook cannot import qwenlab.)

## Step 4. Connect Hugging Face (10 min)

You already have a write token on Kaggle. For the DGX, make a **separate, narrower token** so it can be revoked on its own and so a shared DGX account cannot damage anything else on your Hugging Face account.

1. On huggingface.co: **Settings > Access Tokens > Create new token > Fine-grained**. Name: `dgx-malayalam`. Under *Repositories* add only `Akzhai/malayalam-lm-tournament` with write access. No other permissions. Copy the token (shown once).
2. The repo must already exist (the Kaggle upload cell creates it on first use; or create a private model repo with that name on the website).
3. Give the token to the DGX **without saving it in a file or notebook**:
   * Terminal: `read -rs HF_TOKEN && export HF_TOKEN` (paste, Enter; nothing is echoed). It lives only in that terminal. Start `jupyter`-independent jobs from the same terminal.
   * Notebook: the DGX notebook asks for it with a hidden `getpass` prompt and keeps it in kernel memory only. Never paste a token into a notebook cell or commit it.
   * Why not a file: your DGX ID may be a shared ID (MTech students get shared IDs under the policy), so a token file in the home directory could be read by everyone using that ID.
4. Test the connection (terminal with the venv active and `HF_TOKEN` exported):
```bash
export HF_HOME=~/hf_cache                      # cache inside your home directory
python - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
print("logged in as:", api.whoami()["name"])
repo = "Akzhai/malayalam-lm-tournament"
api.upload_file(path_or_fileobj=b"dgx connection test", path_in_repo="dgx/connection_test.txt", repo_id=repo, repo_type="model")
print("write test OK ->", f"https://huggingface.co/{repo}/tree/main/dgx")
PY
```
A `401/403` means the token lacks write access to that repo (re-create it with the repo added). Delete `dgx/connection_test.txt` on the website afterwards.
5. If the dataset is gated, open it once on huggingface.co while logged in as the same account and accept its terms.
6. When the project ends: revoke the token on the website.

## Step 5. Configure data and tokenizer (5 min)

Edit `configs/malayalam.json` (or pass overrides): dataset (`data.source`), tokenizer path. The dataset is **streamed** from Hugging Face and only the tokenised cache is stored (uint16, about 2 bytes per token = ~3 GB for 1.5B tokens), which respects the "do not store massive datasets" rule. The cache is shared by all runs and all plans.

## Step 6. Dry run on the DGX (30 min)

Terminal (venv active, `HF_TOKEN` exported) or the notebook. Always check which GPUs are free first (`nvidia-smi`).
```bash
cd ~/qwenlab
python -m pytest tests -q -x                                                       # bf16 + real CUDA paths
PLAN=configs/tournament/race1_screen.json
python scripts/tournament.py --plan $PLAN --phase check                            # all candidates ok
nohup python scripts/tournament.py --plan $PLAN --phase prepare > prepare.log 2>&1 &   # data cache, 1-2 h
tail -f prepare.log                                                                # Ctrl-C leaves it running
```

## Step 7. Probe, budget, race (see the runbook for the full program)
```bash
python scripts/tournament.py --plan $PLAN --phase probe --gpus 0                   # ~1 h, one free GPU
python scripts/tournament.py --plan $PLAN --phase budget --hours 5 --gpus 0,1,2,3,4,5
nohup python scripts/tournament.py --plan $PLAN --phase race --gpus 0,1,2,3,4,5 > race.log 2>&1 &
```
Use the GPU ids that were free in `nvidia-smi` (never 7; the launcher refuses it). Monitor:
```bash
tail -f race.log                                   # started / finished jobs
nvidia-smi                                         # all your GPUs busy?
tail -n 1 runs/tournament/race1_screen/*/metrics.jsonl      # loss, tok_s, peak_mem_gb of every run
python scripts/tournament.py --plan $PLAN --phase report    # ranking of what has finished so far
```
**If anything dies** (browser closed, network drop, crash): run the same `race` command again. Finished runs are skipped; unfinished runs resume from their last checkpoint (every 500 steps), or from Hugging Face if `hub_repo` was set and the local folder is gone.
To stop everything on purpose: `pkill -f tournament.py; pkill -f scripts/train.py` (only your own processes are affected).

## Step 8. Save the findings and clean up (every time)
```bash
python scripts/upload_results.py --plan $PLAN --repo Akzhai/malayalam-lm-tournament --subdir dgx
find runs/tournament -name checkpoints -type d -prune -exec rm -rf {} +      # after results.json exists in each run
```
Then, if you used a notebook: run its last cell (kernel shutdown). Finally **File > Hub Control Panel > Stop My Server**.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `refused: GPU 7 ...` | You asked for GPU 7. Use 0-6. |
| `CUDA out of memory` at start | Another user is on that GPU (`nvidia-smi`). The race retries once with half the micro-batch; pick other GPUs. |
| `FAILED: ImportError ... Gemma3TextConfig` in check | transformers too old: `pip install -U "transformers>=4.53"` or drop that candidate. |
| `ModuleNotFoundError: qwenlab` in a notebook | Wrong kernel: choose "qwenlab (venv)" (Step 3). |
| `401` / `403` from Hugging Face | Token lacks write access to the repo, or was revoked: recreate it (Step 4). |
| Download of the dataset is slow / stalls | Re-run `prepare`; it is deterministic. Check `curl` to huggingface.co. |
| Disk full | Delete `checkpoints/` of finished runs; `du -sh runs/* data_cache* ~/hf_cache`. |
| Loss `nan` / many spikes in one candidate | Re-run that candidate at half LR before discarding it (report column "spikes"). |

## No internet on the DGX
Ask the admin whether outbound HTTPS to huggingface.co and pypi.org can be enabled (policy: requests go through your supervisor). Meanwhile: on Kaggle run `pip download -d wheels ".[hf-archs]" sentencepiece pyarrow`, upload the `wheels/` folder and install with `pip install --no-index --find-links wheels ...`; build the data cache on Kaggle (`--phase prepare`) and upload the cache folder; skip `hub_repo`; copy `runs/tournament/<plan>/` (without `checkpoints/`) to your laptop through the JupyterHub file browser and upload it to Hugging Face from there.

## Policy reminders (from the DGX user policy)
GPU 7 is reserved; clear kernel memory at the end; stop your server; no massive datasets or data dumps; back up your own results; research use only; do not share credentials; unused accounts are intimated at 30 days and removed at 60.
