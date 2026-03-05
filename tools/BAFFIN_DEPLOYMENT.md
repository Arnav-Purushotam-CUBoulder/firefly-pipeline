# Baffin cluster notes (BioFrontiers IT)

These notes are distilled from the PDFs you provided:

- `Baffin Documentation.pdf`
- `Baffin FAQ.pdf`
- `Slurm Job Scheduler on Baffin.pdf`
- `Virtual Environments_ Python.pdf`

They’re written specifically for running `tools/train_multi_species_patch_classifiers.py`.

## Connect / login

- Login: `ssh <IdentiKey>@baffin.colorado.edu`
- There are two login nodes (`baffin-1`, `baffin-2`) behind `baffin.colorado.edu`.
- Don’t run compute-heavy work on login nodes; submit work via Slurm.

## Storage (important)

- Home: `/home/$USER` (backed up; default quota ~1TB; snapshots count toward quota).
- Scratch: `/scratch/isilon/$USER` (NOT backed up; recommended location for job I/O).
- Compute nodes are stateless; `/tmp` is memory-backed and reduces available RAM. BIT recommends using scratch for I/O instead of `/tmp`.

**Practical rule:** stage your dataset to `/scratch/isilon/$USER/...`, write training outputs there, then `rsync` the final `outputs/` back to your home (or a project share) at the end.

## Partitions (CPU)

From the FAQ (use `sinfo` to see what’s available/idle/down):

- `short-cascadelake` (default): up to 24h, ~1.5TB RAM/node, 192 cores/node
- `long-cascadelake`: up to 2 weeks, ~1.5TB RAM/node, 192 cores/node
- `short-ivybridge`: up to 24h, ~500GB RAM/node, 64 cores/node
- `long-ivybridge`: listed in the FAQ with ~500GB RAM/node, 64 cores/node

## GPUs (Slurm)

From the Slurm PDF:

- Request GPUs with `#SBATCH --gres=gpu:<N>` (up to **4 per node**).
- There’s an example GPU sbatch for the `a100-genoa` partition at:
  `/tmp/Example-Slurm-Scripts/a100-genoa_Slurm-example.sbatch`

Always verify GPU partitions on Baffin via `sinfo` (names can change).

## Interactive jobs (recommended for debugging)

From the FAQ / Slurm PDF:

```bash
srun --pty --partition=short-cascadelake --nodes=1 --ntasks=2 --mem=10G --job-name=interactive --time=04:00:00 bash
```

Then confirm you’re on a compute node:

```bash
hostname
squeue --me
```

## Modules (Lmod)

From the Baffin docs:

```bash
module avail
module spider
module list
module purge
module load <pkg>
module unload <pkg>
```

They show an example Python module: `module load python/3.14.2`.

Note: for ML/PyTorch workflows, you must use a Python version that PyTorch provides wheels for (commonly 3.10/3.11/3.12). The docs show `python/3.14.2` as an example module, but if `pip install torch ...` fails with “No matching distribution found”, switch to an older Python module and recreate your venv.

## Python virtual environments

From the “Virtual Environments: Python” PDF:

1) Start an **interactive** Slurm session (don’t build envs on login nodes).

2) Load a Python module (pick a version compatible with your packages; PyTorch usually requires 3.10–3.12):

```bash
module load python/3.14.2
```

3) Create + activate a venv:

```bash
mkdir -p ~/venvs
python3 -m venv ~/venvs/firefly
source ~/venvs/firefly/bin/activate
pip install --upgrade pip
```

4) Install packages (you’ll need at least `torch`, `torchvision`, `Pillow` for the trainer script).

## Data transfer (from Biofstorage to Baffin)

From the FAQ:

```bash
scp /mnt/z/filename.tar <IdentiKey>@baffin.colorado.edu:/home/<IdentiKey>
rsync /mnt/z/directory  <IdentiKey>@baffin.colorado.edu:/home/<IdentiKey>
```

For very large datasets with many small files, it’s usually faster to **tar** first on your local machine, upload the tarball, then extract on Baffin.

## Running the multi-species trainer

Script: `tools/train_multi_species_patch_classifiers.py`

### Expected dataset layout

Under `<RUNROOT>/datasets/`:

```
<species>/<version>/final dataset/{train,val,test}/{firefly,background}/*.png
```

The script auto-selects the newest `final dataset` per species.

### Option A: one Slurm job, multiple GPUs on one node

Example `sbatch` (adjust `--partition`, time, memory, CPU cores for your needs):

```bash
#!/bin/bash
#SBATCH --job-name=patchcls_all
#SBATCH --partition=a100-genoa
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/isilon/IDENTIKEY/slurm_%j_%x.out
#SBATCH --error=/scratch/isilon/IDENTIKEY/slurm_%j_%x.err

set -euo pipefail

RUNROOT="/scratch/isilon/$USER/firefly_patch_training"
REPO="$HOME/firefly_pipeline_repo"   # wherever you cloned this repo

mkdir -p "$RUNROOT"

# (optional) stage data from home -> scratch
# rsync -avh "$HOME/datasets/" "$RUNROOT/datasets/"

source ~/venvs/firefly/bin/activate
python3 "$REPO/tools/train_multi_species_patch_classifiers.py" \
  --root "$RUNROOT" \
  --gpus 4 \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 4

# stage outputs back to home for safekeeping
rsync -avh "$RUNROOT/outputs/" "$HOME/firefly_patch_training_outputs/"
```

### Option B: Slurm job array (one model per task, scales across nodes)

This is often the simplest way to ensure “1 GPU trains 1 model”.

1) On Baffin, run:

```bash
python3 tools/train_multi_species_patch_classifiers.py --root <RUNROOT> --list-jobs
```

2) Submit an array where `--array=0-(JOB_COUNT-1)%MAX_CONCURRENT`.

Example:

```bash
#!/bin/bash
#SBATCH --job-name=patchcls_array
#SBATCH --partition=a100-genoa
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-49%10
#SBATCH --output=/scratch/isilon/IDENTIKEY/slurm_%A_%a_%x.out
#SBATCH --error=/scratch/isilon/IDENTIKEY/slurm_%A_%a_%x.err

set -euo pipefail

RUNROOT="/scratch/isilon/$USER/firefly_patch_training"
REPO="$HOME/firefly_pipeline_repo"

source ~/venvs/firefly/bin/activate
python3 "$REPO/tools/train_multi_species_patch_classifiers.py" \
  --root "$RUNROOT" \
  --job-index "$SLURM_ARRAY_TASK_ID" \
  --epochs 100 \
  --batch-size 128 \
  --num-workers 4
```

This uses the script’s single-job mode (`--job-index`) so each array task trains exactly one model and writes to:

`<RUNROOT>/outputs/<job_name>/...`
