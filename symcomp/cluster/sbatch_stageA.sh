#!/bin/bash
#SBATCH --job-name=symcomp_A
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-89          # 6 reps x 5 split-seeds x 3 init-seeds = 90 tasks
# ^ adjust: tasks = len(reps)*len(split_seeds)*len(init_seeds); both tasks
#   (prediction+discovery) run inside each task.
# NOTE: submit from the symcomp/ repo root and `mkdir -p logs` first --
#       SLURM will not create the log directory.

set -euo pipefail

# ---- durable storage guard (D10) --------------------------------------------
# Results must never live only on purgeable scratch. registry.py also enforces
# this, but failing here gives a clearer error before burning queue time.
: "${SYMCOMP_WORK_DIR:?export SYMCOMP_WORK_DIR=/cluster/work/<group>/symcomp (see docs/euler_pipeline.md)}"

# ---- Euler environment (ADJUST module versions to current cluster default) --
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
module load eth_proxy   # network access (wandb sync, pip) from compute nodes

# venv must NOT live on scratch: the 15-day purge would delete it mid-project
# and every subsequent job would die at this line. Default to work storage.
source "${SYMCOMP_VENV:-$SYMCOMP_WORK_DIR/venvs/symcomp}/bin/activate"

export WANDB_MODE=offline
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- map SLURM_ARRAY_TASK_ID -> (rep, split_seed, init_seed) -----------------
# Done inside the runner from a flat index for reproducibility. Durable run
# dirs + master CSV go under $SYMCOMP_WORK_DIR via symcomp/registry.py; raw
# data shards are read from scratch (regenerable).
python scripts/run_task.py \
    --config configs/default.yaml \
    --stage A \
    --task_index "$SLURM_ARRAY_TASK_ID" \
    --data_dir "$SCRATCH/symcomp/data" \
    --workdir "$SYMCOMP_WORK_DIR"

echo "task $SLURM_ARRAY_TASK_ID done"
