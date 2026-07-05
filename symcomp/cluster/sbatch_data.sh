#!/bin/bash
#SBATCH --job-name=symcomp_data
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-49           # one task per operator shard; set to #operators-1
# NOTE: submit from the symcomp/ repo root and `mkdir -p logs` first --
#       SLURM will not create the log directory.

set -euo pipefail

# durable storage guard (D10) -- also hosts the venv, see below
: "${SYMCOMP_WORK_DIR:?export SYMCOMP_WORK_DIR=/cluster/work/<group>/symcomp (see docs/euler_pipeline.md)}"

# ---- Euler environment (ADJUST module versions to current cluster default) --
module load stack/2024-06 gcc/12.2.0 python/3.11.6
module load eth_proxy   # network access from compute nodes

# venv must NOT live on scratch: the 15-day purge would delete it mid-project
source "${SYMCOMP_VENV:-$SYMCOMP_WORK_DIR/venvs/symcomp}/bin/activate"

export PYTHONPATH="$PWD"

# Generates trajectories for the operator(s) assigned to this array index and
# writes them to $SCRATCH/symcomp/data/<canonical>/<noise>/shard.npz
# (shards are regenerable -> scratch is fine; the durable manifest copy is the
# registry's job once gen_data.py is implemented).
python scripts/gen_data.py \
    --config configs/default.yaml \
    --shard_index "$SLURM_ARRAY_TASK_ID" \
    --n_shards "${SLURM_ARRAY_TASK_COUNT:-50}" \
    --outdir "$SCRATCH/symcomp/data"

echo "data shard $SLURM_ARRAY_TASK_ID done"
