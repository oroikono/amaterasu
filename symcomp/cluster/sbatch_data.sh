#!/bin/bash
#SBATCH --job-name=symcomp_data
#SBATCH --output=%x_%A_%a.out
#SBATCH --time=02:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-49           # one task per operator shard; set to #operators-1

module load stack/2024-06 gcc/12.2.0 python/3.11.6
source "$SCRATCH/venvs/symcomp/bin/activate"
export PYTHONPATH="$PWD"

# Generates trajectories for the operator(s) assigned to this array index and
# writes them to $SCRATCH/symcomp/data/<canonical>/<noise>/shard.npz
python scripts/gen_data.py \
    --config configs/default.yaml \
    --shard_index "$SLURM_ARRAY_TASK_ID" \
    --outdir "$SCRATCH/symcomp/data"

echo "data shard $SLURM_ARRAY_TASK_ID done"
