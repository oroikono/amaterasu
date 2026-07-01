#!/bin/bash
#SBATCH --job-name=symcomp_A
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-89          # 6 reps x 5 split-seeds x 3 init-seeds = 90 tasks
# ^ adjust: tasks = len(reps)*len(split_seeds)*len(init_seeds); both tasks
#   (prediction+discovery) run inside each task.

# ---- Euler environment (ADJUST to current cluster default) ------------------
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 || module load eth_proxy
source "$SCRATCH/venvs/symcomp/bin/activate"

export WANDB_MODE=offline
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ---- map SLURM_ARRAY_TASK_ID -> (rep, split_seed, init_seed) -----------------
# Done inside the runner from a flat index for reproducibility.
python scripts/run_task.py \
    --config configs/default.yaml \
    --stage A \
    --task_index "$SLURM_ARRAY_TASK_ID" \
    --outdir "$SCRATCH/symcomp/runs/stageA"

echo "task $SLURM_ARRAY_TASK_ID done"
