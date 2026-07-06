#!/bin/bash
#SBATCH --job-name=symcomp_ord
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=03:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=4
set -euo pipefail
: "${SYMCOMP_WORK_DIR:?export SYMCOMP_WORK_DIR first}"
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6
module load eth_proxy
source "${SYMCOMP_VENV:-$SYMCOMP_WORK_DIR/venvs/symcomp}/bin/activate"
export PYTHONPATH="$PWD"
python scripts/probe_order.py --workdir "$SYMCOMP_WORK_DIR" \
    --data_dir "$SCRATCH/symcomp/data" --stages AD,AX
echo "order probe done"
