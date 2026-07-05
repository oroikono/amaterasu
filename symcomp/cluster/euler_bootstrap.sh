#!/bin/bash
# euler_bootstrap.sh -- one-shot Euler setup + Stage A launch.
# Run ON AN EULER LOGIN NODE from the symcomp/ repo root:
#   bash cluster/euler_bootstrap.sh
# Idempotent: safe to re-run; skips what already exists. Heavy work happens
# in SLURM jobs, not on the login node (only venv/pip + second-scale tests
# run here). Set DRY_RUN=1 to stop before submitting jobs.
set -euo pipefail
cd "$(dirname "$0")/.."   # symcomp repo root

# ---- 1. durable storage (D10): resolve SYMCOMP_WORK_DIR -----------------
if [ -z "${SYMCOMP_WORK_DIR:-}" ]; then
    echo "SYMCOMP_WORK_DIR not set -- probing writable /cluster/work/<group> ..."
    mapfile -t CAND < <(for g in $(id -Gn); do
        [ -d "/cluster/work/$g" ] && [ -w "/cluster/work/$g" ] && echo "/cluster/work/$g"; done)
    if [ "${#CAND[@]}" -eq 1 ]; then
        export SYMCOMP_WORK_DIR="${CAND[0]}/symcomp"
        echo "  -> using $SYMCOMP_WORK_DIR"
    else
        echo "  candidates: ${CAND[*]:-none}"
        echo "  Could not pick automatically. Check 'my_share_info' / 'lquota'"
        echo "  and run:  export SYMCOMP_WORK_DIR=/cluster/work/<group>/symcomp"
        exit 1
    fi
fi
export SYMCOMP_HOME_ARCHIVE="${SYMCOMP_HOME_ARCHIVE:-$HOME/symcomp_archive}"
mkdir -p "$SYMCOMP_WORK_DIR" "$SYMCOMP_HOME_ARCHIVE" logs
lquota "$SYMCOMP_WORK_DIR" 2>/dev/null || true

# persist for future shells + sbatch submissions
ENVF="$HOME/.symcomp_env"
{ echo "export SYMCOMP_WORK_DIR=$SYMCOMP_WORK_DIR"
  echo "export SYMCOMP_HOME_ARCHIVE=$SYMCOMP_HOME_ARCHIVE"; } > "$ENVF"
grep -qF 'source ~/.symcomp_env' "$HOME/.bashrc" 2>/dev/null || \
    echo '[ -f ~/.symcomp_env ] && source ~/.symcomp_env' >> "$HOME/.bashrc"
echo "storage: work=$SYMCOMP_WORK_DIR  archive=$SYMCOMP_HOME_ARCHIVE (persisted to $ENVF)"

# ---- 2. modules + venv on WORK storage (never scratch) ------------------
module load stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 eth_proxy || {
    echo "module load failed -- adjust versions to the current Euler default"
    echo "(module avail python_cuda) and re-run."; exit 1; }
VENV="${SYMCOMP_VENV:-$SYMCOMP_WORK_DIR/venvs/symcomp}"
if [ ! -f "$VENV/bin/activate" ]; then
    python -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements.txt
python -c "import torch; print('torch', torch.__version__, 'cuda build:', torch.version.cuda)"

# ---- 3. validation on the cluster (seconds; login-node safe) ------------
PYTHONPATH=. python tests/test_physics.py | tail -1
PYTHONPATH=. python tests/test_solvers.py | tail -1
# flock/concurrency probe ON THE WORK FILESYSTEM (the whole point):
SYMCOMP_TEST_DIR="$SYMCOMP_WORK_DIR" PYTHONPATH=. python tests/test_registry.py | tail -1

if [ -n "${DRY_RUN:-}" ]; then echo "DRY_RUN set -- stopping before submission."; exit 0; fi

# ---- 4. submit: data array, then Stage A gated on it ---------------------
N_ENTRIES=$(PYTHONPATH=. python scripts/gen_data.py --config configs/default.yaml \
            --shard_index -1 --outdir /dev/null --list | tail -1 | awk '{print $2}')
echo "operator universe: $N_ENTRIES entries"
DATA_JOB=$(sbatch --parsable --array=0-$((N_ENTRIES-1)) cluster/sbatch_data.sh)
echo "data array submitted: job $DATA_JOB (entries 0-$((N_ENTRIES-1)))"
STAGEA_JOB=$(sbatch --parsable --dependency=afterok:"$DATA_JOB" cluster/sbatch_stageA.sh)
echo "Stage A array submitted: job $STAGEA_JOB (90 cells, runs after data)"

echo
echo "monitor:   squeue --me     |  tail -f logs/symcomp_A_*.out"
echo "results:   $SYMCOMP_WORK_DIR/results/master.csv"
echo "aggregate: PYTHONPATH=. python scripts/aggregate.py --csv $SYMCOMP_WORK_DIR/results/master.csv --task prediction --metric rel_l2"
