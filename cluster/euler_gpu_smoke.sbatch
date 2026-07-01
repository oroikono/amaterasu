#!/bin/bash
#SBATCH --job-name=symcomp_gpu_smoke
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=2G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10G
#SBATCH --output=euler_gpu_smoke_%j.out
#SBATCH --error=euler_gpu_smoke_%j.err

set -euo pipefail

echo "started_at=$(date -Is)"
echo "job_id=${SLURM_JOB_ID:-none}"
echo "node=${SLURMD_NODENAME:-$(hostname)}"
echo "cwd=$PWD"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

echo "== Slurm =="
command -v scontrol >/dev/null 2>&1 && scontrol show job "$SLURM_JOB_ID" || true

echo "== GPU =="
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || {
  echo "nvidia-smi not found or no Nvidia GPU visible"
  exit 1
}

echo "== Python =="
python3 --version || true
python3 - <<'PY'
try:
    import torch
    print("torch_version=", torch.__version__)
    print("torch_cuda_available=", torch.cuda.is_available())
    print("torch_cuda_device_count=", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("torch_cuda_device_0=", torch.cuda.get_device_name(0))
except Exception as exc:
    print("torch_check_failed=", repr(exc))
PY

echo "finished_at=$(date -Is)"
