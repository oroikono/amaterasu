# Euler Pipeline Checklist

This checklist validates the travel workflow:

1. Alienware can clone and push the private GitHub repo.
2. Alienware can SSH into Euler.
3. Euler can clone the repo.
4. Euler can submit a small Slurm GPU job.
5. Future SymComp runs write durable outputs to group work/project storage, not
   only personal scratch.

## Local Alienware Checks

The repo was cloned locally with HTTPS:

```powershell
git clone https://github.com/oroikono/amaterasu.git "$HOME\OneDrive\Έγγραφα\code\amaterasu"
```

GitHub SSH is not required for this clone path, but SSH push/pull will require
adding this machine's public key to GitHub.

## Euler SSH

From PowerShell on the Alienware:

```powershell
ssh euler.ethz.ch
```

If passwordless SSH fails, register this machine's public key with Euler:

```powershell
Get-Content $HOME\.ssh\id_ed25519.pub
```

Then add that public key to `~/.ssh/authorized_keys` on Euler using an
interactive login or ETH's documented SSH-key workflow.

## Clone On Euler

After logging into Euler:

```bash
mkdir -p ~/code
cd ~/code
git clone https://github.com/oroikono/amaterasu.git
cd amaterasu
```

If HTTPS prompts are annoying on Euler, add an SSH deploy key or configure GitHub
SSH access there. Do not put private keys or tokens in this repo.

## Slurm GPU Smoke Test

Submit from the repo root on Euler:

```bash
sbatch cluster/euler_gpu_smoke.sbatch
```

Monitor:

```bash
myjobs
squeue --me
```

Inspect output:

```bash
ls -lh euler_gpu_smoke_*.out euler_gpu_smoke_*.err
tail -100 euler_gpu_smoke_*.out
tail -100 euler_gpu_smoke_*.err
```

Euler's current Slurm docs use `--gpus=1` for one GPU and
`--gres=gpumem:<size>` for GPU memory. GPU nodes require shareholder GPU access.

## Durable Storage Probe

On Euler, identify the group/share and durable storage target:

```bash
my_share_info
lquota
```

Before real SymComp experiments, set a durable work directory, for example:

```bash
export SYMCOMP_WORK_DIR=/cluster/work/<group>/symcomp
export SYMCOMP_HOME_ARCHIVE=/cluster/home/$USER/symcomp_archive
```

Replace `<group>` with the actual Euler shareholder/group path. Do not hardcode
private paths into committed code.
