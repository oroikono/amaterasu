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
private paths into committed code. `symcomp/symcomp/registry.py` reads these
variables at runtime and refuses to run on a cluster node without
`SYMCOMP_WORK_DIR` set. The venv also belongs under `$SYMCOMP_WORK_DIR/venvs/`
(NOT scratch — the purge would delete it mid-project).

### flock probe (run once before the Stage A array)

The registry serializes master-CSV appends with `fcntl.flock`. Lustre/NFS
mounts only honor that if mounted with flock support — verify on the actual
work filesystem before trusting concurrent appends:

```bash
cd ~/code/amaterasu/symcomp
SYMCOMP_TEST_DIR="$SYMCOMP_WORK_DIR" PYTHONPATH=. python tests/test_registry.py
```

`SYMCOMP_TEST_DIR` points the test's temp dirs at the work filesystem (they
would otherwise land on node-local /tmp and prove nothing). The 8-process
concurrent-append check ([6]) fails or errors if flock is absent/incoherent
on this mount. If it does, per-run `rows.csv` files are
still safe (no cross-task contention) — regenerate the union afterwards with
`python -c "from symcomp.registry import rebuild_master; rebuild_master()"`.
