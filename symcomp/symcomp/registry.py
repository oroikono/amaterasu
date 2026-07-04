"""registry.py -- durable run registry (defense against the scratch purge, D10).

Euler personal scratch is purged at 15 days and not backed up, so run outputs
must live on durable storage. This module is the single place that resolves
WHERE outputs go and HOW runs are recorded:

  work dir      SYMCOMP_WORK_DIR      e.g. /cluster/work/<group>/symcomp
                                      (local fallback: <repo>/work)
  home archive  SYMCOMP_HOME_ARCHIVE  e.g. /cluster/home/$USER/symcomp_archive
                                      (fallback: ~/symcomp_archive)

Layout under the work dir:
  runs/<run_id>/            one directory per run
      config.json           resolved config
      manifest.json         seeds, data hashes, param counts, git SHA, ...
      rows.csv              this run's result rows (master schema)
  results/master.csv        append-only, file-locked union of all rows

run_id = <UTC timestamp>-<git sha8>-<cell tag>, e.g.
         20260704T031500Z-f3b664f1-cell042

On a cluster (detected via $SCRATCH) SYMCOMP_WORK_DIR is REQUIRED -- we raise
rather than silently write to a purgeable location. Locally we default to
./work so dev machines need no setup. Data shards are regenerable and may
stay on scratch; everything registered here is the durable record.
"""
from __future__ import annotations
import csv
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone

# The fixed Stage-A master schema (CODEX_START item 3). aggregate.py and
# run_task.py must not drift from this. The registry prepends a run_id
# provenance column on write so requeued/re-run SLURM tasks can be deduped;
# callers never supply it.
MASTER_SCHEMA = ["stage", "encoder", "fusion", "backbone", "split_seed",
                 "init_seed", "task", "commutator", "metric_name",
                 "metric_value", "params"]
_CSV_FIELDS = ["run_id"] + MASTER_SCHEMA

# Never archive files with these extensions (checkpoints / raw data are either
# regenerable or belong on work storage, not home quota).
_ARCHIVE_SKIP_EXT = {".pt", ".pth", ".ckpt", ".npz", ".npy"}


def work_dir() -> str:
    """Resolve the durable work dir. Loud failure on clusters, easy locally.

    The local fallback is anchored to the symcomp repo root (not the CWD) so
    every invocation lands in the same registry regardless of where the
    script was started from.
    """
    d = os.environ.get("SYMCOMP_WORK_DIR")
    if d:
        return d
    if os.environ.get("SCRATCH"):
        raise RuntimeError(
            "SYMCOMP_WORK_DIR is not set but $SCRATCH exists, so this looks "
            "like a cluster node. Refusing to default to a purgeable path. "
            "Export SYMCOMP_WORK_DIR=/cluster/work/<group>/symcomp "
            "(see docs/euler_pipeline.md 'Durable Storage Probe').")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "work")


def home_archive_dir() -> str:
    return os.environ.get("SYMCOMP_HOME_ARCHIVE",
                          os.path.join(os.path.expanduser("~"),
                                       "symcomp_archive"))


def git_sha(short: int = 8) -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
        return sha[:short]
    except Exception:
        return "nogit"


def file_hashes(paths) -> dict[str, str]:
    """sha256 (first 16 hex chars) per file, for data-manifest provenance."""
    out = {}
    for p in paths:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[os.path.basename(p)] = h.hexdigest()[:16]
    return out


def new_run_id(cell_tag: str = "local") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{git_sha()}-{cell_tag}"


class Run:
    """Handle for one run directory under <work_dir>/runs/<run_id>/."""

    def __init__(self, run_id: str, root: str | None = None):
        self.root = root or work_dir()
        self.run_id = run_id
        self.dir = os.path.join(self.root, "runs", run_id)

    # ---- creation -----------------------------------------------------------
    @classmethod
    def create(cls, config: dict, cell_tag: str = "local",
               root: str | None = None, **manifest_extra) -> "Run":
        """Create runs/<run_id>/ with the resolved config and a manifest.

        manifest_extra: seeds, data_manifest_hash, param_counts, etc.
        """
        base = new_run_id(cell_tag)
        run = None
        for attempt in range(100):  # same-second re-runs get a -N suffix
            candidate = cls(base if attempt == 0 else f"{base}-{attempt + 1}",
                            root=root)
            try:
                os.makedirs(candidate.dir, exist_ok=False)
                run = candidate
                break
            except FileExistsError:
                continue
        if run is None:
            raise RuntimeError(f"could not allocate a unique run dir for {base}")
        try:
            with open(os.path.join(run.dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2, sort_keys=True, default=str)
            manifest = {
                "run_id": run.run_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "git_sha": git_sha(),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                **manifest_extra,
            }
            with open(os.path.join(run.dir, "manifest.json"), "w") as f:
                json.dump(manifest, f, indent=2, sort_keys=True, default=str)
        except BaseException:
            shutil.rmtree(run.dir, ignore_errors=True)  # no orphan run dirs
            raise
        return run

    # ---- results ------------------------------------------------------------
    def append_rows(self, rows: list[dict]) -> None:
        """Append result rows to this run's rows.csv AND the master CSV.

        rows.csv (one file per run, no cross-task contention) is the source
        of truth; master.csv is a convenience union that can always be
        regenerated with rebuild_master() if locking ever misbehaves on a
        shared filesystem.
        """
        for r in rows:
            missing = set(MASTER_SCHEMA) - set(r)
            extra = set(r) - set(MASTER_SCHEMA)
            if missing or extra:
                raise ValueError(
                    f"row does not match master schema: missing={sorted(missing)} "
                    f"extra={sorted(extra)}")
        stamped = [{**r, "run_id": self.run_id} for r in rows]
        _locked_append(os.path.join(self.dir, "rows.csv"), stamped)
        master = os.path.join(self.root, "results", "master.csv")
        os.makedirs(os.path.dirname(master), exist_ok=True)
        _locked_append(master, stamped)

    # ---- retrieval ----------------------------------------------------------
    @property
    def manifest(self) -> dict:
        with open(os.path.join(self.dir, "manifest.json")) as f:
            return json.load(f)

    @property
    def config(self) -> dict:
        with open(os.path.join(self.dir, "config.json")) as f:
            return json.load(f)

    def rows(self) -> list[dict]:
        p = os.path.join(self.dir, "rows.csv")
        if not os.path.exists(p):
            return []
        with open(p, newline="") as f:
            return list(csv.DictReader(f))

    # ---- archival -----------------------------------------------------------
    def archive_to_home(self, max_bytes: int = 32 * (1 << 20)) -> list[str]:
        """End-of-job copy of small artifacts to the home archive.

        Copies every file in the run dir except checkpoints/raw arrays and
        anything over max_bytes; also refreshes the master CSV copy. Returns
        the list of copied paths (for logging).
        """
        dest = os.path.join(home_archive_dir(), "runs", self.run_id)
        os.makedirs(dest, exist_ok=True)
        copied = []
        for name in sorted(os.listdir(self.dir)):
            src = os.path.join(self.dir, name)
            if not os.path.isfile(src):
                continue
            if os.path.splitext(name)[1].lower() in _ARCHIVE_SKIP_EXT:
                continue
            if os.path.getsize(src) > max_bytes:
                continue
            shutil.copy2(src, os.path.join(dest, name))
            copied.append(os.path.join(dest, name))
        master = os.path.join(self.root, "results", "master.csv")
        if os.path.exists(master) and os.path.getsize(master) <= max_bytes:
            dest_master = os.path.join(home_archive_dir(), "results")
            os.makedirs(dest_master, exist_ok=True)
            # copy to a per-run temp name, then atomically rename: concurrent
            # array tasks all refreshing this copy must never tear it
            tmp = os.path.join(dest_master, f".master.{self.run_id}.tmp")
            with open(master) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    shutil.copy2(master, tmp)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            os.replace(tmp, os.path.join(dest_master, "master.csv"))
            copied.append(os.path.join(dest_master, "master.csv"))
        return copied


def get_run(run_id: str, root: str | None = None) -> Run:
    """Fetch a run by ID; raises KeyError if it does not exist."""
    run = Run(run_id, root=root)
    if not os.path.isdir(run.dir):
        raise KeyError(f"run {run_id!r} not found under {run.root}/runs")
    return run


def list_runs(root: str | None = None) -> list[str]:
    d = os.path.join(root or work_dir(), "runs")
    if not os.path.isdir(d):
        return []
    return sorted(os.listdir(d))


def rebuild_master(root: str | None = None) -> str:
    """Regenerate results/master.csv from every run's rows.csv.

    The per-run rows.csv files are the source of truth (written with zero
    cross-task contention), so the master can always be reconstructed — e.g.
    if flock turns out to be unreliable on the cluster filesystem, or after
    manual pruning of requeued/duplicate runs. Atomic (write temp + rename).
    """
    r = root or work_dir()
    master = os.path.join(r, "results", "master.csv")
    os.makedirs(os.path.dirname(master), exist_ok=True)
    tmp = master + ".rebuild.tmp"
    n = 0
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        for run_id in list_runs(r):
            p = os.path.join(r, "runs", run_id, "rows.csv")
            if not os.path.exists(p):
                continue
            with open(p, newline="") as rf:
                for row in csv.DictReader(rf):
                    w.writerow(row)
                    n += 1
    os.replace(tmp, master)
    print(f"rebuilt {master}: {n} rows from {len(list_runs(r))} runs")
    return master


def _locked_append(path: str, rows: list[dict]) -> None:
    """Append rows to a CSV under an exclusive flock; write header if new.

    Append-only by construction: the file is opened in 'a' mode and existing
    content is never rewritten. Safe across concurrent SLURM array tasks on
    the same filesystem — PROVIDED the mount supports coherent flock (verify
    on Euler with the probe in docs/euler_pipeline.md; rebuild_master() is
    the recovery path if it does not).
    """
    with open(path, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                w.writeheader()
            else:
                # torn-tail guard: if a previous writer died mid-line, start
                # ours on a fresh line instead of merging into the fragment
                with open(path, "rb") as rf:
                    rf.seek(-1, os.SEEK_END)
                    if rf.read(1) != b"\n":
                        f.write("\n")
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
