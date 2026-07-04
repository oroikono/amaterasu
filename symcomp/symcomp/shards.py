"""shards.py -- on-disk trajectory shard IO, shared by gen_data (write) and
run_task (read).

Layout under a data dir (scratch is fine -- shards are regenerable):
  <data_dir>/<entry_dir>/sidecar.json           physics metadata (commutator,
                                                stratum, solver, seeds, ...)
  <data_dir>/<entry_dir>/noise<z>/shard.npz     u0 (M,N) f32, traj (M,T,N) f32
  <data_dir>/manifest_shard<k>.json             entry records written by array
                                                task k (merged on read)

An ENTRY is one operator variant: an S1 constant-coefficient operator (key =
canonical string), an S2 variable-coefficient variant (key = canonical@epsX),
or an S3 nonlinear variant (key = burgers@nuX). IC convention: the first
n_ic_train rows of a shard are the train pool, the last n_ic_test rows are the
eval pool -- readers must respect the split to keep IC-level hygiene.
"""
from __future__ import annotations
import glob
import hashlib
import json
import os
import re

import numpy as np


def entry_dirname(key: str) -> str:
    """Filesystem-safe, collision-proof directory name for an entry key."""
    slug = re.sub(r"[^A-Za-z0-9._+@=-]", "_", key)[:60]
    return f"{slug}-{hashlib.sha256(key.encode()).hexdigest()[:8]}"


def noise_tag(noise: float) -> str:
    return f"noise{noise:g}"


def shard_path(data_dir: str, key: str, noise: float) -> str:
    return os.path.join(data_dir, entry_dirname(key), noise_tag(noise),
                        "shard.npz")


def write_shard(data_dir: str, key: str, noise: float,
                u0: np.ndarray, traj: np.ndarray) -> str:
    p = shard_path(data_dir, key, noise)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, u0=u0.astype(np.float32),
                        traj=traj.astype(np.float32))
    os.replace(tmp, p)
    return p


def write_sidecar(data_dir: str, key: str, sidecar: dict) -> str:
    d = os.path.join(data_dir, entry_dirname(key))
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, "sidecar.json")
    with open(p, "w") as f:
        json.dump(sidecar, f, indent=2, sort_keys=True, default=str)
    return p


def write_manifest(data_dir: str, shard_index, records: dict) -> str:
    """records: key -> {dir, stratum, commutator, ...} for THIS array task."""
    p = os.path.join(data_dir, f"manifest_shard{shard_index}.json")
    with open(p, "w") as f:
        json.dump(records, f, indent=2, sort_keys=True, default=str)
    return p


def load_manifest(data_dir: str) -> dict:
    """Merge every manifest_shard*.json into one key -> record dict."""
    out: dict = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "manifest_shard*.json"))):
        with open(p) as f:
            out.update(json.load(f))
    if not out:
        raise FileNotFoundError(
            f"no manifest_shard*.json under {data_dir} -- run gen_data first")
    return out


def load_shard(data_dir: str, key: str, noise: float):
    """Return (u0 (M,N), traj (M,T,N)) float32 for one entry at one noise."""
    with np.load(shard_path(data_dir, key, noise)) as z:
        return z["u0"], z["traj"]


def manifest_hash(data_dir: str) -> str:
    """Stable hash of the merged manifest (for run reproducibility records)."""
    m = load_manifest(data_dir)
    blob = json.dumps(m, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
