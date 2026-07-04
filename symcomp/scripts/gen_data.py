"""gen_data.py -- generate and shard trajectories per operator (SLURM array).

Each array index handles a slice of the operator universe (across all split
manifests' union) and all noise levels, writing per-entry shards + sidecars +
a per-task manifest (see symcomp/shards.py for the layout).

IMPLEMENTED (was a spec stub):
  1. universe = union of operators across all split seeds
     (symcomp.splits.make_split, which also asserts leakage hygiene per seed)
     + S2 variable-coefficient variants of the advection+diffusion anchor
     (epsilon sweep, WELL-POSED profiles: nu amplitude relative so nu(x)>0)
     + S3 Burgers (nu sweep) when the config rung is S1S2S3.
  2. entries are sorted by key and partitioned round-robin: entry i goes to
     array task (i mod n_shards).
  3. solvers: S1 exact spectral (machine zero); S2 spectral RK4 with
     stability-safe substep count (solver.stable_n_sub -- the fixed default
     would be UNSTABLE at N=256); S3 ETDRK4. All batched over ICs and
     validated against the single-IC solvers in tests/test_solvers.py.
     Trajectories are checked finite before anything is written.
  4. sidecar.json per entry records the analytic ||[A,B]||, stratum, solver
     settings, and IC seed; manifest_shard<k>.json maps key -> record.

IC convention: one rng per entry (seeded from the entry key), M = n_ic_train
+ n_ic_test ICs solved once; noise levels are drawn per level from the clean
trajectory (traj + z * std(traj_per_ic) * randn), so all noise levels share
ICs and clean dynamics. Readers take rows [:n_ic_train] for training and
[n_ic_train:] for eval.

Cubic reaction (S3, 8th-mechanism extension) is NOT generated: its ETDRK4
solver does not exist yet and unvalidated numerics must not enter the data.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symcomp import shards
from symcomp.operators import (Operator, make_varcoeff_profile,
                               variable_coeff_commutator_norm)
from symcomp.solver import (random_initial_condition, solve_constcoeff_batch,
                            solve_varcoeff_advdiff_batch, solve_burgers_batch,
                            stable_n_sub)
from symcomp.splits import make_split

S3_NUS = (0.1, 0.05, 0.02)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_entries(cfg: dict) -> list[dict]:
    """The full, deterministic entry list (identical on every array task)."""
    mechs = list(cfg["mechanisms"]["linear"])
    coeffs = {m: float(cfg["coeffs"][m]) for m in mechs}
    sp = cfg["split"]

    ops: dict[str, Operator] = {}
    for seed in sp["seeds"]:
        man = make_split(mechs, coeffs, int(seed), held_frac=sp["held_frac"],
                         max_terms=sp["max_terms"],
                         triples_sample=sp["triples_sample"])
        for o in man.train + man.test_compose + man.test_decompose:
            ops[o.canonical_str()] = o

    entries = [{"key": k, "stratum": "S1", "op": o}
               for k, o in sorted(ops.items())]

    anchor = Operator({"advection": coeffs["advection"],
                       "diffusion": coeffs["diffusion"]})
    for eps in cfg["data"]["epsilons"]:
        eps = float(eps)
        if eps == 0.0:
            continue  # eps=0 IS the S1 constant-coefficient anchor
        entries.append({"key": f"{anchor.canonical_str()}@eps{eps:g}",
                        "stratum": "S2", "op": anchor, "epsilon": eps})

    if cfg["experiment"]["rung"] == "S1S2S3":
        for nu in S3_NUS:
            entries.append({"key": f"burgers@nu{nu:g}", "stratum": "S3",
                            "op": Operator({"advection": 1.0, "diffusion": nu}),
                            "nu": nu})
    return entries


def _entry_rng(key: str, salt: str = "") -> np.random.Generator:
    h = hashlib.sha256((key + "|" + salt).encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def solve_entry(entry: dict, cfg: dict):
    """Solve one entry for all ICs. Returns (u0 (M,N), traj (M,T,N), sidecar)."""
    d = cfg["data"]
    N, T, t_max = int(d["N"]), int(d["T"]), float(d["t_max"])
    Ldom = 2 * np.pi
    M = int(d["n_ic_train"]) + int(d["n_ic_test"])
    t_eval = np.linspace(0.0, t_max, T)
    rng = _entry_rng(entry["key"])
    U0 = np.stack([random_initial_condition(N, Ldom, n_modes=int(d["ic_modes"]),
                                            rng=rng) for _ in range(M)])
    side = {"key": entry["key"], "canonical": entry["op"].canonical_str(),
            "coeffs": dict(entry["op"].coeffs), "stratum": entry["stratum"],
            "N": N, "T": T, "t_max": t_max,
            "n_ic_train": int(d["n_ic_train"]), "n_ic_test": int(d["n_ic_test"]),
            "ic_modes": int(d["ic_modes"])}

    if entry["stratum"] == "S1":
        traj = solve_constcoeff_batch(entry["op"], U0, t_eval, N, Ldom)
        side.update(commutator=0.0, solver="exact_spectral")
    elif entry["stratum"] == "S2":
        eps = entry["epsilon"]
        a0 = float(entry["op"].coeffs["advection"])
        nu0 = float(entry["op"].coeffs["diffusion"])
        a_field = make_varcoeff_profile(N, Ldom, a0, eps, k=1)
        # RELATIVE nu amplitude: nu(x) = nu0*(1 + eps*cos) stays positive for
        # eps < 1. Absolute amplitude makes nu(x) < 0 -> ill-posed blow-up.
        nu_field = make_varcoeff_profile(N, Ldom, nu0, eps * nu0, k=2)
        assert nu_field.min() > 0, "ill-posed S2 profile (nu <= 0 somewhere)"
        n_sub = stable_n_sub(np.abs(a_field).max(), np.abs(nu_field).max(),
                             N, Ldom, t_max)
        traj = solve_varcoeff_advdiff_batch(a_field, nu_field, U0, t_eval,
                                            N, Ldom, n_sub=n_sub)
        comm = variable_coeff_commutator_norm(a_field, nu_field, N, Ldom)
        side.update(commutator=float(comm), solver="spectral_rk4",
                    epsilon=eps, n_sub=n_sub, nu_min=float(nu_field.min()))
    elif entry["stratum"] == "S3":
        nu = entry["nu"]
        traj = solve_burgers_batch(nu, U0, t_eval, N, Ldom)
        # commutator PROXY for the nonlinear stratum (same convention as
        # dataset.build_benchmark; S3 ordering is qualitative, see DECISIONS)
        comm = variable_coeff_commutator_norm(
            make_varcoeff_profile(N, Ldom, 1.0, 0.5, 1),
            make_varcoeff_profile(N, Ldom, nu, 0.0, 2), N, Ldom)
        side.update(commutator=float(comm), solver="etdrk4", nu=nu,
                    commutator_is_proxy=True)
    else:
        raise ValueError(entry["stratum"])

    if not np.isfinite(traj).all():
        raise FloatingPointError(
            f"non-finite trajectory for {entry['key']} -- refusing to write")
    return U0, traj, side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard_index", type=int, required=True,
                    help="SLURM array index; -1 = generate everything")
    ap.add_argument("--n_shards", type=int, default=50)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--list", action="store_true",
                    help="print the entry list and exit (for sizing arrays)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    entries = build_entries(cfg)
    if a.list:
        for i, e in enumerate(entries):
            print(f"{i:3d} [{e['stratum']}] {e['key']}")
        print(f"total {len(entries)} entries x "
              f"{len(cfg['data']['noise_levels'])} noise levels")
        return

    assigned = entries if a.shard_index < 0 else entries[a.shard_index::a.n_shards]
    print(f"shard {a.shard_index}/{a.n_shards}: {len(assigned)} entries")
    os.makedirs(a.outdir, exist_ok=True)

    records = {}
    for e in assigned:
        t0 = time.time()
        U0, traj, side = solve_entry(e, cfg)
        shards.write_sidecar(a.outdir, e["key"], side)
        per_ic_std = traj.std(axis=(1, 2), keepdims=True)
        for noise in cfg["data"]["noise_levels"]:
            noise = float(noise)
            if noise > 0:
                zrng = _entry_rng(e["key"], f"noise{noise:g}")
                tr = traj + noise * per_ic_std * zrng.standard_normal(traj.shape)
            else:
                tr = traj
            shards.write_shard(a.outdir, e["key"], noise, U0, tr)
        records[e["key"]] = {"dir": shards.entry_dirname(e["key"]),
                             "stratum": side["stratum"],
                             "commutator": side["commutator"],
                             "canonical": side["canonical"],
                             **{k: side[k] for k in ("epsilon", "nu")
                                if k in side}}
        print(f"  {e['key']}  [{side['stratum']}] comm={side['commutator']:.1f}"
              f"  {time.time()-t0:.1f}s")
    tag = "all" if a.shard_index < 0 else a.shard_index
    shards.write_manifest(a.outdir, tag, records)
    print(f"wrote {len(records)} entries + manifest_shard{tag}.json")


if __name__ == "__main__":
    main()
