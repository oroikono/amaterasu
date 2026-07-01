"""gen_data.py -- generate and shard trajectories per operator (SLURM array).

Each array index handles a slice of the operator universe (across all split
manifests' union) and all noise levels, writing:
  <outdir>/<canonical_op>/<noise>/shard.npz   with keys: u0 (M,N), traj (M,T,N)

IMPLEMENTATION SPEC for Euler:
  1. build the full operator universe = union of operators across all split
     seeds (symcomp.splits.enumerate_operators) + the S2 variable-coeff variants
     (epsilon sweep) + S3 nonlinear (burgers, cubic).
  2. partition operators across array indices (round-robin by --shard_index).
  3. for each assigned operator:
       - S1 const-coeff linear  -> symcomp.solver.solve_constcoeff (EXACT)
       - S2 variable-coeff      -> solve_varcoeff_advdiff (+ generalize to other
                                   linear pairs as needed)
       - S3 burgers/cubic       -> solve_burgers (+ a cubic-reaction ETDRK4 to add)
       - record analytic ||[A,B]|| (operators.commutator_* ) in a sidecar json.
     write u0 + traj for n_ic_train + n_ic_test ICs at each noise level.
  4. emit a manifest mapping canonical_op -> shard path + commutator + stratum.
The local solvers are validated to machine zero (tests/test_physics.py); this
script only parallelizes and persists them.
"""
import argparse, os, json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard_index", type=int, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    print(f"gen_data shard {a.shard_index}: implement per spec -> {a.outdir}")
    # TODO(Euler): steps 1-4. Reuse symcomp.solver (validated) + symcomp.splits.


if __name__ == "__main__":
    main()
