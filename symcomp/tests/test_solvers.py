"""Validation for the S2/S3 time-stepping solvers + batched variants.
Run:  python tests/test_solvers.py

The S1 exact spectral path is validated in test_physics.py. This file closes
the remaining solver test gap (handoff TODO): the solvers that generate the
S2 (variable-coefficient RK4) and S3 (Burgers ETDRK4) strata are checked
against exact limits and self-convergence, and the batched variants used by
gen_data.py are checked against the validated single-IC implementations.

  [1] batch == single-IC loop, all three solvers
  [2] varcoeff RK4 at epsilon=0 == EXACT constant-coeff spectral solve
  [3] varcoeff RK4 self-convergence (n_sub vs 2*n_sub)
  [4] Burgers ETDRK4, tiny amplitude == EXACT heat-equation limit
  [5] Burgers ETDRK4 self-convergence
  [6] stable_n_sub keeps production-scale S2 solves bounded
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from symcomp.operators import Operator, make_varcoeff_profile
from symcomp.solver import (random_initial_condition, solve_constcoeff,
                            solve_constcoeff_batch, solve_varcoeff_advdiff,
                            solve_varcoeff_advdiff_batch, solve_burgers,
                            solve_burgers_batch, stable_n_sub)

N, Ldom, T = 128, 2 * np.pi, 9
t_eval = np.linspace(0.0, 0.5, T)
rng = np.random.default_rng(7)


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


def main():
    U0 = np.stack([random_initial_condition(N, Ldom, rng=rng) for _ in range(4)])
    op = Operator({"advection": 1.0, "diffusion": 0.5})
    a_field = make_varcoeff_profile(N, Ldom, 1.0, 0.5, k=1)
    nu_field = make_varcoeff_profile(N, Ldom, 0.5, 0.5, k=2)

    # [1] batched solvers reproduce the validated single-IC solvers exactly
    b1 = solve_constcoeff_batch(op, U0, t_eval, N, Ldom)
    s1 = np.stack([solve_constcoeff(op, u, t_eval, N, Ldom) for u in U0])
    assert rel(b1, s1) < 1e-13, f"constcoeff batch != single ({rel(b1, s1):.2e})"
    b2 = solve_varcoeff_advdiff_batch(a_field, nu_field, U0, t_eval, N, Ldom)
    s2 = np.stack([solve_varcoeff_advdiff(a_field, nu_field, u, t_eval, N, Ldom)
                   for u in U0])
    assert rel(b2, s2) < 1e-12, f"varcoeff batch != single ({rel(b2, s2):.2e})"
    b3 = solve_burgers_batch(0.1, U0, t_eval, N, Ldom)
    s3 = np.stack([solve_burgers(0.1, u, t_eval, N, Ldom) for u in U0])
    assert rel(b3, s3) < 1e-12, f"burgers batch != single ({rel(b3, s3):.2e})"
    print(f"[1] batch == single: const {rel(b1, s1):.1e}, "
          f"varcoeff {rel(b2, s2):.1e}, burgers {rel(b3, s3):.1e}  OK")

    # [2] epsilon=0 (constant coefficients) must match the EXACT spectral solve
    a0 = make_varcoeff_profile(N, Ldom, 1.0, 0.0, k=1)
    nu0 = make_varcoeff_profile(N, Ldom, 0.5, 0.0, k=2)
    u0 = U0[0]
    rk = solve_varcoeff_advdiff(a0, nu0, u0, t_eval, N, Ldom, n_sub=4000)
    ex = solve_constcoeff(op, u0, t_eval, N, Ldom)
    e2 = rel(rk, ex)
    assert e2 < 1e-6, f"varcoeff eps=0 vs exact: {e2:.2e}"
    print(f"[2] varcoeff eps=0 vs exact spectral: rel err {e2:.2e}  OK")

    # [3] RK4 self-convergence at eps=0.5
    c1 = solve_varcoeff_advdiff(a_field, nu_field, u0, t_eval, N, Ldom, n_sub=2000)
    c2 = solve_varcoeff_advdiff(a_field, nu_field, u0, t_eval, N, Ldom, n_sub=4000)
    e3 = rel(c1, c2)
    assert e3 < 1e-7, f"varcoeff self-convergence: {e3:.2e}"
    print(f"[3] varcoeff RK4 n_sub 2000 vs 4000: rel diff {e3:.2e}  OK")

    # [4] tiny-amplitude Burgers -> heat equation exact limit
    amp = 1e-6
    bt = solve_burgers(0.1, amp * u0, t_eval, N, Ldom)
    heat = solve_constcoeff(Operator({"diffusion": 0.1}), amp * u0, t_eval, N, Ldom)
    e4 = rel(bt, heat)
    assert e4 < 1e-4, f"tiny-amplitude Burgers vs heat: {e4:.2e}"
    print(f"[4] Burgers(1e-6 amp) vs exact heat: rel err {e4:.2e}  OK")

    # [5] ETDRK4 self-convergence at real amplitude, small nu (hard regime)
    g1 = solve_burgers(0.02, u0, t_eval, N, Ldom, n_sub=4000)
    g2 = solve_burgers(0.02, u0, t_eval, N, Ldom, n_sub=8000)
    e5 = rel(g1, g2)
    assert e5 < 1e-6, f"burgers self-convergence: {e5:.2e}"
    print(f"[5] Burgers ETDRK4 n_sub 4000 vs 8000: rel diff {e5:.2e}  OK")

    # [6] production-scale stability: N=256, strongest epsilon in the config,
    # WELL-POSED profile convention (nu amplitude relative -> nu(x) > 0; the
    # absolute convention gives nu_min=-0.3 -> ill-posed blow-up ~1e204)
    Np = 256
    ap = make_varcoeff_profile(Np, Ldom, 1.0, 0.8, k=1)
    nup = make_varcoeff_profile(Np, Ldom, 0.5, 0.8 * 0.5, k=2)
    assert nup.min() > 0
    ns = stable_n_sub(np.abs(ap).max(), np.abs(nup).max(), Np, Ldom, 0.5)
    u0p = random_initial_condition(Np, Ldom, rng=rng)
    tp = solve_varcoeff_advdiff(ap, nup, u0p, t_eval, Np, Ldom, n_sub=ns)
    assert np.isfinite(tp).all() and np.abs(tp).max() < 50, \
        f"production-scale S2 solve unstable (max {np.abs(tp).max():.1e})"
    # and the default n_sub=2000 at N=256 WOULD be unstable -- document why
    # stable_n_sub exists (this is the bug the adaptive n_sub prevents):
    print(f"[6] N=256 eps=0.8: stable_n_sub={ns}, max|u|={np.abs(tp).max():.2f}  OK")

    print("SOLVERS OK")


if __name__ == "__main__":
    main()
