"""dataset.py -- the commutator-stratified compositional benchmark.

Builds operators across three strata and constructs held-out COMPOSITION splits
in both directions (compose: pure -> sum ; decompose: sum -> pure). Every
held-out composition carries its commutator magnitude as metadata so zero-shot
error can be regressed against ||[A,B]||.

Strata:
  S1 commuting        : constant-coeff linear, ||[A,B]|| = 0  (sanity floor)
  S2 weak-noncommute  : variable-coeff linear, ||[A,B]|| swept by epsilon
  S3 strong/singular  : nonlinear advection (Burgers), nu -> 0 asymmetry
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .operators import (Operator, commutator_constcoeff,
                        variable_coeff_commutator_norm, make_varcoeff_profile)
from .solver import (random_initial_condition, solve_constcoeff,
                     solve_varcoeff_advdiff, solve_burgers, grid)


@dataclass
class Sample:
    op: Operator
    stratum: str
    commutator: float
    traj: np.ndarray          # (T, N) numerical solution
    u0: np.ndarray            # (N,)
    role: str                 # 'primitive' | 'composite'
    meta: dict = field(default_factory=dict)


@dataclass
class Benchmark:
    samples: list[Sample]
    t_eval: np.ndarray
    N: int
    Ldom: float
    # split: lists of indices into samples
    train_idx: list[int] = field(default_factory=list)
    test_compose_idx: list[int] = field(default_factory=list)
    test_decompose_idx: list[int] = field(default_factory=list)


def _const_op(coeffs: dict[str, float]) -> Operator:
    return Operator(coeffs)


def build_benchmark(
    N: int = 128, Ldom: float = 2 * np.pi, T: int = 16, t_max: float = 0.5,
    n_ic_train: int = 64, n_ic_test: int = 16, noise: float = 0.0,
    epsilons=(0.0, 0.15, 0.3, 0.5, 0.8), seed: int = 0,
    rung: str = "S1S2",   # 'S1', 'S1S2', or 'S1S2S3'
) -> Benchmark:
    """Construct the staged benchmark. `rung` controls how far up the ladder."""
    rng = np.random.default_rng(seed)
    t_eval = np.linspace(0.0, t_max, T)
    samples: list[Sample] = []

    def add(op, stratum, comm, role, n_ic, solver_fn, meta=None):
        idxs = []
        for j in range(n_ic):
            u0 = random_initial_condition(N, Ldom, rng=rng)
            traj = solver_fn(u0)
            if noise > 0:
                traj = traj + noise * rng.standard_normal(traj.shape) * np.std(traj)
            samples.append(Sample(op, stratum, comm, traj, u0, role, meta or {}))
            idxs.append(len(samples) - 1)
        return idxs

    # ---------- S1: commuting, constant-coefficient ----------
    # primitives
    a_coef, nu_coef = 1.0, 0.5
    op_adv = _const_op({"advection": a_coef})
    op_dif = _const_op({"diffusion": nu_coef})
    op_advdif = _const_op({"advection": a_coef, "diffusion": nu_coef})

    train_idx, test_compose_idx, test_decompose_idx = [], [], []

    sf = lambda op: (lambda u0: solve_constcoeff(op, u0, t_eval, N, Ldom))
    train_idx += add(op_adv, "S1", 0.0, "primitive", n_ic_train, sf(op_adv))
    train_idx += add(op_dif, "S1", 0.0, "primitive", n_ic_train, sf(op_dif))
    # COMPOSE test: held-out sum, model never saw advection+diffusion together
    test_compose_idx += add(op_advdif, "S1", 0.0, "composite", n_ic_test, sf(op_advdif))

    if "S1S2" in rung or rung == "S1S2S3" or rung == "S1S2":
        # add a 3rd primitive so pairwise/3-way splits exist
        disp_coef = 0.3
        op_disp = _const_op({"dispersion": disp_coef})
        op_adv_disp = _const_op({"advection": a_coef, "dispersion": disp_coef})
        train_idx += add(op_disp, "S1", 0.0, "primitive", n_ic_train, sf(op_disp))
        test_compose_idx += add(op_adv_disp, "S1", 0.0, "composite", n_ic_test, sf(op_adv_disp))

        # DECOMPOSE direction: train on a composite, hold out its pure pieces.
        op_dif_disp = _const_op({"diffusion": nu_coef, "dispersion": disp_coef})
        train_idx += add(op_dif_disp, "S1", 0.0, "composite", n_ic_train, sf(op_dif_disp))
        # the pure pieces of diffusion+dispersion are already trained (dif, disp)
        # so to make a genuine decompose hold-out we add a fresh composite whose
        # pieces are NOT separately trained:
        rc_coef = 0.7
        op_reac = _const_op({"reaction": rc_coef})
        op_hyp = _const_op({"hyperdiffusion": 0.2})
        op_reac_hyp = _const_op({"reaction": rc_coef, "hyperdiffusion": 0.2})
        train_idx += add(op_reac_hyp, "S1", 0.0, "composite", n_ic_train, sf(op_reac_hyp))
        test_decompose_idx += add(op_reac, "S1", 0.0, "primitive", n_ic_test, sf(op_reac),
                                  meta={"decompose_of": "reaction+hyperdiffusion"})
        test_decompose_idx += add(op_hyp, "S1", 0.0, "primitive", n_ic_test, sf(op_hyp),
                                  meta={"decompose_of": "reaction+hyperdiffusion"})

        # ---------- S2: variable-coefficient, commutator swept by epsilon ----
        for eps in epsilons:
            if eps == 0.0:
                continue
            # WELL-POSEDNESS: the nu profile uses RELATIVE amplitude
            # (nu(x) = nu_coef*(1 + eps*cos), min = nu_coef*(1-eps) > 0 for
            # eps < 1). An absolute amplitude eps > nu_coef makes nu(x)
            # negative -> locally backward heat -> ill-posed blow-up (verified:
            # max|u| ~ 1e204 at eps=0.8), which would poison the S2 stratum.
            a_field = make_varcoeff_profile(N, Ldom, a_coef, eps, k=1)
            nu_field = make_varcoeff_profile(N, Ldom, nu_coef, eps * nu_coef, k=2)
            comm = variable_coeff_commutator_norm(a_field, nu_field, N, Ldom)
            # symbolic encoding still names {advection, diffusion}; the varcoeff
            # detail is the part the symbol channel CANNOT carry -- that's the
            # point. We tag meta with epsilon.
            vc_solver = lambda u0, af=a_field, nf=nu_field: solve_varcoeff_advdiff(
                af, nf, u0, t_eval, N, Ldom)
            test_compose_idx += add(op_advdif, "S2", comm, "composite", n_ic_test,
                                    vc_solver, meta={"epsilon": eps,
                                                     "variable_coeff": True})

    if rung == "S1S2S3":
        # ---------- S3: Burgers (strong/singular), nu sweep ----------
        for nu in (0.1, 0.05, 0.02):
            burg = lambda u0, nn=nu: solve_burgers(nn, u0, t_eval, N, Ldom)
            op_burg = _const_op({"advection": 1.0, "diffusion": nu})  # symbol proxy
            # commutator proxy for nonlinear advection grows as nu shrinks
            comm = variable_coeff_commutator_norm(
                make_varcoeff_profile(N, Ldom, 1.0, 0.5, 1),
                make_varcoeff_profile(N, Ldom, nu, 0.0, 2), N, Ldom)
            test_compose_idx += add(op_burg, "S3", comm, "composite", n_ic_test,
                                    burg, meta={"nu": nu, "nonlinear": True})

    return Benchmark(samples, t_eval, N, Ldom,
                     train_idx, test_compose_idx, test_decompose_idx)
