"""capacity.py -- auto-match parameter counts across representation arms (A1).

The symbolic-sequence arms (grammar, prose_tree, lample_charton, scrambled) are
naturally matched. The coeff_vector (tiny MLP symbol branch) and data_only (no
symbol branch) arms need their DATA branch widened to compensate. This module
searches width_mult per arm so every arm lands within tolerance of a target
param count, and returns the resolved width_mults for the configs.
"""
from __future__ import annotations
from .model import OperatorLearner, count_params


def build_arm(kind, n_grid, T_out, vocab_size, d_model, n_mech, max_len,
              width_mult=None, depth=2):
    return OperatorLearner(n_grid, T_out, vocab_size, d_model=d_model,
                           symbol_kind=kind, fusion="xattn", n_mech=n_mech,
                           depth=depth, max_len=max_len, width_mult=width_mult)


def resolve_hidden_overrides(reps, n_grid, T_out, vocab_sizes, d_model, n_mech,
                             max_len, tol=0.02):
    """Binary-search a per-arm data-branch hidden size so every arm matches the
    symbolic-arm param count within `tol`. Returns {rep: hidden_override|None}.

    NOTE (Euler TODO): integer hidden sizes leave a ~2-3% residual on the
    coeff_vector / data_only arms. To close it fully, either (a) add a matched
    parameter-count dummy adapter block, or (b) tune the FFN ratio on those two
    arms. The residual is reported so the harness can decide; for the headline
    H1 claim, report the exact per-arm param counts in the paper table so the
    small remaining gap is transparent rather than hidden.
    """
    ref_kind = "grammar" if "grammar" in reps else reps[0]
    ref = build_arm(ref_kind, n_grid, T_out, vocab_sizes.get(ref_kind, 8),
                    d_model, n_mech, max_len, width_mult=1)
    target = count_params(ref)

    def params_for(kind, h):
        vs = vocab_sizes.get(kind, 8) if kind not in ("coeff_vector", "none") else 1
        m = OperatorLearner(n_grid, T_out, vs, d_model=d_model, symbol_kind=kind,
                            fusion="xattn", n_mech=n_mech, max_len=max_len,
                            data_hidden_override=h)
        return count_params(m)

    out = {}
    for rep in reps:
        native = params_for(rep, None)
        if abs(native - target) / target <= tol:
            out[rep] = None
            continue
        lo, hi = 16, 1024
        for _ in range(40):
            mid = (lo + hi) / 2
            if params_for(rep, mid) < target:
                lo = mid
            else:
                hi = mid
        h = int(round(lo))
        err = abs(params_for(rep, h) - target) / target
        out[rep] = h
        if err > tol:
            out[rep] = (h, f"residual {err:.1%}")
    return out, target


def resolve_width_mults(reps, n_grid, T_out, vocab_sizes, d_model, n_mech,
                        max_len, tol=0.02, search=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
                                                  1.6, 1.8, 2.0, 2.2, 2.5)):
    """Return {rep: width_mult} so all arms match the median symbolic-arm size.

    width_mult is applied as an integer-ish multiplier inside DataEncoder; here
    we approximate by scanning a small set and picking the closest match. For
    fractional control on Euler, DataEncoder.width_mult can be generalized to a
    hidden-size override -- TODO marker.
    """
    # reference = a symbolic arm at width_mult=1
    ref_kind = "grammar" if "grammar" in reps else reps[0]
    ref = build_arm(ref_kind, n_grid, T_out, vocab_sizes.get(ref_kind, 8),
                    d_model, n_mech, max_len, width_mult=1)
    target = count_params(ref)

    out = {}
    for rep in reps:
        vs = vocab_sizes.get(rep, 8) if rep not in ("coeff_vector", "none") else 1
        best_wm, best_err = 1, 1e9
        for wm in search:
            # width_mult must be int in current DataEncoder; round and test
            wmi = max(1, int(round(wm)))
            m = build_arm(rep, n_grid, T_out, vs, d_model, n_mech, max_len,
                          width_mult=wmi)
            err = abs(count_params(m) - target) / target
            if err < best_err:
                best_err, best_wm = err, wmi
        out[rep] = best_wm
        if best_err > tol:
            # not matchable by integer width alone -> flag for hidden-size override
            out[rep] = (best_wm, f"residual {best_err:.1%} > tol; "
                                 f"use hidden-size override on Euler")
    return out, target
