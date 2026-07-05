"""splits.py -- combinatorial held-out composition splits with leakage hygiene.

Given a set of mechanisms and coefficient choices, enumerate compositions
(singletons, pairs, sampled triples), then for each split-seed designate a
held-out set under the constraint that every held-out combo's primitives are
individually present in training (genuine compositional generalization, not
unseen-mechanism extrapolation). Emits compose AND decompose manifests and runs
automated leakage assertions (defense A2 + A9).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import combinations
import numpy as np
from .operators import Operator, PRIMITIVES


@dataclass
class SplitManifest:
    seed: int
    train: list[Operator]
    test_compose: list[Operator]
    test_decompose: list[Operator]
    meta: dict = field(default_factory=dict)


def _canon(op: Operator) -> str:
    return op.canonical_str()


def enumerate_operators(mechanisms, coeff_choices, max_terms=2,
                        triples_sample=8, rng=None):
    """Build the operator universe: singletons + pairs (+ sampled triples)."""
    rng = rng or np.random.default_rng(0)
    ops: list[Operator] = []
    # singletons
    for m in mechanisms:
        ops.append(Operator({m: coeff_choices[m]}))
    # pairs
    for a, b in combinations(mechanisms, 2):
        ops.append(Operator({a: coeff_choices[a], b: coeff_choices[b]}))
    # sampled triples
    if max_terms >= 3:
        all_tri = list(combinations(mechanisms, 3))
        rng.shuffle(all_tri)
        for tri in all_tri[:triples_sample]:
            ops.append(Operator({m: coeff_choices[m] for m in tri}))
    return ops


def make_split(mechanisms, coeff_choices, seed: int,
               held_frac=0.35, max_terms=2, triples_sample=8,
               protect=frozenset()) -> SplitManifest:
    """`protect`: mechanisms whose singletons must NEVER leave training (e.g.
    the S2 anchor's primitives) -- they may still appear in held-out
    composites, which is exactly the compose test."""
    rng = np.random.default_rng(seed)
    universe = enumerate_operators(mechanisms, coeff_choices, max_terms,
                                   triples_sample, rng)
    singles = [o for o in universe if len(o.coeffs) == 1]
    multis = [o for o in universe if len(o.coeffs) >= 2]

    # DECOMPOSE primitive FIRST (compose-premise defense): pick ONE mechanism
    # (never a protected one, e.g. the S2 anchor's) whose singleton is held
    # out; its composites all stay in train (that is the decompose task:
    # composite trained, pure piece zero-shot). Compose held-outs are then
    # drawn ONLY from composites that avoid it, so every test_compose
    # primitive's singleton remains in train -- the paper's compose claim is
    # "primitives trained INDIVIDUALLY, composite predicted zero-shot".
    dec_pool = sorted(set(mechanisms) - set(protect))
    held_primitive_names = set()
    test_decompose: list[Operator] = []
    if dec_pool:
        drop = dec_pool[int(rng.integers(len(dec_pool)))]
        held_primitive_names.add(drop)
        test_decompose.append(Operator({drop: coeff_choices[drop]}))

    # choose held-out COMPOSITES (compose direction) among eligible multis
    eligible = [o for o in multis
                if not (set(o.names()) & held_primitive_names)]
    rng.shuffle(eligible)
    n_hold = max(1, int(held_frac * len(eligible)))
    test_compose = eligible[:n_hold]
    heldout_canon = {_canon(o) for o in test_compose}
    train_multis = [o for o in multis if _canon(o) not in heldout_canon]

    # train = remaining singletons + remaining composites
    train = [o for o in singles
             if o.names()[0] not in held_primitive_names] + train_multis

    man = SplitManifest(seed, train, test_compose, test_decompose,
                        meta={"n_universe": len(universe),
                              "n_train": len(train),
                              "held_frac": held_frac,
                              "decompose_held_primitives": sorted(held_primitive_names)})
    _assert_no_leakage(man)
    return man


def _assert_no_leakage(man: SplitManifest):
    train_canon = {_canon(o) for o in man.train}
    train_mech = set()
    for o in man.train:
        train_mech.update(o.names())

    # 1) no held-out composite (canonical) appears in train
    for o in man.test_compose:
        assert _canon(o) not in train_canon, f"LEAK: {_canon(o)} in train"
    # 2) every compose held-out's primitives are trained INDIVIDUALLY (their
    #    singletons are in train) -- the compose premise, not merely "the
    #    mechanism occurs somewhere inside a trained composite"
    for o in man.test_compose:
        for m in o.names():
            single = Operator({m: o.coeffs[m]})
            assert _canon(single) in train_canon, (
                f"compose held-out {_canon(o)}: primitive {m} not trained "
                f"as a singleton (compose premise violated)")
    # 3) decompose held-out primitives are NOT in train singletons
    for o in man.test_decompose:
        assert _canon(o) not in train_canon, f"LEAK(decompose): {_canon(o)}"
    return True


def summarize(man: SplitManifest) -> str:
    lines = [f"split seed={man.seed}  {man.meta}"]
    lines.append(f"  train ({len(man.train)}): " +
                 ", ".join(_canon(o) for o in man.train[:8]) + " ...")
    lines.append(f"  test_compose ({len(man.test_compose)}): " +
                 ", ".join(_canon(o) for o in man.test_compose))
    lines.append(f"  test_decompose ({len(man.test_decompose)}): " +
                 ", ".join(_canon(o) for o in man.test_decompose))
    return "\n".join(lines)
