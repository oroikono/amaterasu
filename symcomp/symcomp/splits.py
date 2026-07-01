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
               held_frac=0.35, max_terms=2, triples_sample=8) -> SplitManifest:
    rng = np.random.default_rng(seed)
    universe = enumerate_operators(mechanisms, coeff_choices, max_terms,
                                   triples_sample, rng)
    singles = [o for o in universe if len(o.coeffs) == 1]
    multis = [o for o in universe if len(o.coeffs) >= 2]

    # choose held-out COMPOSITES (compose direction)
    rng.shuffle(multis)
    n_hold = max(1, int(held_frac * len(multis)))
    test_compose = multis[:n_hold]
    train_multis = multis[n_hold:]

    # train always includes all singletons (so primitives are seen)
    train = list(singles) + list(train_multis)

    # DECOMPOSE direction: pick some trained composites whose a pure piece we
    # hold out -- but only if that piece is NOT otherwise in train as a needed
    # primitive. To keep primitives available generally, we instead synthesize a
    # dedicated decompose family: train a composite, hold out one of its pieces,
    # and REMOVE that piece from the singleton training set for this manifest.
    test_decompose: list[Operator] = []
    # take up to 3 trained composites, hold out one primitive each
    cand = [o for o in train_multis if len(o.coeffs) == 2][:3]
    held_primitive_names = set()
    for comp in cand:
        names = comp.names()
        drop = names[rng.integers(len(names))]
        held_primitive_names.add(drop)
        test_decompose.append(Operator({drop: coeff_choices[drop]}))
    # remove those primitives from train singletons to make it genuine
    train = [o for o in train
             if not (len(o.coeffs) == 1 and o.names()[0] in held_primitive_names)]

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
    # 2) every compose held-out's primitives ARE in train (genuine composition)
    for o in man.test_compose:
        for m in o.names():
            assert m in train_mech, (
                f"compose held-out {_canon(o)} has unseen primitive {m}")
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
