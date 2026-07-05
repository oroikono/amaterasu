"""dag_edge_list -- expression DAG serialized as a parent-child edge list.

Graph-serialization axis: instead of a tree TRAVERSAL (prefix / flattened /
production trace), the operator's expression DAG is emitted as an EDGE LIST of
(parent, child) token pairs. This destroys sequential locality (a node's
children are not adjacent to it by traversal order) while preserving the full
structure, so it probes whether the compositional win of structured encoders
depends on locality of the serialization or only on the information content.

Nodes: ROOT, T1..T3 (one per term, in alphabetical mechanism order == the
canonical Operator.names() order), and leaves MECH_<name> / COEF_<bin> where
<bin> is the shared coefficient quantization _q from symcomp.encoders. For
each term k the three edges are emitted consecutively as parent-child pairs:

    ROOT Tk   Tk MECH_<name>   Tk COEF_<bin>

Example (normative, per spec):
    advection:1.0            -> [ROOT, T1, T1, MECH_advection, T1, COEF_1]
    (6 tokens per term; a 3-term operator -> 18 tokens)

Edge-list ordering is itself a serialization choice; it is FIXED here to term
index order (T1, T2, T3 = alphabetical mechanism order), with the three edges
of each term in the fixed order (ROOT->Tk, Tk->MECH, Tk->COEF). This is
deterministic and PYTHONHASHSEED-stable (no hashing, no randomness).

Documented deviation from the earlier draft spec: the draft quoted 27 tokens
for a 3-term operator using explicit edge-TYPE tokens; the normative scheme
adopted here (per the final spec's examples) omits edge-type tokens and uses
plain (parent, child) pairs -> 18 tokens for 3 terms. Injectivity on the
Stage A universe is unaffected: each term carries its MECH_<name> leaf, and
mechanism sets uniquely identify operators there (coefficients are fixed per
mechanism), so quantization-bin collisions (e.g. 0.2 and 0.3 both -> COEF_0.25)
cannot merge two distinct operators.
"""
from __future__ import annotations

from symcomp.encoders import _q
from symcomp.operators import Operator

KEY = "dag_edge_list"


def encode(op: Operator) -> list[str]:
    """Serialize op's expression DAG as a fixed-order parent-child edge list."""
    toks: list[str] = []
    for k, name in enumerate(op.names(), start=1):
        t = f"T{k}"
        toks += ["ROOT", t,
                 t, f"MECH_{name}",
                 t, f"COEF_{_q(op.coeffs[name])}"]
    return toks
