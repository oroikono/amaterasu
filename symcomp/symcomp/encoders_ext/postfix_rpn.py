"""postfix_rpn -- postfix (reverse Polish) serialization of the prefix tree.

Serialization-order axis: the tightest minimal pair vs. `lample_charton`.
This encoder is a MECHANICAL post-order traversal of the exact expression
tree that `enc_lample_charton` builds -- same `_q` coefficient bins, same
alphabetical `Operator.names()` term order, same right-leaning '+' spine.
Nothing about the tree is re-derived here: we take the prefix token stream
from `enc_lample_charton`, reconstruct the tree using each token's fixed
arity, and emit it in post-order.

Tree shape per term (prefix "* c <bin> d<order> u"):

    *(c(<bin>), d<order>(u))   -->  post-order  [<bin>, c, u, d<order>, *]

The right-leaning spine +(t1, +(t2, t3)) puts all n-1 '+' tokens at the END.

Examples (normative, from the arm spec):
  advection:1.0
    -> [1, c, u, d1, *]
  advection:1.0+diffusion:0.5+dispersion:0.3
    -> [1, c, u, d1, *, 0.5, c, u, d2, *, 0.25, c, u, d3, *, +, +]
       (17 tokens; bin token strings are the %.4g forms, so 0.3 -> 0.25)

Pure deterministic function of the Operator: no hashing, no randomness,
no file IO. Vocab on the 5-linear-mechanism universe: 8 coeff bins +
{c, u, *, +} + {d0..d4} = 17 tokens. Max length for a 3-term operator:
3*5 + 2 = 17 <= 48.
"""
from __future__ import annotations

from symcomp.encoders import enc_lample_charton
from symcomp.operators import Operator

KEY = "postfix_rpn"

# Fixed arities of the prefix-tree tokens emitted by enc_lample_charton:
#   '+' and '*' are binary; 'c' (coefficient wrapper) and 'd<order>'
#   (derivative) are unary; coefficient-bin tokens and 'u' are leaves.
def _arity(tok: str) -> int:
    if tok in ("+", "*"):
        return 2
    if tok == "c" or (len(tok) >= 2 and tok[0] == "d" and tok[1:].isdigit()):
        return 1
    return 0  # coefficient-bin tokens and "u"


def encode(op: Operator) -> list[str]:
    """Post-order serialization of the exact enc_lample_charton tree."""
    prefix = enc_lample_charton(op)
    out: list[str] = []
    pos = 0

    def walk() -> None:
        nonlocal pos
        tok = prefix[pos]
        pos += 1
        for _ in range(_arity(tok)):
            walk()
        out.append(tok)

    walk()
    if pos != len(prefix):  # pragma: no cover - guards against upstream drift
        raise ValueError(f"prefix stream not fully consumed for {op.canonical_str()}")
    return out
