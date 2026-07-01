"""encoders.py -- the four symbolic representations (the controlled variable).

The whole paper hinges on comparing these head-to-head on the SAME held-out
composition task. Each turns an Operator into an integer token sequence (or a
fixed vector for the coeff-vector control). A shared vocabulary is built across
the corpus so the four are comparable at matched embedding capacity.

  (a) lample_charton : prefix / Polish notation over an expression tree
                       (the symbolic-math-transformer standard).
  (b) prose_tree     : PROSE-style flattened tree with explicit derivative
                       symbols and term separators.
  (c) grammar        : sequence of GRAMMAR PRODUCTION RULES that derive the
                       operator (composition = applying one '+term' production).
  (d) coeff_vector   : non-symbolic control; fixed-length real vector of
                       per-mechanism coefficients (Unisolver / equation-aware
                       style). Isolates "structure" vs "just the numbers".
"""
from __future__ import annotations
from typing import Iterable
import numpy as np
from .operators import Operator, PRIMITIVES

MECHANISMS = list(PRIMITIVES.keys())  # canonical order

# coefficient quantization for symbolic tokenization (coeff -> token).
# We bucket coefficients so the symbolic vocab is finite; the coeff_vector
# encoder keeps the raw float (its advantage to control for).
_COEFF_BINS = np.round(np.linspace(0.25, 2.0, 8), 4)


def _q(c: float) -> str:
    return f"{_COEFF_BINS[int(np.argmin(np.abs(_COEFF_BINS - c)))]:.4g}"


# ---- (a) Lample-Charton prefix ---------------------------------------------
def enc_lample_charton(op: Operator) -> list[str]:
    """Prefix expression tree of  sum_k c_k * D^{order_k}.
    e.g. advection(1.0)+diffusion(0.5) ->
         + * c 1 d1 u * c 0.5 d2 u
    """
    toks: list[str] = []
    names = op.names()
    # build a right-leaning '+' spine
    for i, n in enumerate(names):
        if i < len(names) - 1:
            toks.append("+")
        order = PRIMITIVES[n]["order"]
        toks += ["*", "c", _q(op.coeffs[n]), f"d{order}", "u"]
    return toks


# ---- (b) PROSE-style tree --------------------------------------------------
def enc_prose_tree(op: Operator) -> list[str]:
    """Flattened tree with explicit derivative symbols and TERM separators."""
    toks: list[str] = ["<bos>"]
    for n in op.names():
        order = PRIMITIVES[n]["order"]
        toks += ["<term>", "coef", _q(op.coeffs[n]), f"u_x{order}"]
    toks.append("<eos>")
    return toks


# ---- (c) Grammar productions ----------------------------------------------
# Grammar:  OP -> TERM | OP '+' TERM ;  TERM -> COEF MECH
# We emit the sequence of production-rule applications. Composition is, by
# construction, a single 'OP -> OP + TERM' production -- this is the inductive
# bias under test.
def enc_grammar(op: Operator) -> list[str]:
    toks: list[str] = []
    names = op.names()
    for i, n in enumerate(names):
        if i == 0:
            toks.append("R_OP->TERM")
        else:
            toks.append("R_OP->OP+TERM")
        toks.append("R_TERM->COEF_MECH")
        toks.append(f"MECH_{n}")
        toks.append(f"COEF_{_q(op.coeffs[n])}")
    return toks


# ---- (d) Coefficient-vector control ---------------------------------------
def enc_coeff_vector(op: Operator) -> np.ndarray:
    """Fixed-length real vector over the canonical mechanism order."""
    v = np.zeros(len(MECHANISMS), dtype=np.float32)
    for n, c in op.coeffs.items():
        v[MECHANISMS.index(n)] = c
    return v


ENCODERS = {
    "lample_charton": enc_lample_charton,
    "prose_tree": enc_prose_tree,
    "grammar": enc_grammar,
}


# ---- (e) SCRAMBLED grammar: the A3 killer control --------------------------
# Same machinery, same token budget as `grammar`, but the production rules are
# PERMUTED so they no longer mirror additive composition. If the real grammar
# beats this, the win is the COMPOSITIONAL STRUCTURE, not "being a grammar."
# The scramble is a fixed bijection on the production vocabulary, applied
# consistently so the representation is still learnable -- just not aligned with
# the operator-sum semantics. A held-out composition therefore does NOT
# correspond to a single clean production step in this arm.
_SCRAMBLE_SEED = 1234


def _scramble_map(tokens: list[str]) -> dict[str, str]:
    rng = np.random.default_rng(_SCRAMBLE_SEED)
    perm = list(tokens)
    rng.shuffle(perm)
    return dict(zip(tokens, perm))


def enc_grammar_scrambled(op: Operator) -> list[str]:
    base = enc_grammar(op)
    # build a stable vocab over all possible grammar tokens, then permute roles
    vocab_tokens = sorted(set(
        ["R_OP->TERM", "R_OP->OP+TERM", "R_TERM->COEF_MECH"]
        + [f"MECH_{n}" for n in MECHANISMS]
        + [f"COEF_{_q(c)}" for c in _COEFF_BINS]
    ))
    sm = _scramble_map(vocab_tokens)
    # ALSO shuffle the ORDER within each term so the additive spine is destroyed
    out = [sm.get(t, t) for t in base]
    rng = np.random.default_rng(hash(op.canonical_str()) % (2**32))
    rng.shuffle(out)
    return out


ENCODERS["grammar_scrambled"] = enc_grammar_scrambled


def build_vocab(ops: Iterable[Operator]) -> dict[str, dict[str, int]]:
    """Build a per-encoder token->id vocab over the corpus (symbolic encoders)."""
    vocabs: dict[str, dict[str, int]] = {}
    for name, fn in ENCODERS.items():
        toks = {"<pad>"}
        for op in ops:
            toks.update(fn(op))
        vocabs[name] = {t: i for i, t in enumerate(sorted(toks))}
    return vocabs


def encode_ids(op: Operator, encoder: str, vocab: dict[str, int],
               max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, mask) padded to max_len for a symbolic encoder."""
    toks = ENCODERS[encoder](op)[:max_len]
    ids = np.full(max_len, vocab["<pad>"], dtype=np.int64)
    mask = np.zeros(max_len, dtype=np.float32)
    for i, t in enumerate(toks):
        ids[i] = vocab[t]; mask[i] = 1.0
    return ids, mask
