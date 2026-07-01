"""SymComp: symbolic-channel compositional generalization for PDE operators.

Research pipeline for the question: when a model is trained jointly on numerical
PDE data + a symbolic encoding of the operator, does it zero-shot generalize to
HELD-OUT mechanism compositions (e.g. train pure advection + pure diffusion,
test advection-diffusion)? And does success track the commutator ||[A,B]||?
"""
__version__ = "0.1.0"
