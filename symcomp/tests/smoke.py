"""smoke.py -- tiny end-to-end run to prove the pipeline executes."""
import numpy as np, torch
from symcomp.dataset import build_benchmark
from symcomp.train import train_model, evaluate
from symcomp.experiments import (E1_composition_curve, E2_channel_masking,
                                 E3_counterfactual_swap, E4_embedding_additivity)
from symcomp.operators import Operator
from symcomp.model import count_params

torch.manual_seed(0); np.random.seed(0)
print("building tiny benchmark (rung S1S2)...")
b = build_benchmark(N=64, T=8, t_max=0.4, n_ic_train=16, n_ic_test=6,
                    epsilons=(0.0, 0.3, 0.6), rung="S1S2", seed=0)
print(f"  samples={len(b.samples)} train={len(b.train_idx)} "
      f"compose_test={len(b.test_compose_idx)} decompose_test={len(b.test_decompose_idx)}")

for enc, fus in [("grammar", "xattn"), ("lample_charton", "xattn")]:
    print(f"\n=== {enc} / {fus} ===")
    model, vocab = train_model(b, encoder=enc, fusion=fus, epochs=12,
                               batch=16, max_len=24, d_model=64, verbose=True)
    print(f"  params={count_params(model):,}")
    e1 = E1_composition_curve(model, b, enc, vocab, 24)
    print("  E1 curve (commutator, mean relL2, std, n):")
    for k, m, s, n in e1["curve"]:
        print(f"     ||[A,B]||~{k:>10.1f}  relL2={m:.3f}±{s:.3f}  n={n}")
    e2 = E2_channel_masking(model, b, enc, vocab, 24)
    print("  E2 symbol leverage by stratum:")
    for st, d in e2.items():
        print(f"     {st}: with={d['err_with_symbol']:.3f} masked={d['err_masked']:.3f} "
              f"leverage={d['symbol_leverage']:+.3f}")
    e3 = E3_counterfactual_swap(model, b, enc, vocab, 24,
                                Operator({"advection": 1.0}),
                                Operator({"advection": 1.0, "diffusion": 0.5}))
    print(f"  E3 symbol-causal fraction = {e3['symbol_causal_fraction_mean']:.3f}"
          f"±{e3['symbol_causal_fraction_std']:.3f}")
    e4 = E4_embedding_additivity(model, b, enc, vocab, 24)
    print(f"  E4 additivity: {e4}")
print("\nSMOKE OK")
