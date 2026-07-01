"""run_all.py -- the full staged sweep for Sid.

Stages (OFAT, not full grid):
  A  representation sweep   (fix fusion=xattn, backbone=transformer)
  B  fusion sweep           (fix best representation from A)
  C  (placeholder) backbone swap -- transformer vs VAE vs in-context

Usage:
  python scripts/run_all.py --stage A --rung S1S2 --epochs 60 --seeds 3
  python scripts/run_all.py --stage B --rung S1S2 --epochs 60 --seeds 3
Outputs: results/<stage>_results.csv  and  results/<stage>_curve.png
"""
import argparse, os, json, csv
import numpy as np, torch
from symcomp.dataset import build_benchmark
from symcomp.train import train_model
from symcomp.experiments import (E1_composition_curve, E2_channel_masking,
                                 E3_counterfactual_swap, E4_embedding_additivity)
from symcomp.operators import Operator
from symcomp.model import count_params

REPRS_A = ["grammar", "prose_tree", "lample_charton", "coeff_vector", "none"]
FUSIONS_B = ["xattn", "concat", "film"]


def normalize_commutator(curve):
    """Rescale commutator axis to [0,1] for readable plotting (order preserved)."""
    if not curve:
        return curve
    mx = max(k for k, *_ in curve) or 1.0
    return [(k / mx, m, s, n) for k, m, s, n in curve]


def run_stage(stage, rung, epochs, seeds, d_model, outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = []
    if stage == "A":
        configs = [(r, "xattn") for r in REPRS_A]
    elif stage == "B":
        configs = [("grammar", f) for f in FUSIONS_B]
    else:
        raise SystemExit("stage C (backbone swap) is a stub: wire VAE/ICON here")

    for enc, fus in configs:
        for seed in range(seeds):
            torch.manual_seed(seed); np.random.seed(seed)
            b = build_benchmark(N=128, T=16, t_max=0.5, n_ic_train=64,
                                n_ic_test=16, epsilons=(0.0, 0.15, 0.3, 0.5, 0.8),
                                rung=rung, seed=seed)
            model, vocab = train_model(b, encoder=enc, fusion=fus, epochs=epochs,
                                       batch=32, max_len=32, d_model=d_model,
                                       seed=seed, verbose=(seed == 0))
            e1 = E1_composition_curve(model, b, enc, vocab, 32)
            e2 = E2_channel_masking(model, b, enc, vocab, 32)
            e4 = E4_embedding_additivity(model, b, enc, vocab, 32)
            try:
                e3 = E3_counterfactual_swap(
                    model, b, enc, vocab, 32,
                    Operator({"advection": 1.0}),
                    Operator({"advection": 1.0, "diffusion": 0.5}))
                e3v = e3["symbol_causal_fraction_mean"]
            except Exception:
                e3v = float("nan")
            for k, m, s, n in e1["curve"]:
                rows.append({"stage": stage, "encoder": enc, "fusion": fus,
                             "seed": seed, "params": count_params(model),
                             "commutator": k, "zeroshot_relL2": m, "relL2_std": s,
                             "n": n,
                             "E3_symbol_causal": e3v,
                             "E4_ridge_R2": e4.get("ridge_R2", float("nan")),
                             "E4_add_resid": e4.get("mean_additive_residual", float("nan"))})
            # stash E2 separately
            with open(os.path.join(outdir, f"{stage}_{enc}_{fus}_s{seed}_E2.json"), "w") as f:
                json.dump(e2, f, indent=2)

    csv_path = os.path.join(outdir, f"{stage}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")

    _plot(rows, stage, outdir)
    return csv_path


def _plot(rows, stage, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable, skipping plot"); return
    encs = sorted(set(r["encoder"] for r in rows))
    plt.figure(figsize=(7, 5))
    for enc in encs:
        pts = sorted((r["commutator"], r["zeroshot_relL2"])
                     for r in rows if r["encoder"] == enc)
        # average over seeds at each commutator
        agg = {}
        for c, v in pts:
            agg.setdefault(round(c, 3), []).append(v)
        xs = sorted(agg)
        ys = [np.mean(agg[x]) for x in xs]
        mx = max(xs) or 1.0
        plt.plot([x / mx for x in xs], ys, marker="o", label=enc)
    plt.xlabel("normalized commutator  ||[A,B]|| (0 = commuting)")
    plt.ylabel("zero-shot composition rel-L2")
    plt.title(f"Stage {stage}: zero-shot compositional error vs commutator")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    p = os.path.join(outdir, f"{stage}_curve.png")
    plt.savefig(p, dpi=130); print(f"wrote {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="A", choices=["A", "B", "C"])
    ap.add_argument("--rung", default="S1S2", choices=["S1", "S1S2", "S1S2S3"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--outdir", default="results")
    a = ap.parse_args()
    run_stage(a.stage, a.rung, a.epochs, a.seeds, a.d_model, a.outdir)
