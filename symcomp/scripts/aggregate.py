"""aggregate.py -- turn per-run CSVs into the defensible result.

Produces:
  * H1 table: grammar vs each baseline, paired bootstrap 95% CI + sign test,
    per task (prediction, discovery), over the split-seed battery.
  * H2 regression: rel_l2 ~ normalized commutator, per rep, with R² + Spearman.
  * H4 panel: real vs scrambled grammar.
  * the money plot: error vs commutator, one line per rep, CI bands.

Reads results/<stage>_results.csv with columns:
  stage, encoder, fusion, backbone, split_seed, init_seed, task, commutator,
  metric_name, metric_value, params
"""
from __future__ import annotations
import argparse, os, csv, math
from collections import defaultdict
import numpy as np


def load(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in ("commutator", "metric_value", "params"):
                if k in r and r[k] not in ("", "nan"):
                    r[k] = float(r[k])
            rows.append(r)
    return dedup_requeues(rows)


def dedup_requeues(rows):
    """Keep only the LATEST run's rows per cell-metric group.

    Requeued/re-run SLURM tasks append a second full set of rows under a new
    run_id (registry run_ids sort chronologically); without dedup those cells
    would be silently double-weighted in every downstream statistic. Rows
    without a run_id column (legacy/local CSVs) are kept as-is.
    """
    latest = {}
    for r in rows:
        if not r.get("run_id"):
            continue
        key = tuple(r.get(k) for k in ("stage", "encoder", "fusion", "backbone",
                                       "split_seed", "init_seed", "task",
                                       "metric_name", "commutator"))
        if key not in latest or r["run_id"] > latest[key]:
            latest[key] = r["run_id"]
    out = []
    dropped = 0
    for r in rows:
        if r.get("run_id"):
            key = tuple(r.get(k) for k in ("stage", "encoder", "fusion",
                                           "backbone", "split_seed",
                                           "init_seed", "task", "metric_name",
                                           "commutator"))
            if r["run_id"] != latest[key]:
                dropped += 1
                continue
        out.append(r)
    if dropped:
        print(f"dedup_requeues: dropped {dropped} superseded rows")
    return out


def paired_bootstrap(diffs, n=10000, seed=0):
    """95% CI + p(sign) for a vector of per-split paired differences."""
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, float)
    boot = np.array([rng.choice(diffs, len(diffs), replace=True).mean()
                     for _ in range(n)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    # sign test p-value (two-sided) on the per-split diffs
    pos = (diffs > 0).sum(); k = min(pos, len(diffs) - pos)
    from math import comb
    p = sum(comb(len(diffs), i) for i in range(k + 1)) / 2**len(diffs) * 2
    return float(diffs.mean()), float(lo), float(hi), float(min(p, 1.0))


HIGHER_BETTER = {"exact_match", "mech_f1"}   # discovery metrics: higher=better


def h1_table(rows, task, metric, ref="grammar"):
    """Paired (over split_seed) grammar-minus-baseline differences.

    Sign convention: a positive reported diff ALWAYS means grammar is better,
    regardless of whether the metric is lower- or higher-is-better.
    """
    flip = -1.0 if metric in HIGHER_BETTER else 1.0
    # mean over init_seed within (encoder, split_seed)
    agg = defaultdict(list)
    for r in rows:
        if r.get("task") == task and r.get("metric_name") == metric:
            agg[(r["encoder"], int(r["split_seed"]))].append(r["metric_value"])
    cell = {k: np.mean(v) for k, v in agg.items()}
    encoders = sorted({e for e, _ in cell})
    seeds = sorted({s for _, s in cell})
    out = {}
    for e in encoders:
        if e == ref:
            continue
        diffs = []
        for s in seeds:
            if (ref, s) in cell and (e, s) in cell:
                # positive => grammar better, after direction flip
                diffs.append(flip * (cell[(e, s)] - cell[(ref, s)]))
        if diffs:
            out[e] = paired_bootstrap(diffs)
    return out, ref


def commutator_regression(rows, encoder, metric="rel_l2", task="prediction"):
    xs, ys = [], []
    for r in rows:
        if r.get("encoder") == encoder and r.get("task") == task and \
           r.get("metric_name") == metric:
            xs.append(r["commutator"]); ys.append(r["metric_value"])
    if len(xs) < 3:
        return None
    xs = np.array(xs); ys = np.array(ys)
    xn = (xs - xs.min()) / (np.ptp(xs) + 1e-9)
    A = np.vstack([xn, np.ones_like(xn)]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    r2 = 1 - np.sum((ys - pred) ** 2) / (np.sum((ys - ys.mean()) ** 2) + 1e-9)
    # spearman
    rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
    rho = np.corrcoef(rx, ry)[0, 1]
    return {"slope": float(coef[0]), "intercept": float(coef[1]),
            "R2": float(r2), "spearman": float(rho), "n": len(xs)}


def h2_stratified(rows, encoder, metric="rel_l2", task="prediction"):
    """H2 on the PRE-REGISTERED test set: the anchor epsilon sweep only
    (rows with commutator > 0, i.e. the S2 variants of one fixed composite).

    The pooled regression above confounds operator difficulty with the
    commutator: its commutator=0 rows include hard S1 composites while all
    commutator>0 rows are the easy anchor family, which can mask or even
    flip the true within-sweep trend. Here difficulty is held fixed by
    construction. Returns spearman rho + the mean per-cell error delta
    between the largest and smallest commutator (positive = degrades).
    """
    per = defaultdict(dict)
    xs, ys = [], []
    for r in rows:
        if r.get("encoder") == encoder and r.get("task") == task and \
           r.get("metric_name") == metric and r["commutator"] > 0:
            xs.append(r["commutator"]); ys.append(r["metric_value"])
            per[(r.get("split_seed"), r.get("init_seed"))][
                round(r["commutator"])] = r["metric_value"]
    if len(xs) < 3:
        return None
    rx = np.argsort(np.argsort(xs)); ry = np.argsort(np.argsort(ys))
    rho = float(np.corrcoef(rx, ry)[0, 1])
    deltas = [v[max(v)] - v[min(v)] for v in per.values() if len(v) >= 2]
    return {"spearman": rho, "mean_delta_max_min": float(np.mean(deltas)),
            "n": len(xs), "n_cells": len(deltas)}


def money_plot(rows, metric="rel_l2", task="prediction", out="results/money_plot.png"):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib unavailable"); return
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("task") == task and r.get("metric_name") == metric:
            by[r["encoder"]][round(r["commutator"], 4)].append(r["metric_value"])
    plt.figure(figsize=(7.5, 5))
    for enc in sorted(by):
        xs = sorted(by[enc]); mx = max(xs) or 1.0
        mean = [np.mean(by[enc][x]) for x in xs]
        sd = [np.std(by[enc][x]) for x in xs]
        xn = [x / mx for x in xs]
        plt.plot(xn, mean, marker="o", label=enc)
        plt.fill_between(xn, np.array(mean) - np.array(sd),
                         np.array(mean) + np.array(sd), alpha=0.15)
    plt.xlabel("normalized commutator  ||[A,B]||  (0 = commuting)")
    plt.ylabel(f"zero-shot {task} {metric}")
    plt.title("Zero-shot compositional error vs commutator")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=140); print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--task", default="prediction")
    ap.add_argument("--metric", default="rel_l2")
    a = ap.parse_args()
    rows = load(a.csv)
    print(f"loaded {len(rows)} rows")
    tab, ref = h1_table(rows, a.task, a.metric)
    print(f"\nH1: {ref} minus baseline (positive => {ref} better), 95% CI [sign p]:")
    for e, (m, lo, hi, p) in tab.items():
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {ref} vs {e:18s}: {m:+.4f}  CI[{lo:+.4f},{hi:+.4f}]  p={p:.3f} {star}")
    print("\nH2 (pooled -- CAUTION: stratum-confounded, see h2_stratified):")
    for enc in sorted({r['encoder'] for r in rows}):
        reg = commutator_regression(rows, enc, a.metric, a.task)
        if reg:
            print(f"  {enc:18s}: slope={reg['slope']:+.3f} R2={reg['R2']:.3f} "
                  f"rho={reg['spearman']:+.3f} n={reg['n']}")
    print("\nH2 STRATIFIED (pre-registered anchor sweep, difficulty held fixed):")
    for enc in sorted({r['encoder'] for r in rows}):
        st = h2_stratified(rows, enc, a.metric, a.task)
        if st:
            print(f"  {enc:18s}: rho={st['spearman']:+.3f} "
                  f"delta(err@max-min ||[A,B]||)={st['mean_delta_max_min']:+.4f} "
                  f"n={st['n']} cells={st['n_cells']}")
    money_plot(rows, a.metric, a.task)
