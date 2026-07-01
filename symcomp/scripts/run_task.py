"""run_task.py -- one SLURM array task = one (rep, split_seed, init_seed) cell.

Maps a flat --task_index to the config grid, trains BOTH heads (prediction +
discovery), evaluates on the held-out compose/decompose sets across the
commutator strata, and appends rows to a per-task CSV (later merged by
scripts/aggregate.py).

IMPLEMENTATION SPEC for Euler (wire these against dataset shards on $SCRATCH):
  1. parse config (yaml), build the mechanism set + coeffs.
  2. idx -> (rep, split_seed, init_seed) via the grid below.
  3. build the split manifest (symcomp.splits.make_split) for split_seed.
  4. LOAD precomputed trajectory shards from --data_dir for every operator in
     the manifest (do NOT regenerate here; gen_data.py wrote them).
  5. resolve data_hidden_override for `rep` (symcomp.capacity) so params match.
  6. train prediction head (MSE on rollout) and discovery head (BCE on mech
     multilabel + MSE on coeffs) -- joint or sequential; log to wandb-offline/CSV.
  7. evaluate: per held-out operator, compute rel_l2 (prediction) and
     exact_match/mech_f1/coef_mae (discovery); attach analytic ||[A,B]|| and
     stratum; write one CSV row per (operator, task, metric).
  8. dump resolved config + git SHA + seed + data-manifest hash for repro.
"""
import argparse, itertools, os, subprocess, json

REPS = ["grammar", "grammar_scrambled", "prose_tree", "lample_charton",
        "coeff_vector", "none"]
SPLIT_SEEDS = [0, 1, 2, 3, 4]
INIT_SEEDS = [0, 1, 2]
GRID = list(itertools.product(REPS, SPLIT_SEEDS, INIT_SEEDS))  # 90 cells


def resolve_cell(idx):
    return GRID[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="A")
    ap.add_argument("--task_index", type=int, required=True)
    ap.add_argument("--data_dir", default=os.environ.get("SCRATCH", ".") + "/symcomp/data")
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    rep, split_seed, init_seed = resolve_cell(a.task_index)
    os.makedirs(a.outdir, exist_ok=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        sha = "nogit"
    meta = {"rep": rep, "split_seed": split_seed, "init_seed": init_seed,
            "stage": a.stage, "git": sha, "task_index": a.task_index}
    print("RESOLVED CELL:", json.dumps(meta))
    # ---- TODO(Euler): steps 3-8 above. The local pipeline in
    #      symcomp.{splits,dataset,train,experiments} + scripts.aggregate already
    #      implements the logic at toy scale; this runner wires it to shards +
    #      the capacity harness + both heads. Keep the CSV schema:
    #      stage,encoder,fusion,backbone,split_seed,init_seed,task,commutator,
    #      metric_name,metric_value,params
    with open(os.path.join(a.outdir, f"cell_{a.task_index}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("wrote cell meta; implement training/eval per spec to emit results CSV.")


if __name__ == "__main__":
    main()
