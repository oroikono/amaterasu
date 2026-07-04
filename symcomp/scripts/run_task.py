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
import argparse, itertools, json, os

from symcomp import registry

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
    ap.add_argument("--workdir", default=None,
                    help="durable output root; defaults to $SYMCOMP_WORK_DIR "
                         "via symcomp.registry (refuses scratch on clusters)")
    a = ap.parse_args()

    rep, split_seed, init_seed = resolve_cell(a.task_index)
    cell = {"rep": rep, "split_seed": split_seed, "init_seed": init_seed,
            "stage": a.stage, "task_index": a.task_index}
    print("RESOLVED CELL:", json.dumps(cell))

    # Durable run registration (D10): run dir + manifest live under the work
    # dir, never scratch. Raw shards stay on --data_dir (regenerable).
    with open(a.config) as f:
        config_text = f.read()
    run = registry.Run.create(
        {"config_path": a.config, "config_text": config_text, "cell": cell},
        cell_tag=f"cell{a.task_index:03d}", root=a.workdir, **cell)
    print(f"registered run {run.run_id} -> {run.dir}")

    # ---- TODO(Euler): steps 3-8 of the spec above. The local pipeline in
    #      symcomp.{splits,dataset,train,experiments} + scripts.aggregate already
    #      implements the logic at toy scale; this runner wires it to shards +
    #      the capacity harness + both heads. Emit result rows through
    #      run.append_rows(rows) -- the registry enforces the fixed master
    #      schema (registry.MASTER_SCHEMA) and file-locks the master CSV.
    #      When loading shards (step 4), record their provenance in the
    #      manifest via registry.file_hashes(shard_paths) (spec item:
    #      data-file hashes). Finish with run.archive_to_home() so small
    #      artifacts survive on home storage even if work storage has an
    #      incident.
    run.archive_to_home()
    print("wrote run manifest; implement training/eval per spec to emit rows.")


if __name__ == "__main__":
    main()
