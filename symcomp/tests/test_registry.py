"""Validation for symcomp/registry.py (durable storage, D10) + the
scrambled-grammar determinism fix. Run:  python tests/test_registry.py

Checks:
  [1] run creation: run dir, config.json, manifest.json fields
  [2] schema enforcement + rows round-trip + master CSV accumulation
  [3] fetch-by-id / list_runs
  [4] home archive: small artifacts copied, big/checkpoint files skipped
  [5] cluster guard: refuses to default to scratch when SYMCOMP_WORK_DIR unset
  [6] concurrent appends from 8 processes -> no lost or torn rows
  [7] enc_grammar_scrambled identical across processes with different
      PYTHONHASHSEED (the H4 control must be reproducible)
"""
import csv
import multiprocessing
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from symcomp import registry
from symcomp.registry import (MASTER_SCHEMA, Run, get_run, list_runs,
                              rebuild_master)


def _row(i, params=1000):
    return {"stage": "A", "encoder": "grammar", "fusion": "xattn",
            "backbone": "transformer", "split_seed": 0, "init_seed": 0,
            "task": "prediction", "commutator": 0.0,
            "metric_name": "rel_l2", "metric_value": i, "params": params}


def _worker(root, wid, n):
    run = Run.create({"worker": wid}, cell_tag=f"w{wid}", root=root)
    for i in range(n):
        run.append_rows([_row(wid * 1000 + i, params=wid)])


def _mkdtemp(prefix):
    # SYMCOMP_TEST_DIR lets the flock/concurrency checks run on a target
    # filesystem (e.g. /cluster/work on Euler) instead of node-local /tmp
    return tempfile.mkdtemp(prefix=prefix,
                            dir=os.environ.get("SYMCOMP_TEST_DIR"))


def main():
    tmp = _mkdtemp("symcomp_registry_test_")

    # [1] creation
    run = Run.create({"lr": 3e-4}, cell_tag="cell000", root=tmp,
                     seeds={"split": 0, "init": 1}, param_counts={"grammar": 233176})
    assert os.path.isdir(run.dir), "run dir missing"
    assert run.config == {"lr": 3e-4}, run.config
    m = run.manifest
    for key in ("run_id", "created_utc", "git_sha", "hostname", "seeds",
                "param_counts"):
        assert key in m, f"manifest missing {key}"
    assert m["run_id"] == run.run_id
    print(f"[1] run created: {run.run_id}  OK")

    # [2] schema + rows + master
    try:
        run.append_rows([{"bogus": 1}])
        raise AssertionError("schema violation not caught")
    except ValueError:
        pass
    run.append_rows([_row(0), _row(1)])
    run2 = Run.create({}, cell_tag="cell001", root=tmp)
    run2.append_rows([_row(2)])
    assert [r["metric_value"] for r in run.rows()] == ["0", "1"]
    assert all(r["run_id"] == run.run_id for r in run.rows()), \
        "rows not stamped with run_id"
    master = os.path.join(tmp, "results", "master.csv")
    with open(master, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3, f"master has {len(rows)} rows, want 3"
    assert set(rows[0].keys()) == set(MASTER_SCHEMA) | {"run_id"}
    # torn-tail guard: simulate a writer that died mid-line
    with open(master, "ab") as f:
        f.write(b"TORN,FRAGMENT")
    run2.append_rows([_row(3)])
    with open(master, newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[-1]["metric_value"] == "3" and rows[-1]["run_id"] == run2.run_id, \
        "append after torn tail merged into the fragment"
    # same-second re-creation with the same cell tag must not collide
    dup = Run.create({}, cell_tag="cell001", root=tmp)
    assert dup.run_id != run2.run_id and os.path.isdir(dup.dir)
    print("[2] schema + run_id stamp + torn-tail guard + collision suffix  OK")

    # [3] fetch by id
    assert get_run(run.run_id, root=tmp).manifest["run_id"] == run.run_id
    assert set(list_runs(tmp)) == {run.run_id, run2.run_id, dup.run_id}
    try:
        get_run("nope", root=tmp)
        raise AssertionError("missing run not caught")
    except KeyError:
        pass
    print("[3] fetch-by-id / list_runs  OK")

    # [4] archive: small files copied; oversized + checkpoint-suffix skipped
    with open(os.path.join(run.dir, "big.csv"), "w") as f:
        f.write("x" * 4096)
    with open(os.path.join(run.dir, "model.pt"), "w") as f:
        f.write("fake checkpoint")
    archive = os.path.join(tmp, "home_archive")
    os.environ["SYMCOMP_HOME_ARCHIVE"] = archive
    try:
        copied = run.archive_to_home(max_bytes=2048)
    finally:
        del os.environ["SYMCOMP_HOME_ARCHIVE"]
    names = {os.path.basename(p) for p in copied}
    assert "manifest.json" in names and "config.json" in names, names
    assert "rows.csv" in names and "master.csv" in names, names
    assert "big.csv" not in names, "size cap not applied"
    assert "model.pt" not in names, "checkpoint not skipped"
    print(f"[4] home archive copied {sorted(names)}  OK")

    # [5] cluster guard
    saved = {k: os.environ.pop(k, None) for k in ("SYMCOMP_WORK_DIR", "SCRATCH")}
    os.environ["SCRATCH"] = "/fake/scratch"
    try:
        registry.work_dir()
        raise AssertionError("work_dir() did not refuse scratch-only cluster")
    except RuntimeError:
        pass
    finally:
        os.environ.pop("SCRATCH", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    print("[5] work_dir() refuses cluster nodes without SYMCOMP_WORK_DIR  OK")

    # [6] concurrent appends (8 procs x 25 rows on a fresh root)
    tmp2 = _mkdtemp("symcomp_registry_conc_")
    procs = [multiprocessing.Process(target=_worker, args=(tmp2, w, 25))
             for w in range(8)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
        assert p.exitcode == 0, "worker crashed"
    with open(os.path.join(tmp2, "results", "master.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    got = {r["metric_value"] for r in rows}
    want = {str(w * 1000 + i) for w in range(8) for i in range(25)}
    assert len(rows) == 200, f"lost/torn rows: {len(rows)} != 200"
    assert got == want, f"row values corrupted ({len(want - got)} missing)"
    print("[6] 8-process concurrent append: 200/200 rows intact  OK")

    # [6b] master.csv is disposable: rebuild it from the per-run rows.csv
    os.remove(os.path.join(tmp2, "results", "master.csv"))
    rebuild_master(tmp2)
    with open(os.path.join(tmp2, "results", "master.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 200 and {r["metric_value"] for r in rows} == want, \
        "rebuild_master lost rows"
    print("[6b] rebuild_master reconstructs 200/200 rows from run dirs  OK")

    # [7] scrambled-grammar determinism across processes
    snippet = ("from symcomp.operators import Operator;"
               "from symcomp.encoders import enc_grammar_scrambled;"
               "print(enc_grammar_scrambled("
               "Operator({'advection':1.0,'diffusion':0.5})))")
    outs = []
    for seed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=ROOT)
        outs.append(subprocess.check_output(
            [sys.executable, "-c", snippet], env=env, cwd=ROOT).decode())
    assert outs[0] == outs[1], (
        "enc_grammar_scrambled differs across PYTHONHASHSEED values -- "
        "the H4 control arm is not reproducible")
    print("[7] enc_grammar_scrambled stable across processes  OK")

    print("REGISTRY OK")


if __name__ == "__main__":
    main()
