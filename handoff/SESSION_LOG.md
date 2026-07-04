# SESSION_LOG

Append-only, timestamped progress log. Purpose: if a session dies (usage cap,
crash) or another person/agent picks the project up, this file says where we
started, what happened since, and where things stand right now. Newest entries
at the bottom. Timestamps are local (with timezone).

Complements — does not replace — the handoff discipline in `AGENTS.md`
(update `TODO.md` / `DECISIONS.md` / `PLAN.md` at the end of meaningful work).
Log **during** work too, not only at the end.

Entry format:

```
## YYYY-MM-DD HH:MM TZ — <machine/agent> — <one-line summary>
- what was done / found / decided
- state right now (what's green, what's broken, what's mid-flight)
- next action if this session dies here
```

---

## 2026-07-01 (from git history) — baseline: repo created

Reconstructed from commits, for the record:
- `6fbad19` (00:59) SymComp AI handoff package (`handoff/` docs).
- `21954af` (01:07) Agent + Euler bootstrap docs (`AGENTS.md`, `docs/euler_pipeline.md`, `cluster/euler_gpu_smoke.sbatch`).
- `06eaf16` (06:10) SymComp experiment scaffold (`symcomp/`: ~1,600 lines — core
  modules, tests, configs, SLURM templates; `gen_data.py` / `run_task.py` are
  docstring-spec stubs).
- `e0c61c4`→`c066c59` (07:13) `external/ViT_FM` added **temporarily for
  inspection** (PR #1) — external ViT foundation-model codebase, not part of
  SymComp proper.

## 2026-07-04 02:20 CEST — Claude Code (Alienware/WSL2) — session start, handoff read

- Read all five `handoff/` docs + `AGENTS.md`. Working tree clean on `main`
  at `c066c59`. No work had been logged between 2026-07-01 and now.
- Project state per handoff: nothing implemented beyond the scaffold; top
  blocker is durable Euler storage (repoint `$SCRATCH` → `/cluster/work/<group>`,
  add `registry.py` run registry); then `gen_data.py`, then `run_task.py`;
  Stage A freeze target ≈ day 14 (~2026-07-14).
- Created this SESSION_LOG.md (user request: always keep a timestamped
  progress log for handoffs/crashes).
- Launched a multi-agent survey of the repo to verify actual implementation
  status vs. what the handoff claims, and to try running
  `tests/test_physics.py` + `tests/smoke.py` locally. Results to be appended
  below.
- Next action if session dies here: read the survey results (or re-survey),
  then start handoff TODO item #1 (durable storage + run registry).

## 2026-07-04 02:26 CEST — Claude Code (Alienware/WSL2) — repo survey done; tests PASS locally

**Verified state (6-agent survey of the whole repo + local test run):**

- **Tests are green on this machine** (miniconda base env: Python 3.13.5,
  numpy 2.3.2, torch 2.8.0+cu128; no project venv exists yet — none in repo).
  - `tests/test_physics.py`: PASS — commuting split identity max rel err
    **0.00e+00** (machine zero confirmed); commutator norm monotone in eps
    (0 → 331080 across eps 0→0.8). Numpy-only, runs in seconds.
  - `tests/smoke.py`: PASS — full pipeline (tiny S1S2 benchmark, grammar +
    lample_charton arms, E1–E4) end-to-end, exit 0, ~couple of minutes on CPU.
  - Phase 0 of PLAN.md (local validation) is therefore effectively **done**
    on Alienware as of this entry.
- **Core package `symcomp/symcomp/` (10 files, ~1,174 lines) is fully
  implemented** — no NotImplementedError anywhere; all 6 representation arms
  wired end-to-end (note: `data_only` is spelled `none` in code). Pure
  library: no file I/O, no env vars, no hardcoded paths. `registry.py` does
  NOT exist yet (as the handoff says).
- **Scripts:** `run_all.py` + `aggregate.py` are real working code (local toy
  pipeline). `gen_data.py` + `run_task.py` confirmed docstring-spec stubs —
  run_task resolves the 90-cell grid and writes meta JSON, but steps 3–8
  (load shards / train / eval / CSV rows) are a TODO block.
- **Cluster layer confirms blocker #1:** every output path is `$SCRATCH`
  (sbatch_data.sh:18, sbatch_stageA.sh:27, configs/default.yaml:59, even the
  venv). `SYMCOMP_WORK_DIR` is proposed in docs/euler_pipeline.md:89-97 but
  consumed by NOTHING (grep-verified). Module loads are placeholders.
- **`external/ViT_FM`** = Bogdan Raonic's ETH CAMLab ViT-foundation-model +
  GenCFD diffusion codebase, 24 MB / 236 files (13 MB is one notebook;
  GenCFD vendored twice). Useful as the reference for the real Euler module
  stack (`stack/2024-06 gcc/12.2.0 python_cuda eth_proxy`), sbatch
  generation, `/cluster/work` conventions, Poseidon-style dataloaders. No
  secrets found, but many hardcoded personal paths + a hardcoded wandb
  entity — must stay temporary.

**New issues found (added to TODO.md):**
1. [BUG] `encoders.py:127` `enc_grammar_scrambled` seeds its shuffle with
   Python `hash()` → salted per process → **H4 control arm is not
   reproducible across runs** unless PYTHONHASHSEED is pinned. Fix before
   any real run.
2. [BUG-RISK] `capacity.py` returns a `(value, "residual…")` tuple instead
   of an int when the capacity tolerance is exceeded — callers will crash.
3. sbatch scripts: no `set -euo pipefail` (failures can exit 0);
   `|| module load eth_proxy` fallback silently swallows a failed Python
   module load; eth_proxy never loaded in sbatch_data.sh; SLURM logs land in
   the submission cwd.
4. Test-coverage gaps: ETDRK4 solvers (S2/S3 data!) have no automated
   validation; smoke covers 2/6 arms, prediction head only; discovery
   metrics in aggregate.py have no producer yet.
5. `run_all.py` E3 wraps in bare try/except → silent NaN on failure.

**State right now:** working tree has this file + TODO.md edits only; no
code changed; nothing committed. Not yet on Euler (no venv, no storage
probe run).

**Next actions if this session dies here:** (1) commit+push the handoff
updates; (2) TODO #1 — implement `WORK_DIR`-based durable storage +
`registry.py` run registry (specs in CODEX_START item 1), fixing the
scrambled-grammar hash bug alongside; (3) then `gen_data.py` per its
docstring spec.

## 2026-07-04 02:56 CEST — Claude Code (Alienware/WSL2) — task list for TODO #1 proposed, waiting on user

- Proposed to the user the ordered implementation plan for TODO #1 (durable
  storage + run registry), all parameterized via `SYMCOMP_WORK_DIR` /
  `SYMCOMP_HOME_ARCHIVE` env vars so no Euler access is needed to write and
  test the code locally.
- Asked the user for: (a) the Euler storage-probe outputs (commands in
  `docs/euler_pipeline.md` "Durable Storage Probe": `my_share_info`,
  `lquota /cluster/work/<group>`, `lquota` home) to fill in the real group
  path + quota; (b) the current default Euler module stack (`module avail`)
  to replace the placeholder module loads; (c) go-ahead to implement; (d)
  whether to commit+push the pending handoff updates.
- Next action if this session dies here: same as previous entry — the plan
  is not blocked on Euler, only the final path value is.

## 2026-07-04 07:38 CEST — Claude Code (Alienware/WSL2) — TODO #1 implemented; mid-review checkpoint

- User gave go-ahead (env-var-parameterized, start before Euler probe) and
  approved commit+push; handoff updates pushed as `f3b664f`.
- **Implemented since then (working tree, uncommitted):**
  - `symcomp/symcomp/registry.py` (NEW): SYMCOMP_WORK_DIR/SYMCOMP_HOME_ARCHIVE
    resolution (refuses scratch-only cluster runs), `Run.create` →
    `runs/<run_id>/` with config.json + manifest.json, file-locked append-only
    master CSV with the fixed Stage-A schema, `get_run`/`list_runs`,
    `archive_to_home` (size-capped, skips checkpoints).
  - Fixed both survey bugs: `encoders.py` scrambled-grammar now seeds from
    sha256 (H4 control reproducible across processes); `capacity.py` both
    resolvers now return `(out, target, residuals)` — values always usable.
  - Wired: `run_task.py` registers runs via registry; `run_all.py --register`
    mirrors local runs into the registry; `configs/default.yaml` cluster
    section documents work_dir/home_archive; both sbatch scripts hardened
    (`set -euo pipefail`, unconditional eth_proxy, SYMCOMP_WORK_DIR guard,
    logs/ prefix + gitkeep).
  - `tests/test_registry.py` (NEW): 7 checks incl. 8-process concurrent
    append (200/200 rows intact) and cross-PYTHONHASHSEED scrambled
    determinism. **All green**; physics + smoke also re-run green.
- **Review state:** 4-lens adversarial review produced 22 raw findings, but
  the usage limit hit mid-verification (~03:15 CEST) — findings are
  UNVERIFIED. Session resumed 07:38 CEST; triaging them inline now. Key
  themes: fcntl.flock semantics on Lustre `/cluster/work` (unverifiable off-
  cluster → need per-run rows.csv as source of truth + master rebuild
  helper + on-cluster probe), no run_id column → requeue duplicates not
  dedupable, run_id 1-second collisions, non-atomic home-archive copy,
  venv itself lives on purgeable scratch, README/handoff staleness.
- Next action if this session dies here: apply the triaged fixes (run_id
  column + rebuild_master + collision suffix + atomic archive copy + venv
  relocation in sbatch + README/docs), rerun tests, update handoff docs,
  commit + push.

## 2026-07-04 07:44 CEST — Claude Code (Alienware/WSL2) — TODO #1 COMPLETE (code side); all tests green

- Applied all triaged review fixes:
  - registry.py: `run_id` provenance column auto-stamped on every CSV row
    (requeue dedup), `rebuild_master()` regenerates master.csv from per-run
    rows.csv (source of truth; flock-failure recovery path), same-second
    run-id collision suffix, torn-tail guard on locked appends, atomic
    (tmp+rename) home-archive copy, orphan-run-dir cleanup on failed create,
    local work-dir fallback anchored to repo root.
  - sbatch scripts: venv moved OFF scratch → `${SYMCOMP_VENV:-$SYMCOMP_WORK_DIR/venvs/symcomp}`
    (purge would have killed it mid-project); WORK_DIR guard added to
    sbatch_data.sh too.
  - run_all.py `--register` now fails fast (resolves work dir before
    training); default.yaml comment no longer implies yaml env expansion;
    run_task.py TODO points at `registry.file_hashes` for shard provenance.
  - README cluster workflow rewritten (export SYMCOMP_WORK_DIR first, venv
    on work storage, aggregate from `$SYMCOMP_WORK_DIR/results/master.csv` —
    old scratch path was stale); docs/euler_pipeline.md gained a flock-probe
    section (`SYMCOMP_TEST_DIR=$SYMCOMP_WORK_DIR ... test_registry.py`).
- **Tests: registry 9/9, physics, smoke — ALL PASS** (WSL2, miniconda base).
- Handoff updated: TODO.md (survey bugs marked FIXED; storage item now
  "Euler validation only"), DECISIONS.md (+D11 storage design), PLAN.md
  (validation #1 rewritten).
- Committing + pushing this change set now.
- **Next actions for whoever picks this up:**
  1. On Euler: storage probe (`my_share_info`, `lquota`), export
     SYMCOMP_WORK_DIR/-HOME_ARCHIVE, venv on work storage, flock probe,
     re-run physics+registry tests on the cluster (TODO #1-3).
  2. Then implement `scripts/gen_data.py` per its docstring spec (TODO #4).
  3. Then `scripts/run_task.py` steps 3-8 (TODO #5).
