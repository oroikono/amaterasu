# TODO

> Progress is logged with timestamps in `SESSION_LOG.md` — append an entry
> there whenever you start/stop meaningful work.

## Open issues (2026-07-04 repo survey — see SESSION_LOG.md for detail)
- **[TEST GAP]** ETDRK4 solvers (`solve_varcoeff_advdiff`, `solve_burgers`)
  — the S2/S3 data generators — have no automated validation (only S1 is
  machine-zero tested). Matches PLAN "needs validation" #8; add a refined-
  reference spot check.
- **[MINOR]** `run_all.py` E3 bare try/except silently records NaN;
  `aggregate.py` discovery metrics currently have no producer (run_task.py
  unimplemented).
- **[UNVERIFIED, check on Euler]** `fcntl.flock` coherence on the
  `/cluster/work` mount — probe per docs/euler_pipeline.md before the Stage A
  array; `registry.rebuild_master()` is the recovery path if it fails.

## Status notes (2026-07-04)
- Phase 0 local validation is DONE on Alienware: `tests/test_physics.py`
  (machine-zero identity, monotone commutator) and `tests/smoke.py` both
  PASS (miniconda base, torch 2.8.0+cu128). No project venv exists yet.
- FIXED (07-04, tested): scrambled-grammar salted-hash bug (H4 now stable,
  sha256-seeded); capacity.py tuple return (now `(out, target, residuals)`);
  sbatch hardening (`set -euo pipefail`, unconditional eth_proxy, logs/,
  SYMCOMP_WORK_DIR guards, venv moved off scratch).
- DONE (07-04, code side): durable storage + run registry
  (`symcomp/symcomp/registry.py` + `tests/test_registry.py`, 9 checks green
  incl. 8-process concurrent append and rebuild-from-run-dirs). Cluster-side
  validation still pending (item 1 below).

## Immediate next actions (this week)
1. **[BLOCKER → Euler validation] Validate durable storage on Euler.**
   Code is implemented (registry.py; runs/<run_id>/ + manifest + locked
   master CSV + home archive). Remaining, on the cluster:
   - Confirm group work path + quota: `my_share_info`; `lquota /cluster/work/<group>`.
   - `export SYMCOMP_WORK_DIR=/cluster/work/<group>/symcomp` (+ HOME_ARCHIVE);
     add both to the login shell profile.
   - Run the flock probe (docs/euler_pipeline.md): `SYMCOMP_TEST_DIR=$SYMCOMP_WORK_DIR
     PYTHONPATH=. python tests/test_registry.py`.
   - Verify a run can be fetched by ID after >15 days (calendar reminder).
2. **Stand up the Euler venv — on work storage, NOT scratch**
   (`$SYMCOMP_WORK_DIR/venvs/symcomp`; sbatch scripts default to this, or
   override via `SYMCOMP_VENV`). Match the module stack + torch CUDA build to
   the current cluster default (placeholders in `cluster/*.sh`). Pin requirements.
3. **Run physics validation on the cluster:** `python tests/test_physics.py` must
   print the machine-zero commuting identity and the monotone commutator.
4. ~~Implement `scripts/gen_data.py`~~ DONE 2026-07-05 (+ batched solvers,
   well-posed S2 per D12, solver validation tests). Cluster launch pending.
5. ~~Implement `scripts/run_task.py`~~ DONE 2026-07-05 (both heads, capacity
   asserted at 2%, D13 input/split fixes, registry rows). Dry-runs green on
   local cuda (mini + production config). Cluster dry-run pending.
5b. **Launch on Euler:** `bash cluster/euler_bootstrap.sh` from the symcomp/
   root on a login node does EVERYTHING (storage probe → venv on work →
   cluster tests incl. flock probe → data array → Stage A array gated on it).
   Blocked only on SSH/login access (Claude's WSL key not authorized yet).

## Later actions
6. **Launch full Stage A** (90 tasks) once the dry-run is green; aggregate to the
   H1 table, H2 regression, H4 panel, money plot. **This is the freeze point.**
7. **Add the autoregressive discovery decoder** + discovery metrics (exact-match,
   mechanism F1, coefficient MAE) for the symbolic arms.
8. **Stage B:** fusion (xattn/FiLM), backbone (add the VAE arm), scale sweep,
   noise levels. Confirm H1 survives.
9. **Stage C:** data-budget curve; decompose direction + nu->0 singular limit (H5);
   the S3 Burgers/cubic rung; finalize commutator regression.
10. **Close the capacity residual** on coeff_vector/data_only or report exact
    per-arm param counts in the paper table.
11. **Write the results memo** mapping each panel to H1–H5; archive a frozen copy
    of results+configs to project storage.
12. **Re-run a prior-art check** immediately before any submission.

## Risks / blockers
- **[HIGH] Scratch purge (15 days).** Early runs deleted before month end if
  storage isn't repointed. Mitigation: action #1 above, do it first.
- **[HIGH] Scoop risk.** A commutator-vs-error correlation from another group
  (this area is hot: SymPlex, Neural Operator Splitting, HyCOP, equation-aware NO
  are 2025–26). Mitigation: move fast on Stage A; re-check prior art before
  submission; the representation comparison + bidirectional decompose test remain
  differentiators even if the commutator law gets partially scooped.
- **[MED] "Grammar bakes in the answer" referee attack.** Mitigation: the
  scrambled-grammar control (H4) must be run and reported prominently; also include
  compositions needing >1 production and nonlinear/product terms the grammar does
  not trivially encode.
- **[MED] Capacity confound.** Mitigation: enforced param matching + scale
  ablation + report exact counts.
- **[MED] Timeline.** 6–8 mechanisms × 2 tasks × all ablations in a month is
  tight. Mitigation: the day-14 freeze point guarantees a standalone result.
- **[MED] Solver fidelity on S2/S3.** Variable-coeff/Burgers use ETDRK4; spot-check
  against a refined reference. S1 is exact and already validated.
- **[LOW] Significance floor.** 5 split-seeds gives a paired sign-test floor of
  p≈0.06; add seeds if a sub-0.05 headline is desired.
- **[LOW] Statistical-direction bug class.** The aggregator handles
  higher-is-better vs lower-is-better metrics via an explicit flag; ensure new
  metrics are registered with the correct direction.

## Definition of done (month 1)
- Stage A complete with durable, retrievable results on Euler.
- Money plot + H1 table + H4 control panel + H2 regression generated from the
  durable master CSV.
- 2-page results memo mapping panels to H1–H5 with the decision taken per the
  pre-registered rules.
- All runs reproducible (config + git SHA + seed + data-manifest hash per run).
