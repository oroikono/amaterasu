# CONTEXT

## What this project is
**SymComp** is a research codebase and experiment program investigating whether a
**symbolic representation of a PDE operator**, trained jointly with **numerical
solution data**, enables **zero-shot generalization to held-out mechanism
compositions** — e.g. train on pure advection and pure diffusion separately, then
predict/recover advection-diffusion without ever having seen the combination.

The core scientific bet: a **grammar-structured** symbolic representation
generalizes compositionally better than flat symbolic representations (prefix /
Polish-notation à la Lample–Charton, PROSE-style trees) or a non-symbolic
coefficient vector, because in a grammar the composition of two mechanisms is a
single production rule (`OP -> OP + TERM`), whereas a flat token model must
recover that structure from a linearized sequence.

The explanatory layer: **symbols carry the generator, not the flow.** A PDE
operator is `L = sum_k c_k P_k` (a sum of mechanism terms = the generator). For
**commuting** mechanisms (constant-coefficient linear), the solution semigroup
factorizes, `exp(t(A+B)) = exp(tA)exp(tB)`, so composition is fully determined by
the additive generator — exactly what the symbol channel encodes. For
**non-commuting** mechanisms (variable-coefficient, nonlinear), the flow picks up
Baker–Campbell–Hausdorff / Zassenhaus corrections involving the commutator
`[A,B]`, which the additive syntax does **not** encode. Therefore zero-shot
compositional success should degrade with `||[A,B]||`, and the symbolic channel is
provably incomplete for the flow in exactly the "hard physics" regime.

## The main goal
Produce, within ~1 month on the ETH Euler cluster, a **referee-defensible**
result set that establishes:
- **H1 (headline):** grammar > {prose_tree, lample_charton, coeff_vector,
  data_only} on zero-shot composition rel-L2, at matched capacity.
- **H2 (mechanism):** zero-shot error increases monotonically with `||[A,B]||`.
- **H3:** grammar's advantage is larger on the **discovery** task (recover the
  operator) than on **prediction** (forecast the trajectory).
- **H4 (killer control):** a **scrambled-grammar** arm (same machinery, permuted
  non-compositional productions) does NOT match real grammar — i.e. the win is
  compositional *structure*, not "being a grammar."
- **H5:** the decompose direction (composite -> pure) is asymmetric — clean for
  the regular limit (a->0, recovers diffusion), broken for the singular limit
  (nu->0, vanishing-viscosity / advection).

Deliverable for the supervisor (Sid): the "money plot" (zero-shot error vs
commutator, one line per representation, CI bands), the H1 paired-CI table, the H4
control panel, the H2 regression, and a short results memo mapping each panel to
H1–H5.

## Current assumptions
- 1D periodic-domain PDEs; spectral solvers. Constant-coefficient linear operators
  are solved **exactly** in Fourier space (machine-zero data) — this is what makes
  the commuting-stratum result near-theorem-quality.
- 7 mechanisms total: 5 linear constant-coefficient (advection, diffusion,
  dispersion, reaction, hyperdiffusion) + 2 nonlinear (Burgers `u*u_x`, cubic
  reaction `u - u^3`). Expandable to 8 (add quadratic `u^2`).
- Three strata of increasing commutator magnitude: S1 commuting (`||[A,B]||=0`),
  S2 variable-coefficient (commutator swept continuously by a parameter epsilon),
  S3 nonlinear / singular.
- "Composition" is operator-sum composition; canonicalization makes `A+B ≡ B+A`
  so split scoring isn't inflated.
- Matched capacity across representation arms is mandatory and enforced in code
  (param counts asserted within ~2%; the data-only and coeff-vector arms widen
  their data branch to compensate for a smaller/absent symbol branch).
- The result is engineered so that **even a negative result is publishable**: the
  commuting identity is exact, and "symbols carry the abelian generator but not
  the non-abelian flow correction" is a finding either way.

## Important constraints
- **Time:** ~1 month to first results for the supervisor. There is a hard
  **freeze point** — full Stage A by ~day 14 is the minimum publishable core; if
  later stages slip, Stage A still stands alone.
- **Defensibility over volume:** the design is built backwards from a hostile
  referee's attacks (capacity artifact, cherry-picked splits, "grammar bakes in
  the answer", confounded tasks, single architecture, significance, post-hoc
  commutator story, no SOTA, leakage, easy-physics-only). Pre-registration of
  hypotheses + decision rules is the central discipline.
- **Novelty window:** the area is moving fast (SymPlex, Neural Operator Splitting,
  HyCOP, equation-aware neural operators are all 2025–26). Prior-art was checked:
  the unoccupied intersection is (held-out mechanism composition) × (controlled
  representation comparison) × (commutator law) × (discovery+prediction). Re-run a
  prior-art check immediately before submission — a commutator-vs-error
  correlation from another group is the main scoop risk.
- **Storage/retrievability (critical):** on Euler, personal scratch
  (`/cluster/scratch`) is purged after 15 days and is not backed up. Results must
  NOT live only on scratch. Durable outputs (configs, result CSVs, plots, run
  manifests, selected checkpoints) must go to home (`/cluster/home`, small +
  backed up) and/or the group's work/project storage (`/cluster/work/<group>`,
  not purged). Raw trajectory shards are regenerable and may live on scratch.

## Machines / environments involved
- **Alienware (local workstation, has GPU):** primary local dev + smoke tests +
  small training runs before pushing to the cluster.
- **MacBook Air (laptop, no CUDA):** lightweight editing, reading, writing,
  plotting, and running CPU-only physics validation / unit tests. Do not expect to
  train real models here.
- **ETH Euler cluster (GPUs, SLURM):** the production environment for the full
  sweep. SLURM array jobs; module-based software stack (LMOD); storage tiers as
  above. wandb runs in offline mode on compute nodes and is synced from a login
  node. Module versions in the scaffold are placeholders to be matched to the
  current Euler default.

## Codebase shape (as delivered)
~1,600 lines of Python. Key modules under `symcomp/`: `operators.py` (mechanism
algebra + commutator), `solver.py` (exact spectral + ETDRK4 solvers), `encoders.py`
(the 4 reps + scrambled control), `splits.py` (combinatorial held-out splits with
leakage assertions), `dataset.py`, `model.py` (matched-capacity multimodal model
with prediction + discovery heads, swappable fusion), `capacity.py` (param
matching), `train.py`, `experiments.py` (the 4 interventions E1–E4). Scripts:
`run_all.py` (local staged sweep), `run_task.py` / `gen_data.py` (Euler array-task
stubs with implementation specs), `aggregate.py` (paired-bootstrap stats + plots).
`cluster/` has SLURM templates. `configs/default.yaml` is the resolved-per-run
config. `tests/test_physics.py` validates the commuting identity to machine zero;
`tests/smoke.py` runs the whole pipeline end-to-end at toy scale.
