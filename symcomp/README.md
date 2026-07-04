# SymComp — does the symbolic channel carry compositional generalization?

**→ Read `EXPERIMENT_PLAN.md` first.** It is the pre-registered, referee-proofed
month-1 plan (headline claim, defensibility table, ablation matrix, timeline,
decision rules, Euler ops). This README is the quickstart.

Headline (H1): a **grammar-structured** symbolic representation beats flat
symbolic reps (Lample–Charton prefix, PROSE tree) and a coefficient-vector
control on **zero-shot generalization to held-out mechanism compositions**, at
matched capacity. Mechanism (H2): success is predicted by the **commutator**
`||[A,B]||` — symbols carry the generator, not the non-abelian flow correction.
Killer control (H4): a **scrambled grammar** isolates that it's the compositional
*structure*, not grammar-ness.

## Install
```bash
pip install -r requirements.txt        # numpy, scipy, torch, matplotlib
```

## Validate the physics first (seconds, CPU)
```bash
PYTHONPATH=. python tests/test_physics.py
```
Confirms (1) the commuting split identity `exp(t(A+B)) = exp(tA)exp(tB)` holds to
machine zero for constant-coefficient operators — this is the ground-truth
cleanliness that makes rung 1–2 near-theorem-quality — and (2) the
variable-coefficient commutator `||[a(x)∂_x, ν(x)∂_xx]||` increases monotonically
with the sweep parameter ε (the continuous "commutator knob").

## Smoke test the full pipeline (≈1 min, CPU)
```bash
PYTHONPATH=. python tests/smoke.py
```
Runs all four experiments on a tiny benchmark for two representations. Numbers
are meaningless at this scale; this only proves the machinery executes.

## Run the real sweep (GPU recommended)
```bash
# Stage A — representation as the controlled variable (the headline comparison)
PYTHONPATH=. python scripts/run_all.py --stage A --rung S1S2 --epochs 80 --seeds 5

# Stage B — fusion sweep on the winning representation
PYTHONPATH=. python scripts/run_all.py --stage B --rung S1S2 --epochs 80 --seeds 5

# climb to the nonlinear/singular rung once S1S2 is clean
PYTHONPATH=. python scripts/run_all.py --stage A --rung S1S2S3 --epochs 120 --seeds 5
```
Outputs land in `results/`: a `*_results.csv` (one row per commutator bin per
config per seed) and a `*_curve.png` (zero-shot rel-L2 vs normalized commutator,
one line per representation).

## The three controlled axes
- **Representation** (`encoders.py`): `lample_charton` (prefix/Polish),
  `prose_tree`, `grammar` (production rules), `coeff_vector` (non-symbolic
  control), `none` (data-only floor). All share a vocab and matched embedding
  capacity.
- **Fusion** (`model.py`): `concat`, `xattn` (= PROSE Feature-Fusion baseline),
  `film`, `none`.
- **Backbone**: transformer encoder-decoder is wired; VAE / in-context are the
  Stage-C stubs to add.

## The commutator-stratified benchmark (`dataset.py`)
- **S1 commuting** — constant-coeff linear, `||[A,B]|| = 0`. Sanity floor:
  composition is exactly determined by the additive generator, which is what the
  symbol channel carries. If grammar fails here, something is broken.
- **S2 weak-noncommuting** — variable-coeff linear, `||[A,B]||` swept by ε. The
  money stratum: the degradation curve lives here.
- **S3 strong/singular** — Burgers (nonlinear advection), ν→0 vanishing-viscosity
  asymmetry. The hard rung where "generator vs flow" is a hypothesis, not algebra.

Splits are built in **both directions**: compose (`test_compose_idx`) and
decompose (`test_decompose_idx`, the apparently-unexplored composite→pure
direction).

## What each experiment proves (`experiments.py`)
- **E1 — composition curve.** Zero-shot rel-L2 vs `||[A,B]||` per representation.
  *Expected:* all representations tie at `||[A,B]||=0`; flat Lample–Charton
  collapses fastest as the commutator grows; grammar holds longest. **The
  grammar–flat gap widening with the commutator is the paper in one figure.**
- **E2 — channel masking.** Mask the symbol channel at test. *Expected:* large
  symbol-leverage in S1 (symbols carry composition), shrinking in S2/S3 (the
  carried generator becomes insufficient for the flow). If masking barely changes
  error, the data did the work and the symbol is decorative — a result you need
  to know.
- **E3 — counterfactual swap.** Feed symbol(A+B) with data(A). `symbol_causal_
  fraction` near 1 ⇒ the symbol causally drives the prediction toward the
  composite; near 0 ⇒ passive tag.
- **E4 — embedding additivity.** Tests `emb(A+B) ≈ emb(A)⊕emb(B)` in the symbol
  latent (ridge fit, language-embedding methodology). *Expected:* additive in the
  grammar latent (low residual), weak/absent in flat tokenization; and the
  residual should track `||[A,B]||` — literally "what symbols carry (additive
  generator) vs what they don't (non-abelian residual)" on one axis.

## Decision rules (what each outcome means for the paper)
- **S1 grammar ≈ flat, both good; gap opens in S2; E4 additive in grammar only** →
  the strong story holds. Write it.
- **Grammar ≈ flat everywhere** → representation isn't the lever; pivot the
  contribution entirely onto the commutator characterization (E1/E2 across
  *any* symbolic encoder vs data-only floor), which is still unoccupied.
- **E2 leverage ≈ 0 even in S1** → symbols aren't carrying composition at all;
  that is a real negative finding about joint multimodal PDE training and worth
  reporting with the commutator framing.
- **Model learns part of the Burgers commutator correction (E4 residual stays
  low into S3)** → not a failure; it's the "architecture recovers some non-abelian
  structure the symbol can't encode" finding. E4 is built to detect this rather
  than return a binary.

## Honesty flags baked in
- Matched parameter count across configs (the `none` floor widens its data branch);
  printed as `params=` so you can confirm — referees will check this.
- Raw commutator magnitudes are large (dense spectral operator norms); plots
  normalize to [0,1]. Only the *ordering* is load-bearing.
- S1/S2 are theorem-clean; S3 is a hypothesis. Keep that distinction in the text.

## Re-check before submitting
This corner is hot (SymPlex, Neural Operator Splitting, HyCOP all 2025–26). Re-run
the arXiv prior-art check immediately before submission; a commutator-vs-error
correlation from someone else is the one thing that scoops the core thesis.

---

## Cluster workflow (Euler)

```bash
# 0. durable storage FIRST (D10): scratch is purged at 15 days, unbacked-up.
#    Confirm the group path + quota (my_share_info; lquota), then:
export SYMCOMP_WORK_DIR=/cluster/work/<group>/symcomp        # durable runs/results
export SYMCOMP_HOME_ARCHIVE=/cluster/home/$USER/symcomp_archive
mkdir -p "$SYMCOMP_WORK_DIR" logs

# 1. env (adjust module versions to current Euler default — see EXPERIMENT_PLAN §9)
#    The venv lives on WORK storage, not scratch, so the purge can't kill it.
module load stack/2024-06 gcc python_cuda/3.11 eth_proxy
python -m venv "$SYMCOMP_WORK_DIR/venvs/symcomp" && source "$SYMCOMP_WORK_DIR/venvs/symcomp/bin/activate"
pip install -r requirements.txt

# 2. validate physics + the run registry on the cluster
PYTHONPATH=. python tests/test_physics.py
SYMCOMP_TEST_DIR="$SYMCOMP_WORK_DIR" PYTHONPATH=. python tests/test_registry.py
#  ^ probes flock/concurrency on the actual work filesystem (docs/euler_pipeline.md)

# 3. generate sharded data (array job; shards on scratch — regenerable)
sbatch cluster/sbatch_data.sh

# 4. run Stage A (6 reps × 5 split-seeds × 3 init-seeds = 90 GPU tasks);
#    each task registers runs/<run_id>/ + appends to results/master.csv
#    under $SYMCOMP_WORK_DIR via symcomp/registry.py
sbatch cluster/sbatch_stageA.sh

# 5. aggregate -> H1 table, H2 commutator regression, H4 panel, money plot
PYTHONPATH=. python scripts/aggregate.py --csv $SYMCOMP_WORK_DIR/results/master.csv \
    --task prediction --metric rel_l2
PYTHONPATH=. python scripts/aggregate.py --csv $SYMCOMP_WORK_DIR/results/master.csv \
    --task discovery --metric exact_match
```

`scripts/run_task.py` and `scripts/gen_data.py` carry precise implementation
specs (in their docstrings) for the two pieces to finish on Euler: the
autoregressive discovery decoder and the data-sharding loop. Everything they call
— solvers, splits, encoders, capacity matching, aggregation — is implemented and
validated here.

## What's validated locally vs to-finish on Euler
**Validated:** operator algebra + solvers (commuting identity to 1e-16, commutator
monotone in ε), 6 representation arms incl. scrambled control, combinatorial
splits with leakage assertions (both directions), matched-capacity binary search,
the four interventions, and the full stats/plot aggregation (tested on synthetic
runs — H1 CIs, H2 regression with grammar showing the shallowest commutator slope,
money plot).
**To finish on Euler (spec'd, not decided):** AR discovery decoder + discovery
metrics; VAE backbone arm; data-sharding loop; wandb sync; closing the last 2–3%
capacity residual on coeff_vector/data_only (report exact param counts in the
paper table regardless).
