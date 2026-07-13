# SymComp Stage A — Results Memo (2026-07-05)

**Provenance.** Full pre-registered Stage A battery on ETH Euler: 90 cells =
6 representation arms × 5 split-seeds × 3 init-seeds, production config
(N=256, T=32, 320 ICs/operator, 80 epochs, d_model 256, capacity matched to
the pre-registered 2% — worst arm +1.9%). Data: S1 exact spectral
(machine-zero) + S2 well-posed variable-coefficient sweep (D12). Jobs
5746065 (data), 5746066 + 5781548 (cells; 2 node-failure requeues, deduped
by run_id). Frozen artifacts: `symcomp/results_stageA/` (master.csv, 2088
rows; money_plot.png). Every run carries config + git SHA + seeds + data
manifest hash. Analysis code: `scripts/aggregate.py` (paired bootstrap,
stratified H2).

## Panel → hypothesis decisions (per pre-registered rules)

**H1 (headline: grammar > flat reps on zero-shot composition): NEGATIVE.**
Paired over split-seeds (positive = grammar better), rel_l2, compose:

| baseline | grammar − baseline | 95% CI | verdict |
|---|---|---|---|
| coeff_vector | −0.037 | [−0.040, −0.033] | coeff_vector BETTER |
| none (data-only) | −0.012 | [−0.020, −0.004] | data-only slightly better |
| lample_charton | −0.006 | [−0.010, −0.002] | slightly better |
| prose_tree | −0.002 | [−0.008, +0.006] | tie |
| grammar_scrambled | −0.000 | [−0.008, +0.006] | tie |

Grammar confers no zero-shot composition advantage at matched capacity; the
non-symbolic coefficient vector is marginally the best conditioning channel.
(Sign-test floor with 5 seeds is p≈0.06 as pre-registered; CIs are primary.)

**H4 (scrambled-grammar killer control): grammar == scrambled.** Per the
pre-registered rule, no compositional-structure claim is licensed. Combined
with H1, the conclusion is symmetric and clean: whatever the symbol channel
contributes, compositional syntax adds nothing beyond it at this scale.

**H2 (error grows with ‖[A,B]‖): CONFIRMED — the pivot headline.**
Within the pre-registered anchor sweep (operator identity held fixed,
epsilon sweeps the commutator):

| arm | Spearman ρ | Δ rel_l2 (max−min ‖[A,B]‖) |
|---|---|---|
| coeff_vector | +0.72 | +0.062 |
| grammar_scrambled | +0.68 | +0.048 |
| prose_tree | +0.66 | +0.053 |
| lample_charton | +0.66 | +0.053 |
| grammar | +0.64 | +0.049 |
| none (data-only) | +0.48 | +0.040 |

Zero-shot error degrades monotonically with the commutator for EVERY arm —
no representation escapes the BCH correction. This is the pre-registered
mechanism ("symbols carry the generator, not the flow") and it holds
regardless of representation choice. NOTE: the naive pooled regression
(also reported by aggregate.py) is stratum-confounded and near-zero — the
stratified test is the correct pre-registered one; keep both in the paper
for transparency.

**H3 (grammar's edge larger on discovery): NOT YET TESTABLE.** The
autoregressive decoder is unimplemented; the rep-agnostic baseline head
(trajectory-prefix input, D13) shows no separation as expected
(mech_f1 ≈ 0.67, coef_mae ≈ 0.31, all arms).

**H5 (decompose asymmetry): DIRECTIONAL DATA ONLY.** Decompose rel_l2 ≈
0.29–0.33 (vs compose ≈ 0.25–0.30), one held-out primitive per seed; the
nu→0 singular-limit test is Stage C. No decision taken.

## E2/E3 interventions (added 2026-07-05; independent full retrain)

The null is **strong-form, not vacuous** — the models demonstrably use the
symbol channel:

- **E2 (channel masking):** ablating symbols at eval costs +0.063–0.069
  rel_l2, positive in **100% of variant-cells** for every symbolic arm
  (data-only arm exactly 0.000 — ablation no-op sanity check). The symbol
  channel carries roughly as much error-reduction as the entire commutator
  degradation range.
- **E3 (counterfactual swap):** feeding the anchor's data with a PURE-
  advection symbol pulls predictions 27–32% of the way toward the wrong
  (pure) solution — the channel is causally steering, in every arm.
- **Replication:** this rerun independently retrained all 90 cells; H1, H2
  (stratified), and H4 reproduced within CI wiggle. Stage A conclusions
  rest on two complete training rounds.

One-sentence synthesis: *models consume the symbols and are steered by
them; the symbolic FORM is irrelevant (down to scrambled syntax); and no
form escapes the commutator law.* Symbols are, in effect, a lookup of the
generator's coefficients — which is why the minimal coefficient vector is
the best (and cheapest) conditioning channel.

## Stage AX — exploratory 16-arm generality sweep (2026-07-06; NOT pre-registered)

240 cells (16 representations × 15 seed-pairs), same protocol, max_len 48.
Multiple-comparison and covariate caveats apply (sequence length, realized
vocab differ by arm; slot/bag arms trigger capacity reallocation to the
data branch). As measured:

- **The commutator law is universal: Spearman ρ positive for all 16 arms**
  (+0.32 slot_vector … +0.75 physics_typed_tags).
- **Semantics does not beat syntax:** `fourier_symbol` (tokenized samples
  of the operator's own Fourier multiplier — the sufficient statistic in
  the commuting stratum) ties grammar (+0.005, CI spans 0).
- **Serialization order is irrelevant:** postfix vs prefix (exact
  minimal pair) statistically indistinguishable.
- **An ordering exists at 15-seed resolution** (grammar-minus-arm,
  positive = grammar better): numeric-flavored conditioning best
  (coeff_vector −0.052; digit_p10 −0.026; unary_order −0.025), classic
  tree serializations mid-pack (−0.004…−0.010), heavily-structured
  layouts worst (subgrammar_typed_rules +0.012; slot_vector +0.051).
  The slot_vector↔coeff_vector gap (≈0.10) isolates the CONTINUOUS-value
  input pathway, not the fixed-slot layout, as coeff_vector's advantage.

## H3 probe (AR discovery decoder; single grammar cell, 2026-07-06)

The autoregressive decoder (rep's own vocab, conditioned on observed
trajectory frames) decodes TRAINING operators nearly perfectly (11/12
strict exact-match in the probe) but essentially never composes the name
of a HELD-OUT combination. Full battery (Stage AD, 90 cells, 2026-07-06):

| arm | exact_match (held-out compose) | S1 | S2 |
|---|---|---|---|
| grammar | 0.0014 | 0.0031 | 0.0000 |
| prose_tree | 0.0003 | 0.0006 | 0.0000 |
| lample_charton | 0.0000 | 0.0000 | 0.0000 |
| grammar_scrambled | 0.0000 | 0.0000 | 0.0000 |

All ≈ 0 with no meaningful separation between syntaxes; mechanism-level
partial credit (mech_f1 ≈ 0.67) is unchanged across arms. The same models
predict these compositions at rel_l2 ≈ 0.2–0.3. **Naming a law and
simulating it dissociate, and the failure to compose names is
representation-independent** — grammar's "composition = one production"
inductive bias did not rescue the decoder. (Strict-metric caveat: exact
sequence match; mech_f1 shows partial mechanism recovery.)

Secondary observation, not a claim: under the joint AR objective, real
grammar's PREDICTION slightly exceeded scrambled for the first time
(+0.009, CI[+0.002,+0.016]; single exploratory stage) — a Stage B check
is warranted before interpreting.

## Caveats
- Single fusion (xattn) + backbone (transformer); Stage B robustness sweep
  pending — H1's negative should be confirmed across fusion/backbone/scale
  before it is called final.
- Stage AX and AD are exploratory (not pre-registered); report them as
  generality/mechanism probes, never pooled with Stage A statistics.
- Discovery baseline only (see H3). 5 split-seeds → sign-test floor 0.06.
- The 18-cell local (Alienware) partial replication was discarded (mixed
  split-code versions); Euler is the sole source of truth for Stage A.

## Recommended next steps (in order)
1. Stage B robustness on the pivoted headline (does the commutator law and
   the H1 null survive fusion/backbone/scale?).
2. AR discovery decoder → H3 becomes testable (the last place a
   representation effect could plausibly live).
3. E2/E3 interventions at Stage A scale; more split seeds if p<0.05 wanted.
4. Re-run the prior-art check before any write-up (scoop risk is on the
   commutator law, which is now the headline).

## Additions (2026-07-07): direction probes, H5, derivative-level grammars, scale

- **Order-swap probe (330 checkpoints):** feeding term-order-reversed symbols
  (dif+adv for the trained adv+dif) changes rel_l2 by |gap| <= 2e-4 in every
  arm; order-free arms give exactly 0.0000 (built-in sanity anchors). The
  fusion is architecturally order-blind — A+B == B+A everywhere.
- **ADEC (train mixtures -> recover pure laws, 90 cells): H5 CONFIRMED.**
  Prediction recovers never-seen-alone diffusion at rel_l2 0.08-0.17 and
  advection at 0.22-0.31 — the pre-registered regular/singular asymmetry, in
  all 6 arms. Naming the pure laws: 0.000 everywhere.
- **ADRV (derivative-level representations, 105 cells): the study's FIRST
  representation effect.** Naming unseen compositions moves off zero only
  when mechanisms share dx substructure, ordered by compositional structure:
  algebra-typed CFG 0.018 > untyped CFG 0.012 > flat infix vocabulary 0.005
  > mechanism-level arms 0.000-0.003. Absolute levels remain small (~2%
  exact match; hits in ~4-5% of variant-cells) — "naming BEGINS to compose
  under derivative-level grammars"; needs more seeds before a strong claim.
  Prediction also improves CI-clean vs mechanism-grammar (-0.007..-0.013).
- **Stage B, scale axis (d_model 512, 60 cells): the H1 null replicates**
  (coeff_vector -0.028 vs grammar; scrambled tie). 7/375 grammar rows nan
  (partial divergence at 512) — excluded, flagged. fusion=film and
  d_model=128 arrays failed on the 2% capacity gate (integer-width
  granularity; 'none' arm +2.6%) — resubmitted at the pre-registered
  fallback tolerance 3.5% with exact per-arm counts reported.

## Additions (2026-07-08): replication, naming-across-all-arms, fusion/scale

- **The naming ordering REPLICATES on fresh init seeds** (ADRV2, 105 cells):
  derivative-CFGs 0.011-0.014 > unrolled vocabularies 0.006-0.008 >
  mechanism-level 0.000-0.001 exact-match. Combined 30 cells/arm:
  deriv_typed_cfg 0.016, deriv_cfg 0.012, unary/infix ~0.006, grammar
  0.0014, lample 0.0000. The CFG-vs-vocabulary gap (~2x) is robust; the
  typed-vs-untyped edge narrowed (0.014 vs 0.011 on fresh seeds) — typing
  helps at most weakly.
- **AXD (decoder over all 16 AX arms) reframes the naming result:** best
  namers are fourier_symbol (0.0149) and physics_typed_tags (0.0107) —
  arms whose tokens encode PHYSICALLY INFERABLE quantities (spectral bins,
  order/parity/class), not conventions. With the ADRV result (dx-counts
  also inferable), the cleaner statement is: **decoders generalize to the
  degree token semantics are grounded in the observable dynamics, and fail
  where tokens are conventional** (mechanism names, L&C symbols, scrambled:
  0.000-0.003). fourier "naming" is partially measurement rather than
  symbolic composition — an interpretive caveat that cuts both ways.
- **Stage B, fusion axis (film, 60 cells): large architectural interaction.**
  FiLM fusion cripples token-sequence conditioning (grammar ~0.20 rel_l2
  WORSE than coeff_vector/none) while the token arms still tie each other
  (grammar-scrambled +0.002). The form-null among token arms survives;
  numeric-vector conditioning is additionally ROBUST TO FUSION CHOICE in a
  way token sequences are not.
- **Stage B, small scale (d_model 128): null holds** (coeff_vector -0.058;
  scrambled tie). With B512 earlier: the H1 null now spans d_model
  128/256/512 and xattn/film fusion.
- Straggler note: ADRV cell 88 (deriv_cfg, s4i1) timed out twice on
  different nodes — excluded (104/105); investigate before final freeze.

## Additions (2026-07-07, second batch): replication + full Stage B

- **The naming ordering REPLICATES on fresh init seeds (ADRV2, 105 cells):**
  typed CFG 0.0139 > untyped CFG 0.0113 > unrolled vocabularies 0.0065-0.0080
  > mechanism-level 0.0000-0.0014. Two independent seed batteries now agree;
  the first representation effect is replicated, not a fluke.
- **AXD (decoder over all 16 AX arms):** best namers are the semantically-
  grounded arms — fourier_symbol 0.0149, physics_typed_tags 0.0107 — above
  every mechanism-word arm. Naming succeeds to the degree the target
  sequence is computable from dynamics-grounded shared structure.
- **Stage B complete (film + 128 + 512): the two core nulls are fully
  robust, plus one new finding.** (a) grammar == scrambled in all four
  architectures; (b) coeff_vector >= all arms in all four; (c) NEW: under
  FiLM fusion the token-symbol arms COLLAPSE (rel_l2 0.43 vs data-only
  0.23) while vector conditioning is unaffected — the coeff_vector
  advantage grows monotonically as fusion/scale weakens (0.030 at d512,
  0.037 at d256/xattn, 0.058 at d128, 0.22 at FiLM). Token-sequence
  reading is the fragile component; numeric conditioning is robust.
  (Known: 7 nan rows in B512 grammar, excluded; ADRV cell 88 timed out
  twice, 104/105.)

## Stage BEST (2026-07-08): the constructive result

Decoupling the conditioning representation from the naming vocabulary
('coeff_vector@deriv_typed_cfg': continuous coefficients IN, algebra-typed
derivative CFG OUT) achieves BOTH optima simultaneously in one
capacity-matched model (60-cell battery):

| arm | prediction | naming unseen |
|---|---|---|
| coeff_vector@deriv_typed_cfg | 0.211 | 0.0139 |
| coeff_vector (no decoder) | 0.203 | — |
| deriv_typed_cfg (tokens both ways) | 0.257 | 0.0168 |

Prediction at float-conditioning level (+0.008 objective-sharing cost, vs
+0.046 saved over token conditioning); naming inside the typed-decoder's
replicated 0.014-0.018 band. Every ingredient of the recipe traces to a
controlled result. This is the study's constructive deliverable: numbers
for the physics, grammar for the description, one model.

## Anticipated question: "Why did PROSE work, then?"

PROSE's results are consistent with — indeed predicted by — ours:
(1) its "symbols help" effect is real and we reproduce it (masking a
trained model's symbol channel costs +0.06–0.08 rel-L2) — but the arm
comparison PROSE never ran shows the naked coefficient vector delivers
the same benefit: the cause is the numbers inside the equation, not its
form; (2) PROSE's only OOD test is wider coefficient ranges on the SAME
equation families (verified) — exactly the regime where the coefficient
channel shines; unseen forms/compositions were never tested; (3) its
symbolic-decode success (>99.9% validity, 0.01% error) is on SEEN
operators — we reproduce that cell (11/12 exact on trained laws); the
near-zero decode lives in the cell they never evaluated; (4) PROSE has
no capacity-matched data-only, coefficient-only, or scrambled control.
"The bi-modal system performs well" and "the symbolic modality causes
it" are different claims: PROSE established the first, we tested the
second.
