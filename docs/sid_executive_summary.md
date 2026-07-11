# SymComp — Executive Summary for Sid (2026-07-08)

**North star (your framing).** Toward a multimodal symbolic-numeric
foundation model that zero-shots to new physics by exploiting
compositionality, the program's question was: *which symbolic tokens buy
that?* Answer, with controls: (1) for zero-shot BEHAVIOR — none: a
continuous parameter channel beats every token scheme everywhere tested;
(2) for zero-shot DESCRIPTION — algebra-typed derivative-level grammars
(vocabulary built from the physics' generators, not phenomenon names) —
the only replicated positive; (3) zero-shot reliability on new physics is
predictable from the operator algebra (the commutator law) — a trust
certificate no token can override. The Stage BEST decoupled model
(parameters in, typed grammar out) is the seed architecture; Stage SYM
(running) tests whether learned composition transfers through symbols
alone to mechanisms never seen mixed. Scaling order that keeps this
evidence-first: coefficient ranges → mechanisms/nonlinear/2D → model scale.

**Bottom line.** The pre-registered program is complete, twice-replicated at
its core, and lands three results: (1) a **universal commutator law** —
zero-shot compositional error rises with ‖[A,B]‖ in every representation and
architecture tested; (2) a **strong-form representation null with a
dissociation** — models consume symbols as coefficients (form-irrelevant down
to scrambled syntax) and can *simulate* unseen mechanism compositions but
essentially cannot *name* them; (3) a **replicated constructive exception** —
algebra-typed derivative-level grammars are the only representation feature
that ever moved naming (≈1.4–1.8% strict exact-match, three independent
batteries), and a decoupled numbers-in/grammar-out model achieves the best
prediction *and* best naming simultaneously. Full evidence: ~2,100 trained
capacity-matched multimodal models on Euler, every run reproducible
(config+SHA+seeds+data hash). Everything below is in
`docs/stageA_results_memo.md` with per-claim tables.

## Verdicts on the pre-registered hypotheses

| # | hypothesis | verdict |
|---|---|---|
| H1 | grammar > flat reps (prediction) | **Negative** — replicated ×2, robust across xattn/FiLM/128/256/512; coefficient vector best everywhere |
| H2 | error grows with ‖[A,B]‖ | **Confirmed, universal** — ρ +0.32…+0.75 across all 16 arms; the pivot headline per decision rules |
| H3 | grammar's edge on discovery | **Dissociation** — naming ≈0 in all directions at mechanism level; first nonzero + replicated ordering only for derivative-level typed CFGs |
| H4 | scrambled control | Ties real grammar in every architecture — no structure claim at mechanism level |
| H5 | decompose asymmetry | **Confirmed** — pure diffusion recovered 2–3× better than pure advection (regular vs singular limit), all arms |

## What we know beyond the hypotheses

- **The null is causal-grade:** masking symbols costs +0.06–0.08 rel-L2 in
  100% of cells; wrong symbols steer predictions ~30%; fusion is
  architecturally order-blind (A+B ≡ B+A, gap ~2×10⁻⁴, 330 checkpoints).
- **What matters in the symbol channel:** the continuous-value pathway
  (coeff_vector vs its discrete twin: ≈0.10 gap); token reading is fragile
  (FiLM collapses token arms to worse-than-no-symbols while vectors are
  unaffected).
- **What moves naming (and what doesn't):** shared derivative substructure +
  algebraic typing help (three batteries: 0.018/0.014/0.016); constituent-
  aligned chunking is neutral-positive; statistical BPE (pre-registered
  control) does not help; plan-first linearization and unary numerals
  actively hurt. Semantic grounding helps (Fourier-symbol tokens are the
  best-naming round-1 arm).
- **The recipe (Stage BEST):** condition on coefficients, decode in the typed
  derivative CFG — prediction 0.211 (vs 0.257 token-conditioned) and naming
  0.0139 in one capacity-matched model.

## Decisions needed

1. **Scaling scope** — evidence-first path we recommend: coefficient RANGES
   first (the thinnest current axis; makes the "numbers" story real), then
   nonlinear/2D rung, then model scale. Each is a config away except 2D.
2. **Write-up target** — the paper is "the laws of the symbolic lane in
   equation-conditioned neural operators": pivoted headline (commutator law)
   + strong-form null + dissociation + the constructive recipe. Prior-art
   re-check is running now; novelty statement will be appended when it lands.
3. **Known blemishes we will disclose:** exact-match strictness (mech_f1
   ≈0.67 shows partial recovery), fixed coefficient per mechanism in Stages
   A–R2, 1D linear universe + one nonlinear rung untested at scale, 7 nan
   rows at d512, one twice-timed-out cell (104/105).

*Figures: `docs/figures/` — pipeline_diagram, money_plot, h1_forest,
e2_leverage, ax_ranking, h2_rho_all_arms, h3_dissociation, latent_tokens,
latent_operators, capability_matrix.*
