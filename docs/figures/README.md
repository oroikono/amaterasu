# SymComp result figures (generated from the frozen master CSV, 2026-07-06)

All figures are computed from Euler-run data only (`results_stageA/` frozen
CSVs); nothing is illustrative. One line per figure:

| figure | what it shows |
|---|---|
| `money_plot_stageA.png` | **H2, the headline:** zero-shot error rises with ‖[A,B]‖ for every representation (anchor ε-sweep, difficulty held fixed) |
| `h1_forest_stageA.png` | **H1 negative:** grammar − baseline paired CIs; coefficient vector best, scrambled ties grammar |
| `e2_leverage.png` | **E2:** masking the symbol channel costs +0.06–0.07 rel-L2 in every arm — models genuinely use symbols |
| `ax_ranking.png` | **Stage AX (exploratory):** all 16 symbolic encodings ranked; numeric conditioning best, exotic structure worst |
| `h2_rho_all_arms.png` | **The law is universal:** Spearman ρ positive for all 16 arms |
| `h3_dissociation.png` | **H3:** decoders name *seen* laws (0.92) but essentially never compose the name of an *unseen* combination (≤0.0014) |

Model accounting (trained multimodal models — data branch + symbol branch,
cross-attention fusion, capacity-matched):

| sweep | job(s) | models |
|---|---|---|
| Stage A (pre-registered) | 5746066 + 5781548 | 90 |
| Stage A replication + E2/E3 | 5818788 | 90 |
| Stage AX (16 arms) | 5847283 | 240 |
| Stage AD (AR decoder) | 5847781 + 5882055 | 90 |
| **total (Euler, used in results)** | | **510** |

(~27 additional local validation trainings and 2 requeue duplicates are
excluded from all statistics; requeues deduped by run_id.)
