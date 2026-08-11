# Results — Full Reference

## Full comparison, Hits@1 (PSC-scored where applicable)

| Method | random | domain | domain_lexical | n |
|---|---|---|---|---|
| TF-IDF | 0.867 | 0.770 | 0.521 | 330 |
| BM25 | 0.855 | 0.776 | 0.521 | 330 |
| Tooka-SBERT (Persian enc.) | 0.867 | 0.779 | 0.464 | 330 |
| Persian-Emb (Persian enc.) | 0.849 | 0.755 | 0.533 | 330 |
| LaBSE (multiling. enc.) | 0.700 | 0.624 | 0.464 | 330 |
| BGE-m3 (best encoder) | 0.894 | 0.815 | 0.518 | 330 |
| Zero-shot Gemma — single-pass | 0.861 | 0.761 | 0.491 | 330 |
| Zero-shot Gemma — PSC (fair) | — | — | 0.367 | 50 |
| Zero-shot Llama — single-pass | 0.809 | 0.727 | 0.485 | 330 |
| Zero-shot Dorna — single-pass | 0.433 | 0.397 | 0.294 | 330 |
| Lean prompt + PSC (Gemma) | 0.96 | 0.88 | 0.52 | 50 |
| Chain-of-Prompting + PSC | 0.96 | 0.86 | 0.50 | 50 |
| Fine-tuning SFT-only | 0.94 | 0.84 | 0.48 | 50 |
| RAG-ICL (frozen, k=3) | 0.94 | 0.88 | 0.52 | 50 |
| **MERGE — Gemma (9B)** | 0.94 | 0.827 | **0.594** | 330 |
| **MERGE — Llama (8B)** | 0.879 | 0.809 | **0.573** | 330 |
| **MERGE — Dorna (8B)** | 0.903 | 0.815 | **0.570** | 330 |

Note: merge rows confirmed on all 330 test cases (Gemma random at n=50). Baseline/prompt/SFT/RAG rows
are n=50 subsets — say "converges around 0.52," not "exactly 0.52."

## Fair-conditions three-model baseline (pre-merge)

| Model | params | lang | random | domain | domain_lexical | wc (dlex) |
|---|---|---|---|---|---|---|
| Gemma-2-9B | 9B | multiling. | 0.94 | 0.88 | 0.52 | 30% |
| Llama-3.1-8B | 8B | English | 0.80 | 0.54 | 0.44 | 14% |
| Dorna2-8B | 8B | Persian | 0.78 | 0.68 | 0.50 | 18% |

Key: on domain_lexical, Dorna (0.50) edges its own base Llama (0.44) — Persian fine-tuning did not
destroy ranking ability; the gap to Gemma is architectural.

## Position bias evidence (Gemma, domain_lexical)

| Evaluation | Hits@1 | Winner-consistency |
|---|---|---|
| Single-pass (fixed order) | 0.491 | — |
| PSC, per-shuffle (fair) | 0.367 | 18% |

Gold-position balance across the 5 slots (55–79 of 330 per slot) rules out a data artifact.

## Parse-error / reliability (full n=330, pre-merge)

| Strategy | Model | Hits@1 | MRR | Errors |
|---|---|---|---|---|
| Random | Dorna2 | 0.433 | 0.652 | 0 |
| | Llama-3.1 | 0.809 | 0.878 | 0 |
| | Gemma-2-9B | 0.861 | 0.916 | 0 |
| Domain | Dorna2 | 0.397 | 0.626 | 0 |
| | Llama-3.1 | 0.727 | 0.832 | 1 |
| | Gemma-2-9B | 0.761 | 0.854 | 0 |
| Dom-Lexical | Dorna2 | 0.294 | 0.545 | 0 |
| | Llama-3.1 | 0.485 | 0.661 | 4 |
| | Gemma-2-9B | 0.491 | 0.670 | 0 |

## Complementarity analysis (this session — domain_lexical, n=330, all 3 merged models)

### Per-model Hits@1 (this rerun; minor noise vs. original 0.594/0.573/0.570)

| Model | Hits@1 |
|---|---|
| Gemma merge | 0.576 |
| Llama merge | 0.573 |
| Dorna merge | 0.570 |

### Oracle bound

| | Hits@1 |
|---|---|
| Best single model (Gemma) | 0.576 |
| **Oracle (any of 3 correct)** | **0.715** |
| Oracle gain | +0.139 |

### Four-cell decomposition (n=330 per pair)

| Pair | Both correct | A only | B only | Both wrong |
|---|---|---|---|---|
| Gemma vs Llama | 152 (46.1%) | 38 (11.5%) | 37 (11.2%) | 103 (31.2%) |
| Gemma vs Dorna | 153 (46.4%) | 37 (11.2%) | 35 (10.6%) | 105 (31.8%) |
| Llama vs Dorna | 162 (49.1%) | 27 (8.2%) | 26 (7.9%) | 115 (34.8%) |

### Pairwise error Jaccard (higher = more redundant failures)

| Pair | Jaccard |
|---|---|
| Gemma vs Llama | 0.579 |
| Gemma vs Dorna | 0.593 |
| **Llama vs Dorna** | **0.685** (highest — shared base architecture/lineage) |

### Confidence-routed ensemble (ACTUALLY ACHIEVED — not the oracle)

Routes each instance to whichever model has the highest PSC winner-consistency (a signal available
without the gold label). Ties broken toward Gemma.

| | Value |
|---|---|
| Routing distribution | Gemma 282, Llama 34, Dorna 14 (of 330) |
| **Realized Hits@1** | **0.618** |
| Gain over best single (Gemma 0.576) | **+0.042** |
| % of oracle gap (0.139) realized | ~30% |
| McNemar p-value (routed vs. Gemma alone) | **0.0001** (significant) |
| Discordant instances | 0 losses, 14 wins (out of 330) |

**This is meaningfully above the ~10% "naive majority voting captures the oracle gain" figure reported
in the ensemble-diversity literature — worth stating as a comparison point.**

## What still needs figures (not yet generated)

- Bar chart: six-method ceiling on domain_lexical (the "wall" chart from the presentation deck)
- Line chart: Gemma merge Hits@1 vs. sample size (0.47@30 → 0.567@180 → 0.594@330), showing
  stabilization
- Bar chart: 3-model fair-baseline → merge, before/after
- New: Venn-style or bar chart of the four-cell decomposition per model pair
- New: oracle vs. best-single vs. routed-ensemble bar chart
