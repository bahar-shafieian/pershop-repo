# Response Ranking for Persian Social-Commerce Conversations

Thesis repository — Economics with Data Science, University of Cassino and Southern Lazio (UniCS)
**Author:** Bahar Shafieian · **Supervisors:** Prof. Ciro Russo, Prof. Mario Molinara

A comparative study of lexical, semantic, and large language model approaches to listwise response
ranking on the PerSHOP Persian social-commerce dataset, with a focus on evaluation bias (position bias /
Permutation Self-Consistency), a method-invariant performance ceiling, and a retrieval-conditioned
fine-tuning method that breaks it across three model architectures.

> **Honest framing.** Every technique used here — PSC, retrieval-augmented in-context learning, LoRA
> fine-tuning, RA-DIT-style retrieval-conditioned training — is established in the literature. The
> contribution of this work is the systematic, bias-controlled application and comparison of these
> techniques on a low-resource ranking task, the empirical demonstration of a method-invariant ceiling,
> and the demonstration that retrieval-conditioned fine-tuning breaks that ceiling across architectures.
> No new algorithm is claimed.

---

## Repository contents

```
pershop-repo/
├── README.md                        <- you are here
├── docs/
│   ├── 00_project_overview.md       <- full narrative: task, findings, timeline
│   ├── 01_methodology.md            <- methods, in prose (mirrors thesis Ch.3)
│   ├── 02_results.md                <- all results tables, in one place
│   ├── 03_complementarity_analysis.md  <- oracle bound / routed ensemble writeup
│   ├── 04_engineering_lessons.md    <- Colab/T4/Drive gotchas, so you never re-learn them
│   ├── 05_open_questions.md         <- what's unfinished, what's future work
│   └── pershop_vs_original_paper.md <- the Ch.2 comparison, standalone
├── src/
│   ├── data/                        <- dataset loading, PerSHOP schema helpers
│   ├── eval/                        <- PSC, metrics, complementarity analysis
│   ├── methods/                     <- prompt templates, LoRA configs, retrieval
│   └── ensemble/                    <- the confidence-routed ensemble (Section 2.x finding)
├── notebooks/
│   └── colab_cells_reference.py     <- every Colab cell used, in order, as a flat reference
├── results/
│   ├── per_instance/                <- the three merge JSONL files (schema-normalized copies)
│   ├── tables/                      <- CSV exports of every results table
│   └── figures/                     <- (placeholder — see docs/02_results.md for what's needed)
└── configs/
    └── hyperparameters.yaml         <- every LoRA/training config used, in one file
```

## Quickstart

This repo is primarily a **record and reference**, not a one-command pipeline — the actual experiments
ran across many Colab sessions with manual checkpointing. To reproduce or extend:

1. Read `docs/00_project_overview.md` first — it's the single-file summary of everything.
2. Dataset schema and loading: `src/data/pershop_loader.py`
3. Evaluation protocol (PSC): `src/eval/psc.py`
4. To rerun the complementarity analysis on saved per-instance results:
   `python src/eval/complementarity.py --results_dir results/per_instance/`
5. Full Colab cell history, in the order actually run: `notebooks/colab_cells_reference.py`

## Current status (see `docs/05_open_questions.md` for detail)

- ✅ Experimental work complete: 6 baseline/method families + retrieval-conditioned merge, on 3 LLMs
- ✅ Bias-controlled evaluation (PSC) applied throughout
- ✅ Complementarity analysis complete: oracle bound + confidence-routed ensemble, statistically confirmed
- ✅ Thesis Chapter 2 (Background & Related Work) and Chapter 3 (Methodology) drafted
- ⬜ Thesis Chapters 1, 4 (partial — folded into Ch.3), 5, 6, 7 not yet drafted
- ⬜ Complementarity/ensemble finding not yet written into any thesis chapter
- ⬜ SimPO preference-optimization stage designed, never run (abandoned / future work)
