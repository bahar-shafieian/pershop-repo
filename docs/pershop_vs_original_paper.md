# This Thesis vs. the Original PerSHOP Paper

**Source:** Mahmoudi, K. & Faili, H. (2024). "PerSHOP -- A Persian dataset for shopping dialogue systems
modeling." arXiv:2401.00811.

## The core finding: no direct comparison is possible, and that's itself worth stating

The original PerSHOP paper does **not** perform response selection or ranking. Its two stated
contributions are: (1) construction and annotation of a Persian shopping-dialogue dataset (~22k
utterances, 1,061 dialogues, crowd-sourced human-to-human), and (2) an NLU baseline on top of it,
covering exactly two tasks — **intent classification** (14 intent labels, e.g. Purchase, Inform,
User-Confirm) and **entity recognition** (36 product-attribute slots, e.g. brand, volume, flavor) —
evaluated with micro/macro/weighted F1 using a DIETClassifier (+ ParsBERT or LaBSE embeddings) and a CRF
entity extractor. Reported results: ~91% micro-F1 (intent), ~93% micro-F1 (entity).

There is no candidate-ranking task, no negative sampling, no Hits@1/MRR/nDCG anywhere in the original
paper.

## Comparison table

| Dimension | Original PerSHOP paper | This thesis |
|---|---|---|
| Task | NLU: intent classification + entity recognition, per-utterance | Response ranking: order 5 candidates so gold is first |
| Unit of prediction | A label (intent) or span (entity) per utterance | A full ordering over 5 candidates per query |
| Output space | 14-class intent taxonomy; 36 slots | A permutation of 5 candidate identifiers |
| Models | DIETClassifier + ParsBERT/LaBSE; CRF | Lexical/encoder baselines; Gemma/Llama/Dorna as listwise rerankers, with/without retrieval-conditioned fine-tuning |
| Metric | Micro/macro/weighted F1 | Hits@1, Recall@3, MRR, nDCG@5, winner-consistency, under PSC |
| Negative sampling | N/A — no ranking task | 3 strategies (random/domain/domain_lexical), central to this thesis |
| Headline result | ~91% F1 (intent), ~93% F1 (entity) | Hits@1 = 0.594 (Gemma merge, domain_lexical, n=330) vs. ~0.52 ceiling |

## How to correctly describe the relationship (use this framing, not "we compared and got different numbers")

This thesis **reuses PerSHOP's underlying dialogue and product data** — the shopping conversations and
the candidate-response structure that can be built from them — but **defines an entirely new task** on
top of that data: response ranking with three negative-sampling strategies, a bias-controlled (PSC)
evaluation protocol, and ranking-specific metrics, none of which exist in the original paper. The
benchmark used throughout this thesis (2,116 query–response pairs) is a **derivative construction**,
not something defined or evaluated by PerSHOP's original authors.

## The one comparison that IS legitimate

Both papers report a strong result on their respective *easiest* sub-task using a broadly similar kind
of technique: the original paper's ~91% intent-classification F1, and this thesis's ~0.86–0.94 Hits@1 on
the `random` negative-sampling strategy. Both reflect settings where the distinguishing signal is easy
to detect from surface features. Neither number says anything about the harder question this thesis
investigates (lexically confusable negatives), which the original paper's task doesn't have a
counterpart for at all — which is itself part of the motivation for building this thesis's benchmark on
top of PerSHOP's raw data in the first place.

## Where this lives in the thesis

Chapter 2, Section 2.8 (see `Chapter2_Background_and_Related_Work.docx`) — placed in Background/Related
Work rather than Results, since the comparison concerns *what is being measured*, not *how well either
study's methods perform*.
