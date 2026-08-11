# Project Overview

## The task

Given a Persian customer message from a social-commerce conversation and five candidate seller
responses (one correct/gold, four negatives), produce a full ranking of the five so the gold response
is ranked first. Listwise formulation. A domain rule breaks ties: a response that directly gives the
requested product beats one that asks a clarifying question first, even if that question mentions the
same product.

## The benchmark (PerSHOP, reframed)

- Source: PerSHOP (Mahmoudi & Faili, 2024) — a Persian shopping-dialogue dataset. **Important:** the
  original paper does NLU (intent classification + entity recognition) only. It does not define any
  response-ranking task. The ranking benchmark used in this thesis is a derivative construction built
  on top of PerSHOP's raw dialogue/product data — see `docs/pershop_vs_original_paper.md`.
- Working benchmark: 2,116 query–response pairs. Per strategy: 1,483 train / 330 test.
- Three negative-sampling strategies, increasing difficulty:
  - `random` — unrelated negatives (easy)
  - `domain` — same product domain (medium)
  - `domain_lexical` — same domain, lexically near-identical to gold (hardest — the focus of the study)
- File format: JSONL, `pershop_{strategy}_{train|test}.jsonl`, one record per line:
  `{"id", "query", "candidates": [{"text","label"}, ...], "negative_strategy", "split"}`. Gold candidate
  has `label == 1`.

## Metrics

Hits@1 (primary), Recall@3, MRR, nDCG@5. Winner-consistency (wc) is a secondary diagnostic — fraction
of PSC shuffles where the model's top pick stays the same; not a quality metric, a stability metric.
**No comparison to the original PerSHOP paper's metric is possible or claimed** — see
`docs/pershop_vs_original_paper.md`.

## The central empirical findings, in order of discovery

### 1. Position bias is severe, and single-pass LLM ranking is unreliable
Gemma zero-shot on domain_lexical: single-pass Hits@1 = 0.491, but under PSC (5 shuffles, Borda
aggregation) this drops to 0.367, with only 18% winner-consistency — the model's top pick changed in
82% of cases purely from reshuffling the same 5 candidates. Gold-position balance in the raw data
(55–79 per slot across 5 slots) rules out a data artifact. **Consequence: every LLM result in this
thesis uses PSC (k=5, Borda) from this point on.**

### 2. A method-invariant ceiling
Six independent method families converge at Hits@1 ≈ 0.52 on domain_lexical:

| Method | domain_lexical Hits@1 | n |
|---|---|---|
| TF-IDF | 0.521 | 330 |
| BM25 | 0.521 | 330 |
| BGE-m3 (best encoder) | 0.518 | 330 |
| Lean prompt + PSC (Gemma) | 0.52 | 50 |
| SFT-only (self-distilled curriculum LoRA) | 0.48 | 50 |
| RAG-ICL (frozen, k=3 retrieval) | 0.52 | 50 |

This convergence across lexical, dense-retrieval, prompting, fine-tuning, and retrieval approaches is
treated as a genuine empirical finding, not a weak-method artifact.

### 3. The merge (RAG-augmented SFT) breaks the ceiling
Training the model to condition on retrieved same-strategy neighbours (RA-DIT-style), rather than
merely prompting with them, is the only method that exceeds 0.52:

| Model (merge) | random | domain | domain_lexical | n (dlex) |
|---|---|---|---|---|
| Gemma-2-9B | 0.94 | 0.827 | **0.594** | 330 |
| Llama-3.1-8B | 0.879 | 0.809 | **0.573** | 330 |
| Dorna2-8B | 0.903 | 0.815 | **0.570** | 330 |

The gain rose and stabilized as the test sample grew (Gemma: 0.47 @n=30 → 0.567 @n=180 → 0.594
@n=330), evidence it's real rather than small-sample noise. Cost: Gemma's `domain` score fell
0.88 → 0.827 (real but small). Winner-consistency stayed low post-merge (35.5–40%) — the merge improves
*content judgment*, not position robustness.

### 4. It generalizes across architecture
The identical merge recipe was applied to Gemma (multilingual), Llama (English-primary), and Dorna
(Persian fine-tune of Llama) — all three clear the ceiling, each +0.07 to +0.13 over its own
fair-conditions baseline. Bonus: the merge also repaired Dorna's instruction-following (clean 5/5
output format) where a training-free repair (Chat Vector) had failed at every interpolation strength.

### 5. NEW — Complementarity across the three merged models (this session)
Do the three merged models fail on the *same* instances, or different ones? Computed directly from
saved per-instance results on domain_lexical (n=330, all three models):

- **Oracle upper bound** (any of the 3 correct): **0.715**, vs. best single model (Gemma) **0.576**
  → oracle gain **+0.139**.
- **Pairwise error Jaccard** (higher = more redundant failures): Gemma–Llama 0.579, Gemma–Dorna 0.593,
  **Llama–Dorna 0.685** (highest — consistent with Llama–Dorna sharing a base architecture/lineage).
- **A real, buildable confidence-routed ensemble** (routes each instance to whichever model had the
  highest PSC winner-consistency, a signal available without seeing the gold label) achieves
  **Hits@1 = 0.618** — a **statistically significant** +0.042 gain over Gemma alone
  (McNemar p = 0.0001; 0 losses, 14 wins out of 330 discordant instances). This realizes ~30% of the
  theoretical oracle gap — well above the ~10% "naive combiner captures the oracle gain" figure
  reported in the literature for majority-vote ensembles.
- **Caveat, stated plainly:** the oracle number (0.715) is a ceiling, not an achieved result — only the
  confidence-routed ensemble's 0.618 is an actually-realized, tested outcome. See
  `docs/03_complementarity_analysis.md` for full detail and the reasoning literature behind this
  distinction.

## The Persian-model question (answered)

Supervisor question: why does a Persian-specialized model (Dorna) underperform a multilingual model
(Gemma) on a Persian task? Evidence assembled across the study:
- Under fair, identical-protocol evaluation, Dorna (0.50) actually **edges its own base model**
  Llama (0.44) on domain_lexical — Persian adaptation did not destroy ranking ability.
- Chat Vector (training-free instruction-following repair) failed at every strength tested
  (λ=1.0 → 0/5 valid; λ=0.5/0.3 → 3/5 valid) — the limitation isn't simple weight-space drift.
- The merge lifts Dorna to 0.570, level with Gemma (0.594) and Llama (0.573), and separately repairs
  its output-format reliability to 100%.
- **Conclusion:** the Gemma–Dorna gap is architectural (base-model capability), not damage from
  Persian adaptation. The right training signal (retrieval-conditioned fine-tuning) closes it.

## Honest limitations (state these explicitly, everywhere)

- Most non-merge baselines are n=50 subsets; only merge + encoder/lexical baselines are full n=330.
  Say "converges around 0.52," never "exactly 0.52."
- Only Gemma's zero-shot was PSC-scored (0.367). Llama/Dorna zero-shot numbers are single-pass and
  were never reshuffled — never compare them directly to Gemma's fair zero-shot number.
- No comparison to the original PerSHOP paper's own metric is possible — different task entirely
  (NLU vs. ranking). See `docs/pershop_vs_original_paper.md`.
- SimPO preference-optimization stage was designed but never run (Colab preemption killed both DPO
  attempts; SimPO was the planned reference-free replacement) — remains future work.
- Winner-consistency stays modest even after the merge (35.5–40%) — position sensitivity is reduced,
  not solved.
- The oracle upper bound (0.715) is a ceiling, not an achieved result — see above.

## Timeline / how we got here

1. Zero-shot baselines on 3 LLMs → discovered position bias problem (single-pass vs PSC).
2. Lean prompt design (fixed a prompt-collapse bug from an earlier heavy-prompt version).
3. Chain-of-Prompting — negative result, no gain over lean prompt at 4–5x cost.
4. Self-distilled curriculum LoRA (SFT-only) — negative result on domain_lexical (exposure bias).
5. RAG-ICL (frozen model + retrieval) — stabilizes (raises wc) but doesn't raise Hits@1.
6. Six-method ceiling identified (~0.52 on domain_lexical, all families).
7. The merge (RAG-augmented SFT) — breaks the ceiling on Gemma, then confirmed on Llama and Dorna.
8. Chat Vector attempted on Dorna as a cheaper alternative repair — negative result, motivates the merge
   as the real fix.
9. Fair-conditions cross-model baseline — answers the "why is Persian model worse" question.
10. **This session:** re-ran the missing Gemma merge per-instance eval; built the complementarity
    analysis (oracle bound, four-cell decomposition, error-Jaccard); built and statistically validated
    a confidence-routed ensemble.
