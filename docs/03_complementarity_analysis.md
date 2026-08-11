# Complementarity Analysis — Full Writeup

## Motivation

The six-method ceiling (domain_lexical Hits@1 ≈ 0.52) and the fact that the three merged models
converge tightly (0.570–0.594) raise a natural question: are these methods/models solving the *same*
instances, so the ceiling reflects a genuine, shared limit — or are they solving *different* instances,
in which case the ceiling is an artifact of using single models rather than a real limit on what's
extractable from the data?

## Method

Standard multi-classifier complementarity analysis, adapted to a ranking task's binary Hits@1 outcome
per instance:

1. **Oracle upper bound** — for each test instance, mark it correct if *any* constituent model got
   Hits@1 = 1. Oracle Hits@1 = fraction of instances where at least one model succeeded. This is the
   ceiling achievable by a hypothetical perfect router, not a realistic target.
2. **Four-cell decomposition** (per model pair) — both correct / A-only / B-only / both wrong. Large
   "only" cells relative to "both correct" indicate genuine complementarity; small ones indicate
   redundancy.
3. **Pairwise error-Jaccard** — |shared errors| / |union of errors| between two models' failure sets.
   High Jaccard (literature comparison point: 0.57–0.64 reported as "high overlap" in comparable LLM
   error-correlation studies) = largely redundant failures.
4. **A realistic combiner** (not just the oracle) — since the oracle is known in the literature to be
   positive almost universally and to say little on its own about whether a real combiner can capture
   it, we built and tested a **confidence-routed ensemble**: route each instance to whichever model had
   the highest PSC winner-consistency (a per-instance stability signal computed during PSC evaluation,
   available without knowledge of the gold label). This is a legitimate, non-leaking combiner — no
   information about correctness is used to make the routing decision.

## Why the oracle number alone is not the finding

A 2026 audit of LLM ensembles (cited in-conversation during this project) found latent complementarity
via the oracle bound was positive in essentially all tested subsets, yet simple majority voting beat the
strongest individual member in only ~10% of cases. The oracle bound tells you the *ceiling* on possible
improvement; it says nothing about whether any *practical* combining strategy reaches it. This is why
the confidence-routed ensemble — an actual, tested, non-leaking combiner — is the number that belongs
in the thesis as an "achieved" result, not the 0.715 oracle figure.

## Results (domain_lexical, n=330, all 3 merged models — this session)

- Oracle Hits@1 = **0.715**, vs. best single model (Gemma) = **0.576** → gain = **+0.139**
- Error Jaccard: Gemma–Llama 0.579, Gemma–Dorna 0.593, **Llama–Dorna 0.685** (highest)
  → Llama and Dorna, which share a base architecture (Dorna is a Persian fine-tune of Llama), have the
  most correlated errors. Gemma, the architecturally distinct model, contributes more unique correct
  answers relative to either Llama-family model than they contribute to each other. This is consistent
  with literature findings that models sharing lineage/architecture have more correlated errors even at
  similar accuracy levels.
- Confidence-routed ensemble: **Hits@1 = 0.618**, McNemar p = 0.0001 (0 losses / 14 wins vs. Gemma
  alone out of 330 instances) — **statistically significant, actually achieved**, realizing ~30% of the
  oracle gap.

## How to state this in the thesis (Results + Discussion)

**Results chapter (factual):**
> "A complementarity analysis across the three merged models (domain_lexical, n=330) found an oracle
> upper bound of 0.715 — a +0.139 gap over the best single model (Gemma, 0.576) — indicating the models'
> errors are only partially correlated (pairwise Jaccard 0.579–0.685) rather than fully redundant. A
> confidence-routed ensemble, selecting per-instance among the three models by PSC winner-consistency,
> achieved Hits@1 = 0.618, a statistically significant gain over the best single model (McNemar
> p = 0.0001), realizing approximately 30% of the theoretical oracle gap."

**Discussion chapter (interpretive, with the honest caveat):**
> "This finding qualifies the method-invariant ceiling reported earlier: the ceiling appears to hold at
> the level of any single model or method family, but is not an absolute limit on what is extractable
> from the benchmark — a modest, principled combination strategy recovers real additional performance.
> The gap between the oracle bound (0.715) and the realized ensemble (0.618) should be read as
> unexploited headroom rather than a failure: consistent with reported patterns in the ensemble-diversity
> literature, where naive combiners typically capture only a small fraction of oracle gain, our
> confidence-routed approach captures a comparatively larger share (~30%), but a majority of the
// theoretical ceiling remains unrealized. Closing more of this gap — e.g., via a learned router rather
> than a hand-specified winner-consistency threshold — is left as future work."

## Caveats to state explicitly, always

- The 0.715 oracle figure is a **ceiling**, never an achieved result. Never state it as "we obtained
  0.715."
- The routing tie-break (favoring Gemma when winner-consistency is equal) was specified *before* seeing
  outcome-level results, and should be described that way in the methodology — not discovered post hoc.
- 14 discordant instances is a small absolute number; the McNemar significance is real but the practical
  effect size is modest. Don't oversell this as solving the ceiling — it's a real, incremental, honestly
  earned improvement.
- This analysis used only the three post-merge models. It was not run across method families (lexical
  vs. encoder vs. LLM), which would likely show different (probably lower) redundancy given how
  architecturally different those baselines are from the LLMs — flagged as a natural extension in
  `docs/05_open_questions.md`.
