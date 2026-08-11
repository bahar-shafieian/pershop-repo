# Open Questions / Unfinished Work

## Thesis writing status

| Chapter | Status |
|---|---|
| 1 — Introduction | Not drafted |
| 2 — Background & Related Work | **Drafted** (2.1–2.7 + 2.8 PerSHOP comparison) — delivered as docx |
| 3 — Methodology | **Drafted** — delivered as docx (covers what would otherwise be a separate "Experimental Setup" chapter too) |
| 4 — (folded into Ch.3 in current plan) | N/A unless split out later |
| 5 — Results | Not drafted — all numbers exist in `docs/02_results.md`, need prose + tables in thesis format |
| 6 — Discussion | Not drafted — key discussion points exist in `docs/00_project_overview.md` and `docs/03_complementarity_analysis.md`, need prose |
| 7 — Conclusion | Not drafted |
| Complementarity/ensemble finding | Analysis complete and validated; **not yet written into any chapter** — see `docs/03_complementarity_analysis.md` for ready-to-adapt paragraphs |

## Experimental work still open

- **SimPO preference-optimization stage**: designed (reference-free, validated on Gemma-2 in the
  literature), chosen over DPO specifically because DPO needs a reference model (≈2x forward passes) and
  hit a `/tmp` precompute-cache bug. Both DPO attempts died to Colab preemption before completion. SimPO
  itself was never run. Legitimate future-work item, not a result.
- **Complementarity analysis was only run across the 3 post-merge LLMs.** Running the same analysis
  across method *families* (e.g., BGE-m3 vs. lean-prompt vs. merge) would likely show different, probably
  lower, error-Jaccard given how architecturally different those approaches are — this could reveal an
  even larger oracle gap and is a natural, low-cost extension (all needed per-instance data may not yet
  exist for the non-merge baselines — check before assuming this is free).
- **The confidence-routed ensemble uses a hand-specified rule** (route to highest winner-consistency,
  tie-break to Gemma). A learned router (even a simple logistic regression over per-model
  winner-consistency + agreement features) might close more of the oracle gap — untested.
- **Rerunning key n=50 comparisons at full n=330** with significance tests, if pursuing anything beyond
  thesis-complete (e.g., a workshop paper) — reviewers will want this.
- **No comparison to the original PerSHOP paper's own metric** — established in this project that no
  such comparison is meaningful (different task entirely; see `docs/pershop_vs_original_paper.md`), so
  this is closed, not open, but worth remembering it was investigated rather than skipped.

## Presentation deck (pershop_presentation.pptx) — pending fixes noted previously

- Slide 9 (three-LLM comparison): should show fair-baseline→merge (0.52/0.44/0.50 → 0.594/0.573/0.570),
  not zero-shot→merge (mixing Gemma's fair PSC zero-shot with Llama/Dorna's unfair single-pass zero-shot
  is apples-to-oranges). Confirm this was actually applied — last known status was "discussed, not
  confirmed done."
- Slide 7: label sample sizes on the 0.47→0.567→0.594 progression so it's self-explanatory without
  narration.
- Complementarity/ensemble finding (this session) is not yet in the deck at all — would need at least
  one new slide if the defense should cover it.

## Publication framing (if pursued beyond the thesis)

Workshop paper, not top-tier venue — techniques are established, novelty is in application +
systematic comparison + evaluation methodology + the ceiling/generality/complementarity findings. Good
fits: low-resource NLP or multilingual NLP workshops at *ACL/EMNLP, LREC-COLING, or Persian/regional NLP
venues. A supervisor co-author would strengthen a first submission. The complementarity/ensemble result
from this session is a genuinely new angle that could strengthen a paper version beyond what's in the
thesis alone.
