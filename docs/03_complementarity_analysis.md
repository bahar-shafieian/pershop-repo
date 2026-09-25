# Exploratory Complementarity Analysis

**Status: provisional.** This analysis explores whether the three retrieval-conditioned LLMs make different errors on the same domain-lexical test queries. It should not yet be presented as a validated ensemble improvement.

## Available outputs

The three saved per-instance files in [`results/per_instance/`](../results/per_instance) contain 330 rows each. The Gemma file is a later rerun and scores **0.576 Hits@1**, while the earlier full-test Gemma result in the main comparison table is **0.594**. Llama and Dorna score **0.573** and **0.570**, respectively. The analysis in [`src/eval/complementarity.py`](../src/eval/complementarity.py) reports the following *conditional on its row alignment*:

| Quantity | Value | Interpretation |
| --- | ---: | --- |
| Best single output (Gemma rerun) | 0.576 | Observed in these saved files |
| Oracle upper bound | 0.715 | Hypothetical router that knows which model is right |
| Confidence-routed output | 0.618 | Exploratory result from the current routing script |

The oracle is an upper bound, not an achieved model score. The pairwise error Jaccard values produced by the script are 0.579 (Gemma–Llama), 0.593 (Gemma–Dorna), and 0.685 (Llama–Dorna), also conditional on correct row alignment.

## Why the routing result is provisional

1. **Instance matching needs verification.** Gemma rows have string IDs, while Llama and Dorna rows use integer positions (`idx`). The script aligns Gemma by row order. Matching totals and per-model accuracy are insufficient to prove that row 17 in all three files refers to the same query. Oracle, overlap, routing, and paired significance calculations require that guarantee.
2. **The routing signals use different scales.** Gemma records fractional `winner_consistency` values; Llama and Dorna record a boolean `wc`. Selecting the maximum across these encodings does not compare the same calibrated per-instance quantity. The result may change when consistency is recomputed under a shared definition.
3. **The routing rule needs independent evaluation.** Once the first two checks are resolved, choose or confirm the rule on validation data and evaluate it on a held-out test set. Any paired significance result from the current script is conditional on the unresolved matching and signal issues.

Run the current exploratory calculation with:

```bash
pip install -r requirements.txt
python src/eval/complementarity.py --results_dir results/per_instance/
```

The next step is to attach the canonical benchmark ID to every model output and save the five permutation-level top choices for each model. That will permit an ID-based join and a shared winner-consistency definition before reporting an ensemble result.
