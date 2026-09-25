# Limitations and Next Steps

## Evaluation

- Verify that all per-instance outputs refer to the same test queries by adding canonical benchmark IDs. Recompute candidate-order consistency from comparable permutation-level outputs before making claims about a three-model ensemble. See [the exploratory analysis](03_complementarity_analysis.md).
- Repeat the prompted, SFT-only, and retrieval-in-context comparisons currently reported on 50-query subsets across the full test set when compute permits.
- Document complete inference and training artifacts for selected experiments if the work is extended beyond the thesis.

## Modeling

- Investigate whether a learned router improves over a single model *after* the ensemble inputs have been aligned and evaluated fairly.
- Compare models on additional Persian shopping datasets or domains to test how broadly the observed hard-negative difficulty generalizes.
- Preference optimization (including SimPO) was considered but **not completed**. It is future work, not a reported result.

## Dataset and reuse

The [benchmark is published](https://huggingface.co/datasets/blueharu/persian-response-ranking) with train, validation, and test splits and a permission and citation notice. The GitHub repository contains research code and outputs, but does not contain trained model adapters or a single command to reproduce every GPU run.
