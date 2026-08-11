# Per-instance result files

These are the raw per-instance JSONL files used in the complementarity analysis
(`src/eval/complementarity.py`). The actual files (330 lines each, one per test
instance) live on the project's Google Drive at:

- `gemma_merge_eval330/gemma_merge_domain_lexical_330_checkpoint.jsonl`
- `llama_merge_eval330/llama_merge_domain_lexical_330_checkpoint.jsonl`
- `dorna_merge_eval330/dorna_merge_domain_lexical_330_checkpoint.jsonl`

**Copy them into this folder** (or symlink) to run `complementarity.py` locally.
Schemas differ between files -- see docs/04_engineering_lessons.md for the exact
field-by-field mapping (`id`+`hit1` for Gemma vs. `idx`+`gold_rank` for Llama/Dorna)
before assuming they can be loaded the same way.

Aggregate results (not per-instance) for all other methods/baselines are in
`results/tables/full_comparison.csv`.
