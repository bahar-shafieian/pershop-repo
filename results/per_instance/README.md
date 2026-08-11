# Per-instance result files

These are the raw per-instance JSONL files used in the complementarity analysis
(`src/eval/complementarity.py`). Each file contains 330 records, one per test
instance:

- `gemma_merge_domain_lexical_330_checkpoint.jsonl`
- `llama_merge_domain_lexical_330_checkpoint.jsonl`
- `dorna_merge_domain_lexical_330_checkpoint.jsonl`

The files are included directly in this folder, so the analysis can be run locally:

```bash
python src/eval/complementarity.py --results_dir results/per_instance/
```

Schemas differ between files—see `docs/04_engineering_lessons.md` for the exact
field-by-field mapping (`id` + `hit1` for Gemma versus `idx` + `gold_rank`
for Llama/Dorna) before assuming they can be loaded identically.

Aggregate results for all other methods and baselines are in
`results/tables/full_comparison.csv`.
