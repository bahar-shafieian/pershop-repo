# Persian Response Ranking for Social Commerce

**Bachelor's thesis · NLP · retrieval and reranking · Persian**

How can a system choose the right seller reply when several responses sound plausible? I studied this as a **five-candidate response-ranking task**: given a customer's Persian message, rank one correct seller response above four alternatives. The most difficult alternatives come from the same shopping domain and share words with the query, so surface similarity alone is often misleading.

This project turns [PerSHOP](https://arxiv.org/abs/2401.00811), an existing Persian shopping-dialogue dataset, into a response-ranking benchmark and compares lexical retrieval, multilingual embeddings, prompted LLMs, and retrieval-conditioned fine-tuning. **The original PerSHOP paper studied intent and entity recognition; response ranking and the domain-lexical configuration are contributions of this thesis.**

**[Explore the public benchmark on Hugging Face](https://huggingface.co/datasets/blueharu/persian-response-ranking)** · **[Read the methodology](docs/01_methodology.md)** · **[Inspect the results](docs/02_results.md)**

## What I built

- Constructed a benchmark of **2,116 customer-query / seller-response pairs** with five candidates per query. The same pairs and dialogue-level splits are used across three difficulty settings: **1,483 train / 303 validation / 330 test** per setting.
- Added **domain-lexical hard negatives**: incorrect replies from the same product domain with high word overlap. The public benchmark also includes random and same-domain configurations.
- Compared TF-IDF, BM25, sentence embeddings, listwise LLM prompting, and LoRA fine-tuning. Evaluated ranking with Hits@1 as the primary metric, alongside Recall@3, MRR, and nDCG@5.
- Investigated sensitivity to candidate order using **Permutation Self-Consistency**: shuffle candidates five times, map predictions back to the original candidates, and aggregate their ranks.
- Trained retrieval-conditioned LoRA adapters on solved neighboring examples for **Gemma-2-9B, Llama-3.1-8B, and Dorna2-8B**.

## Selected results

**Hardest setting: domain-lexical negatives, test set of 330 queries.** Hits@1 is the fraction of queries for which the correct response is ranked first; higher is better.

| Method | Hits@1 | Evaluation |
| --- | ---: | --- |
| TF-IDF | 0.521 | Full 330-query test set |
| BGE-m3 sentence embeddings | 0.518 | Full 330-query test set |
| Gemma-2-9B with retrieval-conditioned LoRA | **0.594** | Reported full 330-query run |
| Llama-3.1-8B with retrieval-conditioned LoRA | 0.573 | Full 330-query test set |
| Dorna2-8B with retrieval-conditioned LoRA | 0.570 | Full 330-query test set |

The harder negatives substantially reduced the effectiveness of lexical and embedding baselines. Retrieval-conditioned fine-tuning improved top-choice accuracy in the reported full-test runs. Several prompted, SFT-only, and retrieval-in-context comparisons in the [complete table](results/tables/full_comparison.csv) used **50-query subsets**; check each row's sample size before comparing scores.

**Run-to-run and exploratory analysis:** A later Gemma per-instance rerun scored **0.576**, compared with **0.594** in the earlier reported run. The [complementarity analysis](docs/03_complementarity_analysis.md) explores combining that rerun with Llama and Dorna, but its positional instance alignment and differently encoded winner-consistency fields need further validation. I do not use its ensemble score as a headline result here.

## Data and repository guide

The [published dataset](https://huggingface.co/datasets/blueharu/persian-response-ranking) includes the three configurations and their train, validation, and test splits. Redistribution permission and citation terms are documented on the dataset card. You can inspect an example with:

```python
# pip install datasets
from datasets import load_dataset

dataset = load_dataset("blueharu/persian-response-ranking", "domain_lexical")
print(dataset["test"][0])
```

| Location | Contents |
| --- | --- |
| [`docs/01_methodology.md`](docs/01_methodology.md) | Models, evaluation protocol, and training approach |
| [`docs/02_results.md`](docs/02_results.md) | Full experiment table and sample sizes |
| [`notebooks/pershop_experiments.ipynb`](notebooks/pershop_experiments.ipynb) | Experiment notebook |
| [`src/`](src) and [`configs/`](configs) | Data loading, retrieval/evaluation utilities, and recorded hyperparameters |
| [`results/`](results) | Aggregate tables and saved per-instance outputs for the exploratory analysis |

**Reproducibility scope:** The benchmark is public. This repository contains experiment code and recorded outputs, but it does not package the trained LLM adapters or a one-command pipeline to recreate every GPU experiment. The original PerSHOP authors' intent/entity results measure a different task and are not directly comparable to these ranking scores.

## About

BSc thesis in Economics with Data Science, **University of Cassino and Southern Lazio**. Built by **Bahar Shafieian**. The [dataset card](https://huggingface.co/datasets/blueharu/persian-response-ranking) gives the requested citation for this derivative benchmark and the original PerSHOP paper.
