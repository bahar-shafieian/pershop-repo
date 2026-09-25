# Project Overview

## Research question

Given a Persian customer query and five candidate seller replies, can a ranking model place the correct reply first when the four alternatives are closely related? The main difficulty setting draws incorrect replies from the same product domain with high lexical overlap.

## Benchmark and provenance

The [public response-ranking benchmark](https://huggingface.co/datasets/blueharu/persian-response-ranking) contains **2,116 query–gold-response pairs**, with three negative-sampling configurations and the same dialogue-level splits for each: 1,483 training, 303 validation, and 330 test instances. Each instance has one correct response among five candidates. The configurations are random, same-domain, and domain-lexical (the new hard-negative construction).

This is a *derived ranking benchmark* built from [PerSHOP](https://arxiv.org/abs/2401.00811) conversations. The original PerSHOP paper studies intent classification and entity recognition, not response ranking; its F1 scores cannot be compared directly with this project's ranking scores. See [the task comparison](pershop_vs_original_paper.md) and the dataset card for provenance and permissions.

## Approach

- Establish lexical (TF-IDF, BM25) and dense embedding baselines.
- Test listwise LLM ranking and diagnose sensitivity to the order of candidate replies. For the later LLM experiments, evaluate five candidate permutations and aggregate their predicted rankings by Borda count.
- Compare prompting, retrieval of solved examples, and retrieval-conditioned LoRA fine-tuning on Gemma-2-9B, Llama-3.1-8B, and Dorna2-8B.
- Measure Hits@1 (primary), Recall@3, MRR, and nDCG@5. See [methodology](01_methodology.md) and [complete results](02_results.md).

## Selected finding

On the **330-query domain-lexical test set**, TF-IDF and BGE-m3 score 0.521 and 0.518 Hits@1, respectively. Retrieval-conditioned Gemma scores **0.594** in the reported full-test run; Llama and Dorna score 0.573 and 0.570. A later Gemma per-instance rerun scores 0.576. Several other method comparisons use 50-query subsets, so compare rows only after checking sample size and protocol.

The [three-model complementarity analysis](03_complementarity_analysis.md) remains exploratory: the saved outputs must be joined by verified instance IDs and their different winner-consistency encodings reconciled before its ensemble score can be presented as a validated result.

## Scope and limitations

- The public dataset allows readers to inspect the ranking task and splits. GPU training code and outputs are provided as research artifacts; the repository does not include trained adapters or a one-command recreation of every experiment.
- Five-candidate ranking is an offline benchmark, not a deployed customer-support system.
- Position sensitivity motivated the evaluation protocol, but this does not imply that model responses are invariant to candidate order.
