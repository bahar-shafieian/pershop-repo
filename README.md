# Persian Response Ranking

Response ranking for Persian social-commerce conversations using lexical retrieval, dense retrieval, prompting, LoRA fine-tuning, and retrieval-conditioned training.

## Main result

Six independent method families converged near **0.52 Hits@1** on the hardest domain-lexical benchmark. Retrieval-conditioned LoRA fine-tuning broke this ceiling across three model architectures:

| Model | Baseline | Retrieval-conditioned SFT |
|---|---:|---:|
| Gemma-2-9B | 0.52 | **0.594** |
| Llama-3.1-8B | 0.44 | **0.573** |
| Dorna2-8B | 0.50 | **0.570** |

A confidence-routed ensemble reached **0.618 Hits@1**, significantly outperforming the best single model.

## Repository

- [Experiment notebook](notebooks/pershop_experiments.ipynb) — complete experiment notebook
- [Source code](src) — data loading, retrieval, evaluation, and ensemble code
- [Results](results) — aggregate and per-instance results
- [Documentation](docs) — methodology, findings, limitations, and engineering notes
- [Hyperparameters](configs/hyperparameters.yaml) — training configuration

## Reproduce the complementarity analysis

    pip install -r requirements.txt
    python src/eval/complementarity.py --results_dir results/per_instance/

## Data

The benchmark contains three negative-sampling strategies: random, domain, and domain-lexical. Raw conversation files are not distributed here because their redistribution terms require confirmation.

## Thesis

BSc Economics with Data Science, University of Cassino and Southern Lazio.
