"""
Prompt templates for the lean prompt and RAG-augmented (merge) methods.
See docs/01_methodology.md, sections 3.6.1 and 3.6.5, for the design rationale
and the history of failed variants (heavy prompt -> output collapse; fixed by
going lean and dropping the vocabulary glossary).
"""

SYSTEM_INSTRUCTION_EN = (
    "You are a response ranking assistant for a Persian social-commerce chat system. "
    "Your task: given a customer query and 5 candidate seller responses numbered [1]-[5], "
    "output a ranking from best to worst as: [X] > [X] > [X] > [X] > [X]\n"
    "Rule: a response that directly provides the exact product beats one that asks a "
    "clarifying question first.\n"
    "Output the ranking line only. No explanation."
)

# Dorna receives the same structure in Persian -- language-per-model fairness rule
# (English helps Gemma/Llama, Persian helps Dorna). Fill in with a professional
# translation matching the English instruction's content exactly if used.
SYSTEM_INSTRUCTION_FA = None  # populate before use with Dorna


def format_candidates(candidates: list[dict], order: list[int]) -> str:
    """Format candidates in a given presentation order (list of 0-based indices)."""
    lines = []
    for rank, idx in enumerate(order, 1):
        lines.append(f"[{rank}] {candidates[idx]['text']}")
    return "\n".join(lines)


def build_lean_prompt(query: str, candidates: list[dict], order: list[int],
                       system: str = SYSTEM_INSTRUCTION_EN) -> str:
    """The lean prompt, no few-shot examples, no glossary. Used for the frozen
    zero-shot and lean-prompt-only baselines."""
    cand_text = format_candidates(candidates, order)
    return (
        f"{system}\n\n"
        f"### New instance (rank this one)\n"
        f"Customer query: {query}\n"
        f"Candidates:\n{cand_text}\n"
        f"Ranking:"
    )


def build_rag_augmented_prompt(query: str, candidates: list[dict], order: list[int],
                                neighbours: list[dict],
                                system: str = SYSTEM_INSTRUCTION_EN) -> str:
    """
    Lean prompt + retrieved neighbours as worked examples -- used for both
    RAG-ICL (frozen model) and the merge (trained to condition on this format).
    `neighbours` is a list of already-solved training instances (with their own
    `candidates` list, one label==1 gold) retrieved via BGE-m3 cosine-kNN.

    IMPORTANT: at inference, retrieval must be present if the model was trained
    with retrieval (the merge) -- evaluating a retrieval-conditioned adapter on
    bare prompts tests it out-of-distribution and understates performance.
    """
    few_shot = ""
    for nb in neighbours:
        nb_cands = nb["candidates"]
        gold_idx = next(i for i, c in enumerate(nb_cands) if c["label"] == 1)
        nb_order = list(range(len(nb_cands)))  # original order for the worked example
        cand_text = format_candidates(nb_cands, nb_order)
        gold_rank = nb_order.index(gold_idx) + 1
        remaining = [str(r + 1) for r in range(len(nb_cands)) if r + 1 != gold_rank]
        ranking_str = f"[{gold_rank}] > " + " > ".join(f"[{r}]" for r in remaining)
        few_shot += (
            f"\n\n### Example\n"
            f"Customer query: {nb['query']}\n"
            f"Candidates:\n{cand_text}\n"
            f"Ranking: {ranking_str}"
        )

    cand_text = format_candidates(candidates, order)
    return (
        f"{system}"
        f"{few_shot}"
        f"\n\n### New instance (rank this one)\n"
        f"Customer query: {query}\n"
        f"Candidates:\n{cand_text}\n"
        f"Ranking:"
    )
