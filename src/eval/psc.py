"""
Permutation Self-Consistency (PSC) evaluation protocol.

Tang et al. -- shuffle candidate order k times, rank each shuffle independently,
map back to original identities, aggregate by Borda count. Winner-consistency is
the fraction of the k shuffles whose top pick agrees -- a position-bias diagnostic,
not a ranking-quality metric.

This module is model-agnostic: `rank_fn` is any callable that takes
(query, candidates_in_some_order) -> list[int] (a permutation of range(len(candidates)),
best first), e.g. a wrapper around an LLM generation call. See
notebooks/colab_cells_reference.py for the concrete Gemma/LoRA implementation used
in this thesis.
"""
import re
from collections import Counter
from typing import Callable, Optional

import numpy as np


def parse_ranking(text: str, n: int = 5) -> Optional[list[int]]:
    """
    Extract a ranking from raw model output text of the form '[k] > [k] > ...'.
    Returns a 0-indexed permutation (best first), or None if parsing fails
    (fewer than n distinct valid identifiers found).
    """
    nums = re.findall(r"\[(\d)\]", text)
    nums = [int(x) for x in nums if 1 <= int(x) <= n]
    seen = []
    for x in nums:
        if x not in seen:
            seen.append(x)
    if len(seen) == n:
        return [x - 1 for x in seen]
    return None


def psc_evaluate_instance(
    query: str,
    candidates: list[dict],
    gold_idx: int,
    rank_fn: Callable[[str, list[dict], list[int]], Optional[list[int]]],
    k: int = 5,
    seed: int = 42,
) -> dict:
    """
    Run PSC evaluation on a single instance.

    rank_fn(query, candidates, order) -> ranking or None
        `order` is the shuffled presentation order (list of original indices);
        rank_fn is responsible for presenting candidates in that order and
        returning the model's ranking as a permutation of *shuffled positions*
        (0-indexed, best first) -- this function handles mapping back to
        original candidate indices.

    Returns a dict matching the schema used in results/per_instance/*.jsonl:
        {id-independent fields only; caller should attach "id"}
        hit1, gold_rank, borda_scores, parse_errors, winner_consistency
    """
    n = len(candidates)
    rng = np.random.default_rng(seed)
    borda_scores = [0] * n
    top_picks = []
    parse_errors = 0

    for i in range(k):
        order = rng.permutation(n).tolist()
        shuffled_ranking = rank_fn(query, candidates, order)  # positions in shuffled space

        if shuffled_ranking is None:
            parse_errors += 1
            for j in range(n):
                borda_scores[j] += n  # uniform max-rank penalty
            continue

        original_ranking = [order[p] for p in shuffled_ranking]
        for rank, cand_idx in enumerate(original_ranking):
            borda_scores[cand_idx] += rank
        top_picks.append(original_ranking[0])

    final_ranking = sorted(range(n), key=lambda j: borda_scores[j])
    hit1 = int(final_ranking[0] == gold_idx)
    gold_rank = final_ranking.index(gold_idx) + 1

    if top_picks:
        most_common_count = Counter(top_picks).most_common(1)[0][1]
        wc = most_common_count / len(top_picks)
    else:
        wc = 0.0

    return {
        "hit1": hit1,
        "gold_rank": gold_rank,
        "borda_scores": borda_scores,
        "parse_errors": parse_errors,
        "winner_consistency": round(wc, 3),
    }
