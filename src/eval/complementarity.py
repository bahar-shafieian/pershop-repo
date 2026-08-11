"""
Complementarity analysis across multiple models' per-instance results.

This is the cleaned-up, schema-normalized version of the code actually run in
Colab this session (see notebooks/colab_cells_reference.py for the exact,
unmodified cell history including the schema-mismatch debugging).

Computes:
  1. Oracle upper bound (ceiling -- NOT an achieved result, see docs/03_complementarity_analysis.md)
  2. Four-cell decomposition per model pair
  3. Pairwise error-Jaccard
  4. A real, testable confidence-routed ensemble (an ACTUALLY ACHIEVED result)
  5. McNemar significance test for the routed ensemble vs. the best single model

Usage:
    python complementarity.py --results_dir results/per_instance/
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def load_gemma_schema(path: str | Path) -> tuple[pd.Series, pd.Series]:
    """Schema: {"id": str, "hit1": 0/1, "winner_consistency": float, ...}"""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    hit1 = df.set_index("id")["hit1"].astype(int)
    wc = df.set_index("id")["winner_consistency"]
    # Re-index positionally (0..n-1, file order) so it can align with idx-based files.
    # Valid ONLY if all files evaluated the same test file in the same, unshuffled order.
    hit1 = hit1.reset_index(drop=True)
    wc = wc.reset_index(drop=True)
    hit1.index.name = "idx"
    wc.index.name = "idx"
    return hit1, wc


def load_idx_gold_rank_schema(path: str | Path) -> tuple[pd.Series, pd.Series]:
    """Schema: {"idx": int, "gold_rank": int, "wc": bool/float, ...}"""
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["hit1"] = (df["gold_rank"] == 1).astype(int)
    hit1 = df.set_index("idx")["hit1"]
    wc = df.set_index("idx")["wc"].astype(float)
    return hit1, wc


def oracle_upper_bound(matrix: pd.DataFrame) -> dict:
    oracle = (matrix.sum(axis=1) > 0).mean()
    best_col = matrix.mean().idxmax()
    best = matrix.mean().max()
    return {
        "oracle_hits1": oracle,
        "best_single_model": best_col,
        "best_single_hits1": best,
        "oracle_gain": oracle - best,
    }


def four_cell_decomposition(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m1, m2 in combinations(matrix.columns, 2):
        a, b = matrix[m1], matrix[m2]
        n = len(a)
        bc = int(((a == 1) & (b == 1)).sum())
        ao = int(((a == 1) & (b == 0)).sum())
        bo = int(((a == 0) & (b == 1)).sum())
        bw = int(((a == 0) & (b == 0)).sum())
        rows.append({
            "pair": f"{m1} vs {m2}", "n": n,
            "both_correct": bc, "both_correct_pct": round(100 * bc / n, 1),
            f"{m1}_only": ao, f"{m1}_only_pct": round(100 * ao / n, 1),
            f"{m2}_only": bo, f"{m2}_only_pct": round(100 * bo / n, 1),
            "both_wrong": bw, "both_wrong_pct": round(100 * bw / n, 1),
        })
    return pd.DataFrame(rows).set_index("pair")


def pairwise_error_jaccard(matrix: pd.DataFrame) -> pd.DataFrame:
    methods = list(matrix.columns)
    jac = pd.DataFrame(index=methods, columns=methods, dtype=float)
    for m1 in methods:
        for m2 in methods:
            e1, e2 = (matrix[m1] == 0), (matrix[m2] == 0)
            union = (e1 | e2).sum()
            inter = (e1 & e2).sum()
            jac.loc[m1, m2] = inter / union if union > 0 else float("nan")
    return jac


def confidence_routed_ensemble(
    hit1_matrix: pd.DataFrame,
    wc_matrix: pd.DataFrame,
    tie_break_model: str,
) -> tuple[np.ndarray, pd.Series]:
    """
    Routes each instance to whichever model had the highest winner-consistency
    (a signal available WITHOUT knowledge of the gold label -- no leakage).
    Ties broken toward `tie_break_model`. This decision rule must be specified
    BEFORE looking at outcome-level results (it was, in this project).

    Returns (ensemble_hit1_array, routed_model_per_instance).
    """
    order = list(hit1_matrix.columns)

    def route(row):
        return max(order, key=lambda m: (row[m], m == tie_break_model))

    routed_model = wc_matrix.apply(route, axis=1)
    ensemble_hit1 = np.array(
        [hit1_matrix.loc[i, routed_model.loc[i]] for i in hit1_matrix.index]
    )
    return ensemble_hit1, routed_model


def mcnemar_significance(baseline: np.ndarray, challenger: np.ndarray) -> dict:
    """
    McNemar's test comparing `challenger` (e.g. the routed ensemble) against
    `baseline` (e.g. the best single model) on the SAME paired test instances.
    Requires: pip install statsmodels
    """
    from statsmodels.stats.contingency_tables import mcnemar

    baseline_right_challenger_wrong = int(((baseline == 1) & (challenger == 0)).sum())
    challenger_right_baseline_wrong = int(((baseline == 0) & (challenger == 1)).sum())
    both_right = int(((baseline == 1) & (challenger == 1)).sum())
    both_wrong = int(((baseline == 0) & (challenger == 0)).sum())

    table = [[0, baseline_right_challenger_wrong], [challenger_right_baseline_wrong, 0]]
    n_discordant = baseline_right_challenger_wrong + challenger_right_baseline_wrong
    result = mcnemar(table, exact=(n_discordant < 25), correction=True)

    return {
        "both_correct": both_right,
        "both_wrong": both_wrong,
        "baseline_right_challenger_wrong": baseline_right_challenger_wrong,
        "challenger_right_baseline_wrong": challenger_right_baseline_wrong,
        "p_value": result.pvalue,
        "significant_at_0.05": bool(result.pvalue < 0.05),
    }


def main(results_dir: Path):
    gemma_hit1, gemma_wc = load_gemma_schema(results_dir / "gemma_merge_domain_lexical_330_checkpoint.jsonl")
    llama_hit1, llama_wc = load_idx_gold_rank_schema(results_dir / "llama_merge_domain_lexical_330_checkpoint.jsonl")
    dorna_hit1, dorna_wc = load_idx_gold_rank_schema(results_dir / "dorna_merge_domain_lexical_330_checkpoint.jsonl")

    hit1_matrix = pd.DataFrame({"gemma": gemma_hit1, "llama": llama_hit1, "dorna": dorna_hit1}).dropna()
    wc_matrix = pd.DataFrame({"gemma": gemma_wc, "llama": llama_wc, "dorna": dorna_wc}).dropna()

    print(f"Matched instances: {len(hit1_matrix)}")
    print("Hits@1:", hit1_matrix.mean().round(3).to_dict())

    print("\n--- Oracle bound ---")
    print(oracle_upper_bound(hit1_matrix))

    print("\n--- Four-cell decomposition ---")
    print(four_cell_decomposition(hit1_matrix))

    print("\n--- Pairwise error Jaccard ---")
    print(pairwise_error_jaccard(hit1_matrix).round(3))

    print("\n--- Confidence-routed ensemble ---")
    ensemble_hit1, routed_model = confidence_routed_ensemble(hit1_matrix, wc_matrix, tie_break_model="gemma")
    print("Routing distribution:", routed_model.value_counts().to_dict())
    print("Ensemble Hits@1:", ensemble_hit1.mean().round(3))

    print("\n--- McNemar test (routed ensemble vs. best single model) ---")
    print(mcnemar_significance(hit1_matrix["gemma"].values, ensemble_hit1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.results_dir)
