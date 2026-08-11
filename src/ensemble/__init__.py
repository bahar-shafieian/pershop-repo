"""
Confidence-routed ensemble across the three merged models (Gemma, Llama, Dorna).

See docs/03_complementarity_analysis.md for the full writeup, and
src/eval/complementarity.py for the implementation actually used
(confidence_routed_ensemble + mcnemar_significance). This module is a thin
re-export so the ensemble logic can be imported from a natural location.
"""
from src.eval.complementarity import confidence_routed_ensemble, mcnemar_significance  # noqa: F401
