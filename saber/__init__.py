"""
SABER: Scaling-Aware Best-of-N Estimation of Risk

A statistical framework for predicting large-scale adversarial risk in LLMs
under Best-of-N sampling from small-budget measurements.

Quick Start:
    >>> from saber import SABER
    >>> 
    >>> # Fit to your jailbreak data
    >>> model = SABER()
    >>> model.fit(k=successes_per_query, n=trials_per_query)
    >>> 
    >>> # Predict ASR at large N
    >>> result = model.predict(N=1000)
    >>> print(f"ASR@1000 = {result.asr:.2%}")
    >>> 
    >>> # Estimate budget needed for target ASR
    >>> budget = model.budget_for_asr(target=0.95)
    >>> print(f"Need {budget.budget:.0f} attempts for 95% ASR")

Reference:
    Feng et al., "Statistical Estimation of Adversarial Risk in Large
    Language Models under Best-of-N Sampling" (arXiv:2601.22636, 2026)
"""

__version__ = "0.1.0"

# Main class
from .estimator import SABER

# Data structures
from ._types import DistributionFitResult, CurveFitResult, PredictionResult, BudgetResult

# Low-level functions for advanced usage
from .fitting import fit_beta_binomial_MLE, fit_beta_MLE, fit_log_linear_curve
from .prediction import (
    predict_plugin,
    predict_anchored,
    predict_naive,
    predict_combinatorial,
    predict_log_linear_curve,
    estimate_budget,
    confidence_interval_anchored,
)

__all__ = [
    # Main API
    "SABER",
    # Result types
    "DistributionFitResult",
    "CurveFitResult",
    "PredictionResult",
    "BudgetResult",
    # Fitting functions
    "fit_beta_binomial_MLE",
    "fit_beta_MLE",
    "fit_log_linear_curve",
    # Prediction functions
    "predict_plugin",
    "predict_anchored",
    "predict_naive",
    "predict_combinatorial",
    "predict_log_linear_curve",
    "estimate_budget",
    "confidence_interval_anchored",
]
