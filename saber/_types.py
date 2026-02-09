"""
Data structures and types for SABER.
"""

from dataclasses import dataclass
from typing import Optional, Union
import numpy as np


@dataclass
class DistributionFitResult:
    """Result of fitting the Sample-ASR distribution with a Beta distribution.

    Attributes:
        alpha: Fitted α parameter of Beta distribution.
        beta: Fitted β parameter of Beta distribution.
        n_queries: Number of queries (K).
        n_trials: Number of trials per query (n), or array if heterogeneous.
        asr_at_n: Observed ASR at the measurement budget.
        method: Fitting method used ('beta_binomial' or 'beta').
    """
    alpha: float
    beta: float
    n_queries: int
    n_trials: Union[int, np.ndarray]
    asr_at_n: float
    method: str = "beta_binomial"

    @property
    def mean_theta(self) -> float:
        """Expected value of θ under the fitted Beta distribution."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance_theta(self) -> float:
        """Variance of θ under the fitted Beta distribution."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))


@dataclass
class CurveFitResult:
    """Result of fitting the log-linear curve to Sample-ASR scaling.

    Model: -log(ASR) = a * N^(-b).

    Attributes:
        a: Fitted scale parameter.
        b: Fitted exponent parameter.
        n_queries: Number of queries (K).
        n_trials: Number of trials per query (n), or array if heterogeneous.
        asr_at_n: Observed ASR at the measurement budget.
        method: Fitting method used ('log_linear_curve').
    """
    a: float
    b: float
    n_queries: int
    n_trials: Union[int, np.ndarray]
    asr_at_n: float
    method: str = "log_linear_curve"


@dataclass
class PredictionResult:
    """Result of ASR@N prediction.
    
    Attributes:
        N: Target number of attempts.
        asr: Predicted attack success rate.
        ci_lower: Lower bound of confidence interval (if computed).
        ci_upper: Upper bound of confidence interval (if computed).
        confidence: Confidence level for CI (e.g., 0.95).
        method: Prediction method used ('anchored', 'plugin', or 'naive').
    """
    N: int
    asr: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    confidence: Optional[float] = None
    method: str = "anchored"

    def __repr__(self) -> str:
        if self.ci_lower is not None:
            return (
                f"PredictionResult(N={self.N}, asr={self.asr:.4f}, "
                f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
                f"confidence={self.confidence})"
            )
        return f"PredictionResult(N={self.N}, asr={self.asr:.4f})"


@dataclass
class BudgetResult:
    """Result of budget estimation for target ASR.
    
    Attributes:
        target_asr: Target ASR level (τ).
        budget: Estimated number of attempts needed.
        ci_lower: Lower bound of confidence interval (if computed).
        ci_upper: Upper bound of confidence interval (if computed).
        confidence: Confidence level for CI.
    """
    target_asr: float
    budget: float
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    confidence: Optional[float] = None

    def __repr__(self) -> str:
        if self.ci_lower is not None:
            return (
                f"BudgetResult(target={self.target_asr:.2%}, budget={self.budget:.1f}, "
                f"CI=[{self.ci_lower:.1f}, {self.ci_upper:.1f}])"
            )
        return f"BudgetResult(target={self.target_asr:.2%}, budget={self.budget:.1f})"