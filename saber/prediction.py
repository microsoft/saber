"""
ASR prediction under Best-of-N sampling.

This module provides functions for predicting Attack Success Rate (ASR)
at large N from small-budget measurements using the SABER scaling law.
"""

import numpy as np
from scipy.special import gamma, comb
from scipy.stats import norm
from typing import Tuple, Optional, Union


def predict_plugin(
    N: Union[int, np.ndarray],
    alpha: float,
    beta: float,
    correction: bool = False,
    ) -> Union[float, np.ndarray]:
    """
    SABER-Plugin estimator (Equation 11).

    Directly applies the scaling law:
        ASR@N ≈ 1 - Γ(α+β)/Γ(β) · N^(-α)

    Args:
        N: Target number of attempts.
        alpha: Fitted α parameter.
        beta: Fitted β parameter.
        correction: Apply small-Ncorrection (Equation 56).

    Returns:
        Predicted ASR@N.

    Example:
        >>> asr = predict_plugin(N=1000, alpha=0.4, beta=4.0)
        >>> print(f"ASR@1000 = {asr:.2%}")
    """
    if correction:
        N_eff = N + beta + (alpha - 1) / 2
    else:
        N_eff = N

    leading_constant = gamma(alpha + beta) / gamma(beta)
    asr = 1.0 - leading_constant * (N_eff ** (-alpha))
    asr = np.clip(asr, 0.0, 1.0)
    if isinstance(N, int):
        asr = asr.item()
    return asr


def predict_anchored(
    N: Union[int, np.ndarray],
    n: int,
    asr_at_n: float,
    alpha: float,
    correction: bool = False,
    ) -> float:
    """
    SABER-Anchored estimator (Equation 12).

    Anchors at a known ASR@n to eliminate sensitivity to β:
        ASR@N ≈ 1 - (1 - ASR@n) · (n/N)^α

    This is the recommended estimator when ASR@n is available.

    Args:
        N: Target number of attempts.
        n: Measurement budget where ASR@n is known.
        asr_at_n: Observed ASR at budget n.
        alpha: Fitted α parameter.
        correction: Apply small-N correction.

    Returns:
        Predicted ASR@N.

    Example:
        >>> asr = predict_anchored(N=1000, n=100, asr_at_n=0.73, alpha=0.4)
        >>> print(f"ASR@1000 = {asr:.2%}")
    """
    if correction:
        beta_approx = 1.0
        n_eff = n + beta_approx + (alpha - 1) / 2
        N_eff = N + beta_approx + (alpha - 1) / 2
    else:
        n_eff = n
        N_eff = N

    asr = 1.0 - (1.0 - asr_at_n) * ((n_eff / N_eff) ** alpha)
    asr = np.clip(asr, 0.0, 1.0)
    if isinstance(N, int):
        asr = asr.item()
    return asr


def predict_log_linear_curve(
    N: Union[int, np.ndarray],
    a: float,
    b: float,
    ) -> Union[float, np.ndarray]:
    """
    Predict ASR using log-linear curve.

    Equation:
        ASR = exp(-a * N ** (-b))
    """
    asr = np.exp(-a * N**(-b))
    asr = np.clip(asr, 0.0, 1.0)
    if isinstance(N, int):
        asr = asr.item()
    return asr
    

def predict_combinatorial(
    N: Union[int, np.ndarray],
    k: np.ndarray,
    n: Union[np.ndarray, int],
    ) -> Union[float, np.ndarray]:
    """
    Predict ASR using combinatorial method.
    """
    k = np.asarray(k, dtype=np.float64).reshape(-1, 1)
    if isinstance(n, int):
        n = np.full_like(k, n, dtype=np.float64)
    else:
        n = np.asarray(n, dtype=np.float64).reshape(-1, 1)

    N_array = np.asarray(N, dtype=np.float64).reshape(1, -1)

    if k.shape != n.shape:
        raise ValueError('k and n must have the same shape')
    
    if not (all(n >= k) and all(k >= 0) and all(n >= 1)):
        raise ValueError(f"Require 0 <= k_i <= n_i and n_i >= 1 for all i.")
    
    if not (all(N_array >= 0) and N_array.max() <= n.min()):
        raise ValueError(f"Require 0 <= N_i <= min(n_i) for all i.")

    numerator = comb(n-k, N_array).mean(axis=0)
    denominator = comb(n, N_array).mean(axis=0)
    asr = 1.0 - numerator / denominator
    asr = np.clip(asr, 0.0, 1.0)
    if isinstance(N, int):
        asr = asr.item()
    return asr


def predict_naive(
    N: Union[int, np.ndarray],
    k: np.ndarray,
    n: Union[np.ndarray, int],
    ) -> Union[float, np.ndarray]:
    """
    Naive baseline estimator (Equation 15).

    Uses observed Sample-ASR (θ̂_i = k_i/n_i) directly:
        ASR@N = (1/K) Σ (1 - (1 - θ̂_i)^N)

    This systematically underestimates ASR@N at large N.

    Args:
        N: Target number of attempts.
        k: Successes per query.
        n: Trials per query.

    Returns:
        Predicted ASR@N.
    """
    k = np.asarray(k, dtype=np.float64)
    if isinstance(n, int):
        n = np.full_like(k, n, dtype=np.float64)
    else:
        n = np.asarray(n, dtype=np.float64)

    if k.shape != n.shape:
        raise ValueError('k and n must have the same shape')
    
    if not (all(n >= k) and all(k >= 0) and all(n >= 1)):
        raise ValueError(f"Require 0 <= k_i <= n_i and n_i >= 1 for all i.")
    
    theta_hat = (k / n).reshape(-1, 1)

    N_array = np.asarray(N, dtype=np.float64).reshape(1, -1)

    asr = 1.0 - (1.0 - theta_hat) ** N_array
    asr = asr.mean(axis=0)
    asr = np.clip(asr, 0.0, 1.0)
    if isinstance(N, int):
        asr = asr.item()
    return asr


def estimate_budget(
    target_asr: float,
    n: int,
    asr_at_n: float,
    alpha: float,
    ) -> float:
    """
    Estimate budget N needed to reach target ASR (Equation 16).

    Inverts the SABER-Anchored formula:
        N̂_τ = n · ((1 - ASR@n) / (1 - τ))^(1/α)

    Args:
        target_asr: Target ASR level τ (e.g., 0.95).
        n: Measurement budget where ASR@n is known.
        asr_at_n: Observed ASR at budget n.
        alpha: Fitted α parameter.

    Returns:
        Estimated budget N_τ.

    Example:
        >>> budget = estimate_budget(target_asr=0.95, n=100, asr_at_n=0.73, alpha=0.4)
        >>> print(f"Need {budget:.0f} attempts to reach 95% ASR")
    """
    if target_asr >= 1.0:
        return float("inf")
    if asr_at_n >= target_asr:
        return float(n)

    N_tau = n * ((1.0 - asr_at_n) / (1.0 - target_asr)) ** (1.0 / alpha)
    return float(N_tau)


def confidence_interval_anchored(
    N: int,
    n: int,
    asr_at_n: float,
    alpha: float,
    alpha_se: float,
    confidence: float = 0.95,
    ) -> Tuple[float, float]:
    """
    Confidence interval for SABER-Anchored (Equation 14).

    Uses Wald interval for α transformed through the prediction formula.

    Args:
        N: Target number of attempts.
        n: Measurement budget.
        asr_at_n: Observed ASR at budget n.
        alpha: Fitted α parameter.
        alpha_se: Standard error of α hat.
        confidence: Confidence level (default 0.95).

    Returns:
        (lower, upper): CI bounds for ASR@N.
    """
    z = norm.ppf((1 + confidence) / 2)

    alpha_lower = max(alpha - z * alpha_se, 1e-6)
    alpha_upper = alpha + z * alpha_se

    ratio = n / N
    # Note: larger α → lower ASR@N for N > n (slower scaling)
    # So alpha_upper gives lower ASR, alpha_lower gives upper ASR
    asr_at_alpha_upper = 1.0 - (1.0 - asr_at_n) * (ratio ** alpha_upper)
    asr_at_alpha_lower = 1.0 - (1.0 - asr_at_n) * (ratio ** alpha_lower)

    # Ensure proper ordering
    ci_lower = min(asr_at_alpha_upper, asr_at_alpha_lower)
    ci_upper = max(asr_at_alpha_upper, asr_at_alpha_lower)

    return (
        float(np.clip(ci_lower, 0.0, 1.0)),
        float(np.clip(ci_upper, 0.0, 1.0)),
    )
