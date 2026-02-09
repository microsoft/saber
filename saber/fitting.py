"""
Distribution fitting for Sample-ASR estimation.

This module provides functions for fitting the Beta distribution to
observed jailbreak outcomes using different estimation methods.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, psi, comb
from scipy.stats import beta as sp_beta
from scipy.stats import betabinom as sp_betabinom
from typing import Tuple, Optional, Union


def fit_beta_binomial_MLE(
    k: np.ndarray, 
    n: Union[np.ndarray, int], 
    init: Tuple[float, float] = (0.5, 2.0), 
    optimize_method: str = 'L-BFGS-B',
    ) -> Tuple[float, float]:
    """
    One-stage Beta-Binomial MLE.

    It is the recommended estimator. Hierarchical model:
        theta_i ~ Beta(alpha, beta)
        k_i | theta_i ~ Binomial(n_i, theta_i)
    After marginalization, k_i ~ Beta-Binomial(n_i; alpha, beta).

    Parameters
    ----------
    k : np.ndarray
        Array of non-negative integers representing successes per observation.
    n : np.ndarray
        Array of positive integers representing trials per observation.
    init : Tuple[float, float], default=(0.5, 2.0)
        Initial guess for (alpha, beta). Must be positive; optimization is done
        in log-space to enforce positivity.
    optimize_method : str, default='L-BFGS-B'
        Optimization method passed to scipy.optimize.minimize.

    Returns
    -------
    Tuple[float, float]
        Estimated (alpha, beta) parameters of the Beta prior.
    """
    if isinstance(n, int):
        n = np.full_like(k, n, dtype=np.float64)
    else:
        n = np.asarray(n, dtype=np.float64)

    k = np.asarray(k, dtype=np.float64)

    if n.shape != k.shape:
        raise ValueError("k and n must have the same shape.")

    if not (all(n >= k) and all(k >= 0) and all(n >= 1)):
        raise ValueError(f"Require 0 <= k_i <= n_i and n_i >= 1 for all i.")

    def nll(par):
        a, b = np.exp(par)
        return -np.sum(sp_betabinom.logpmf(k, n, a, b))

    res = minimize(nll, np.log(init), method=optimize_method)
    return tuple(np.exp(res.x))


def fit_beta_MLE(
    k: np.ndarray,
    n: Union[np.ndarray, int],
    init: Tuple[float, float] = (0.5, 2.0),
    optimize_method: str = 'L-BFGS-B', 
    eps: float = 1e-4
    ) -> Tuple[float, float]:
    """
    Two-stage Beta MLE. (baseline method).

    First computes theta_i = k_i/n_i, then fits Beta(alpha, beta) via MLE.
    
    This treats theta_i as noise-free observations and can be miscalibrated
    when n is small. Use `fit_beta_binomial_MLE` for better accuracy.

    Values exactly at 0 or 1 are nudged inward by `eps` to
    avoid -inf log-likelihood under the Beta distribution.

    Parameters
    ----------
    k : np.ndarray
        Array of non-negative integers representing successes per observation.
    n : np.ndarray
        Array of positive integers representing trials per observation.
    init : Tuple[float, float], default=(0.5, 2.0)
        Initial guess for (alpha, beta). Must be positive; optimization is done
        in log-space to enforce positivity.
    optimize_method : str, default='L-BFGS-B'
        Optimization method passed to scipy.optimize.minimize.
    eps : float, default=1e-4
        Small value to clip proportions away from 0 and 1.

    Returns
    -------
    Tuple[float, float]
        Estimated (alpha, beta) parameters of the Beta distribution.
    """
    k = np.asarray(k, dtype=np.float64)
    if isinstance(n, int):
        n = np.full_like(k, n, dtype=np.float64)
    else:
        n = np.asarray(n, dtype=np.float64)
    
    if n.shape != k.shape:
        raise ValueError("k and n must have the same shape.")

    if not (all(n >= k) and all(k >= 0) and all(n >= 1)):
        raise ValueError(f"Require 0 <= k_i <= n_i and n_i >= 1 for all i.")

    theta = k / n

    # Clamp boundary values
    theta = np.clip(theta, eps, 1 - eps)

    def nll(params: np.ndarray) -> float:
        a, b = np.exp(params)
        return -np.sum(sp_beta.logpdf(theta, a, b))

    result = minimize(nll, np.log(init), method=optimize_method)
    return tuple(np.exp(result.x))


def fit_log_linear_curve(
    k: np.ndarray, 
    n: Union[np.ndarray, int], 
    n_goal: Union[int , None] = None, 
    num_points: Union[int, None] = 30, 
    ) -> Tuple[float, float]:
    """
    Fit a log-linear curve to the data.

    Equation:
        -log(ASR) = a * N ** (-b)

    Parameters
    ----------
    k : np.ndarray
        Array of non-negative integers representing successes per observation.
    n : np.ndarray
        Array of positive integers representing trials per observation.
    n_goal : int, optional
        The number of trials to target. If None, uses the minimum number of trials.
    num_points : int, optional
        The number of points to use for the fit. If None, uses 30 points.

    Returns
    -------
    Tuple[float, float]
        The fitted (a, b) parameters of the log-linear curve.

    Example
    -------
    >>> k = np.array([3, 5, 0, 2, 8])
    >>> n = np.array([10, 10, 10, 10, 10])
    >>> a, b = fit_log_linear_curve(k, n)
    >>> print(f"a = {a}, b = {b}")

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

    if n_goal is None:
        n_goal = n.min()

    if num_points is None:
        m = np.arange(5, n_goal, 1).reshape(1, -1)
    else:
        gap = np.maximum(1, (n_goal - 5) // num_points)
        m = np.arange(5, n_goal, gap).reshape(1, -1)

    repeat_times = (n_goal-1) // n + 1
    n = (n * repeat_times).reshape(-1, 1)
    k = (k * repeat_times).reshape(-1, 1)


    numerator = comb(n-k, m).mean(axis=0)
    denominator = comb(n, m).mean(axis=0)
    asr_at_ms = 1 - numerator / denominator

    m = m.flatten()
    log_neg_log_asr_at_ms = np.log(-np.log(asr_at_ms))
    log_ms = np.log(m)

    # fit a line to the data with self-defined weights
    weights = 1 / m
    line_fit = np.polyfit(log_ms, log_neg_log_asr_at_ms, 1, w=weights)
    line_fit
    slope = line_fit[0]
    offset = line_fit[1]

    b = -slope
    a = np.exp(offset)
    
    return (a, b)

# Deprecated
def fit_beta_binomial_MLE_custom(
    k: np.ndarray,
    n: np.ndarray,
    init: Optional[Tuple[float, float]] = None,
    method: str = "L-BFGS-B",
    ) -> Tuple[float, float]:
    """
    One-stage Beta-Binomial MLE for Sample-ASR distribution.

    This is the recommended estimator. It models the hierarchical structure:
        θ_i ~ Beta(α, β)
        k_i | θ_i ~ Binomial(n_i, θ_i)

    The log-likelihood (Equation 9 from the paper):
        ℓ(α,β) = Σ log B(k_i+α, n_i-k_i+β) - K·log B(α,β)

    Args:
        k: Successes per query, shape (K,).
        n: Trials per query, shape (K,). Supports heterogeneous budgets.
        init: Optional (α₀, β₀) initialization. If None, uses moment matching.
        method: Scipy optimization method.

    Returns:
        (α hat, β hat): Maximum likelihood estimates.

    Raises:
        ValueError: If inputs are invalid.

    Example:
        >>> k = np.array([3, 5, 0, 2, 8])
        >>> n = np.array([10, 10, 10, 10, 10])
        >>> alpha, beta = fit_beta_binomial(k, n)
    """
    k = np.asarray(k, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)

    if k.shape != n.shape:
        raise ValueError("k and n must have the same shape.")
    if np.any(k < 0) or np.any(n <= 0) or np.any(k > n):
        raise ValueError("Require 0 <= k_i <= n_i and n_i >= 1 for all i.")

    K = k.size

    # Initialize via moment matching on smoothed proportions
    if init is None:
        theta_hat = (k + 0.5) / (n + 1.0)
        m = np.clip(theta_hat.mean(), 1e-3, 1 - 1e-3)
        v = max(np.var(theta_hat, ddof=1), 1e-4 * m * (1 - m))
        t = m * (1 - m) / v - 1.0
        alpha0 = max(m * t, 1e-2)
        beta0 = max((1 - m) * t, 1e-2)
    else:
        alpha0, beta0 = init
        if alpha0 <= 0 or beta0 <= 0:
            raise ValueError("Initial alpha and beta must be > 0.")

    # Optimize in log-space to enforce positivity
    x0 = np.log([alpha0, beta0])

    def nll_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
        a, b = np.exp(x)

        # Log-likelihood
        ll = np.sum(betaln(k + a, n - k + b)) - K * betaln(a, b)

        # Gradients via digamma function
        psi_sum = psi(n + a + b)
        psi_ab = psi(a + b)

        g_a = np.sum(psi(k + a) - psi_sum) - K * (psi(a) - psi_ab)
        g_b = np.sum(psi(n - k + b) - psi_sum) - K * (psi(b) - psi_ab)

        # Negative log-likelihood and gradient (chain rule for log-params)
        return -ll, np.array([-g_a * a, -g_b * b])

    result = minimize(
        fun=nll_and_grad,
        x0=x0,
        method=method,
        jac=True,
        options={"maxiter": 500},
    )

    alpha_hat, beta_hat = np.exp(result.x)
    return float(alpha_hat), float(beta_hat)

