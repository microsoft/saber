"""
Main SABER estimator class.

Provides a scikit-learn-style API for fitting and predicting ASR under
Best-of-N sampling.
"""

import warnings

import numpy as np
from typing import Optional, Union, List, Literal
from scipy.special import psi as digamma, polygamma

from ._types import (
    DistributionFitResult,
    CurveFitResult,
    PredictionResult,
    BudgetResult,
)
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
from .uncertainty import var_alpha_beta_hat


class SABER:
    """
    SABER: Scaling-Aware Best-of-N Estimation of Risk.

    A statistical framework for predicting large-scale adversarial risk in LLMs
    under Best-of-N sampling from small-budget measurements.

    The typical workflow is:
        1. Collect jailbreak outcomes: k successes out of n trials per query
        2. Fit the Sample-ASR distribution using `fit()`
        3. Predict ASR@N for large N using `predict()`

    Example:
        >>> from saber import SABER
        >>> 
        >>> # Your jailbreak data: k successes out of n trials for each query
        >>> k = np.array([3, 5, 0, 2, 8, 1, 4, ...])  # successes per query
        >>> n = np.array([100, 100, 100, ...])        # trials per query
        >>> 
        >>> # Fit and predict
        >>> model = SABER()
        >>> model.fit(k, n)
        >>> result = model.predict(N=1000)
        >>> print(f"ASR@1000 = {result.asr:.2%}")
        >>> 
        >>> # Or as a one-liner
        >>> asr = SABER().fit(k, n).predict(1000).asr

    Attributes:
        alpha: Fitted α parameter (scaling exponent).
        beta: Fitted β parameter.
        fit_result: Detailed fitting results.
        is_fitted: Whether the model has been fitted.

    Reference:
        Feng et al., "Statistical Estimation of Adversarial Risk in Large
        Language Models under Best-of-N Sampling" (arXiv:2601.22636, 2026)
    """

    def __init__(
        self,
        method: Literal["anchored", "plugin", "naive", "combinatorial", "log_linear_curve"] = "anchored",
        fit_method: Literal["beta_binomial", "beta"] = "beta_binomial",
        ):
        """
        Initialize SABER estimator.

        Args:
            method: Prediction method to use.
                - "anchored": SABER-Anchored (recommended, uses observed ASR@n)
                - "plugin": SABER-Plugin (uses both α and β directly)
                - "naive": Naive baseline
                - "combinatorial": Direct combinatorial (N must be <= min(n))
                - "log_linear_curve": Log-linear curve (use fit(..., also_fit_curve=True) or will auto-fit on first predict)
            fit_method: Distribution fitting method (Beta only).
                - "beta_binomial": One-stage MLE (recommended)
                - "beta": Two-stage MLE (baseline)
        """
        self.method = method
        self.fit_method = fit_method

        # State: Beta distribution fit (always after fit())
        self._alpha: Optional[float] = None
        self._beta: Optional[float] = None
        self._distribution_fit_result: Optional[DistributionFitResult] = None
        # State: log-linear curve fit (optional, via also_fit_curve or auto in predict)
        self._a: Optional[float] = None
        self._b: Optional[float] = None
        self._curve_fit_result: Optional[CurveFitResult] = None
        self._k: Optional[np.ndarray] = None
        self._n: Optional[np.ndarray] = None

    @property
    def alpha(self) -> float:
        """Fitted α parameter (scaling exponent)."""
        self._check_fitted()
        return self._alpha

    @property
    def beta(self) -> float:
        """Fitted β parameter."""
        self._check_fitted()
        return self._beta

    @property
    def a(self) -> Optional[float]:
        """Fitted log-linear scale parameter a (None until curve is fitted)."""
        self._check_fitted()
        return self._a

    @property
    def b(self) -> Optional[float]:
        """Fitted log-linear exponent parameter b (None until curve is fitted)."""
        self._check_fitted()
        return self._b

    @property
    def distribution_fit_result(self) -> Optional[DistributionFitResult]:
        """Beta distribution fit result (set after fit())."""
        return self._distribution_fit_result

    @property
    def curve_fit_result(self) -> Optional[CurveFitResult]:
        """Log-linear curve fit result (set if also_fit_curve=True or when using log_linear_curve predict)."""
        return self._curve_fit_result

    @property
    def fit_result(self) -> DistributionFitResult:
        """Distribution fit result (for backward compatibility). Same as distribution_fit_result after fit()."""
        self._check_fitted()
        return self._distribution_fit_result

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted (always via Beta distribution fit)."""
        return self._alpha is not None and self._beta is not None


    def fit(
        self,
        k: Union[np.ndarray, List[int]],
        n: Union[np.ndarray, List[int], int],
        fit_method: Optional[Literal["beta_binomial", "beta"]] = None,
        also_fit_curve: bool = False,
        ) -> "SABER":
        """
        Fit the Sample-ASR distribution to observed jailbreak outcomes.

        Always fits the Beta distribution (using fit_method). Optionally also fits
        the log-linear curve when also_fit_curve=True.

        Args:
            k: Number of successful jailbreaks per query. Shape (K,).
            n: Number of trials per query. Either:
                - int: Same budget for all queries
                - array of shape (K,): Heterogeneous budgets
            fit_method: Distribution fitting method (Beta only).
                - "beta_binomial": One-stage MLE (recommended)
                - "beta": Two-stage MLE (baseline)
            also_fit_curve: If True, also fit the log-linear curve (a, b) for log_linear_curve prediction.

        Returns:
            self: For method chaining.

        Example:
            >>> model = SABER()
            >>> model.fit(k=[3, 5, 0, 2], n=100)
            >>> print(f"α = {model.alpha:.3f}, β = {model.beta:.3f}")
            >>> model.fit(k, n, also_fit_curve=True)  # also fit curve for log_linear_curve method
        """
        fit_method = fit_method or self.fit_method
        if fit_method not in ["beta_binomial", "beta"]:
            raise ValueError(f"Invalid fit_method: {fit_method}. Must be 'beta_binomial' or 'beta'.")


        k = np.asarray(k, dtype=np.float64)

        if isinstance(n, (int, float)):
            n = np.full_like(k, n, dtype=np.float64)
        else:
            n = np.asarray(n, dtype=np.float64)

        # Validate
        if k.ndim != 1:
            raise ValueError("k must be 1-dimensional")
        if k.shape != n.shape:
            raise ValueError("k and n must have the same shape")

        self._k = k
        self._n = n

        n_common = int(n[0]) if np.all(n == n[0]) else n
        asr_at_n = float(np.mean(k > 0))

        # Always fit Beta distribution
        if fit_method == "beta_binomial":
            alpha, beta = fit_beta_binomial_MLE(k, n)
        else:
            alpha, beta = fit_beta_MLE(k, n)
        self._alpha = alpha
        self._beta = beta
        self._distribution_fit_result = DistributionFitResult(
            alpha=alpha, beta=beta,
            n_queries=len(k),
            n_trials=n_common,
            asr_at_n=asr_at_n,
            method=fit_method,
        )

        # Optionally fit log-linear curve
        if also_fit_curve:
            self._fit_curve(k, n)
        else:
            self._a = None
            self._b = None
            self._curve_fit_result = None

        return self


    def _fit_curve(
        self,
        k: np.ndarray,
        n: np.ndarray,
        ) -> "SABER":
        """
        Fit the log-linear curve to the Sample-ASR distribution.
        """
        a, b = fit_log_linear_curve(k, n)
        self._a = a
        self._b = b
        n_common = int(n[0]) if np.all(n == n[0]) else n
        asr_at_n = float(np.mean(k > 0))
        self._curve_fit_result = CurveFitResult(
            a=a, b=b,
            n_queries=len(k),
            n_trials=n_common,
            asr_at_n=asr_at_n,
        )
        return self


    def predict(
        self,
        N: Union[int, List[int], np.ndarray],
        confidence: Optional[float] = None,
        method: Optional[Literal["anchored", "plugin", "naive", "combinatorial", "log_linear_curve"]] = None,
        correction: bool = False,
        ) -> Union[PredictionResult, List[PredictionResult]]:
        """
        Predict ASR@N for given budget(s).

        Args:
            N: Target number of attempts. Can be:
                - int: Single prediction
                - list/array: Multiple predictions
            confidence: If provided, compute confidence interval at this level
                (e.g., 0.95 for 95% CI). Only supported for anchored method.
            method: Override the prediction method for this call.
            correction: If True, apply small-N correction (Equation 56) for plugin/anchored.

        Returns:
            PredictionResult or list of PredictionResult.

        Example:
            >>> result = model.predict(N=1000)
            >>> print(f"ASR@1000 = {result.asr:.2%}")
            >>> 
            >>> # With small-N correction
            >>> result = model.predict(N=1000, correction=True)
            >>> 
            >>> # With confidence interval
            >>> result = model.predict(N=1000, confidence=0.95)
            >>> print(f"ASR@1000 = {result.asr:.2%} [{result.ci_lower:.2%}, {result.ci_upper:.2%}]")
            >>> 
            >>> # Multiple predictions
            >>> results = model.predict(N=[100, 500, 1000])
        """
        self._check_fitted()

        method = method or self.method

        # If log_linear_curve requested but curve not fitted, auto-fit and remind
        if method == "log_linear_curve" and (self._curve_fit_result is None or self._a is None or self._b is None):
            warnings.warn(
                "Curve was not fitted previously. Fitting log-linear curve now for this prediction.",
                UserWarning,
                stacklevel=2,
            )
            self._fit_curve(self._k, self._n)


        # Handle array input
        if isinstance(N, (list, np.ndarray)):
            return [self.predict(int(n_val), confidence, method, correction) for n_val in N]

        N = int(N)
        n_budget = int(self._n[0]) if np.all(self._n == self._n[0]) else int(np.mean(self._n))

        # Compute prediction
        if method == "naive":
            asr = predict_naive(N, self._k, self._n)
        elif method == "plugin":
            asr = predict_plugin(N, self._alpha, self._beta, correction=correction)
        elif method == "combinatorial":
            asr = predict_combinatorial(N, self._k, self._n)
        elif method == "log_linear_curve":
            asr = predict_log_linear_curve(N, self._a, self._b)
        else:  # anchored
            asr = predict_anchored(
                N, n_budget, self._distribution_fit_result.asr_at_n, self._alpha, correction=correction
            )

        # Compute CI if requested
        ci_lower, ci_upper = None, None
        if confidence is not None and method == "anchored":
            alpha_se = self._estimate_alpha_se()
            ci_lower, ci_upper = confidence_interval_anchored(
                N, n_budget, self._distribution_fit_result.asr_at_n,
                self._alpha, alpha_se, confidence
            )

        return PredictionResult(
            N=N,
            asr=asr,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence=confidence,
            method=method,
        )


    def predict_asr(
        self,
        N: Union[int, List[int], np.ndarray],
        ) -> Union[float, np.ndarray]:
        """
        Convenience method to get just the ASR value(s).

        Args:
            N: Target number of attempts.

        Returns:
            ASR value(s) as float or array.

        Example:
            >>> asr = model.predict_asr(1000)
            >>> asrs = model.predict_asr([100, 500, 1000])
        """
        if isinstance(N, (list, np.ndarray)):
            results = self.predict(N)
            return np.array([r.asr for r in results])
        return self.predict(N).asr


    def budget_for_asr(
        self,
        target: Union[float, List[float]],
        confidence: Optional[float] = None,
        ) -> Union[BudgetResult, List[BudgetResult]]:
        """
        Estimate budget needed to reach target ASR level(s).

        Uses SABER-Anchored inversion; requires Beta distribution fit (fit_method='beta_binomial' or 'beta').

        Args:
            target: Target ASR (e.g., 0.95 for 95%). Can be a list.
            confidence: If provided, compute CI at this level.

        Returns:
            BudgetResult or list of BudgetResult.

        Example:
            >>> result = model.budget_for_asr(0.95)
            >>> print(f"Need {result.budget:.0f} attempts for 95% ASR")
            >>> 
            >>> # Multiple targets
            >>> results = model.budget_for_asr([0.90, 0.95, 0.99])
        """
        self._check_fitted()
        if self._alpha is None or self._beta is None:
            raise RuntimeError(
                "budget_for_asr requires Beta distribution fit (fit_method='beta_binomial' or 'beta')."
            )

        if isinstance(target, (list, np.ndarray)):
            return [self.budget_for_asr(float(t), confidence) for t in target]

        n_budget = int(self._n[0]) if np.all(self._n == self._n[0]) else int(np.mean(self._n))
        asr_at_n = self._distribution_fit_result.asr_at_n

        budget = estimate_budget(target, n_budget, asr_at_n, self._alpha)

        # CI via delta method (optional)
        ci_lower, ci_upper = None, None
        if confidence is not None:
            from scipy.stats import norm
            z = norm.ppf((1 + confidence) / 2)
            alpha_se = self._estimate_alpha_se()

            # Budget is monotonic in alpha
            alpha_lo = max(self._alpha - z * alpha_se, 1e-6)
            alpha_hi = self._alpha + z * alpha_se

            budget_lo = estimate_budget(target, n_budget, asr_at_n, alpha_hi)
            budget_hi = estimate_budget(target, n_budget, asr_at_n, alpha_lo)
            ci_lower, ci_upper = budget_lo, budget_hi

        return BudgetResult(
            target_asr=target,
            budget=budget,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence=confidence,
        )


    def scaling_curve(
        self,
        N_values: Optional[Union[List[int], np.ndarray]] = None,
        include_naive: bool = False,
        include_combinatorial: bool = False,
        correction: bool = False,
        ) -> dict:
        """
        Generate ASR scaling curve data for plotting.

        Args:
            N_values: List of N values. Default: logarithmic spacing 1 to 10000.
            include_naive: Include naive baseline predictions.
            include_combinatorial: Include combinatorial predictions (only valid for N <= min(n)).
            correction: Apply small-N correction for anchored/plugin.

        Returns:
            Dictionary with 'N', 'asr_anchored'/'asr_plugin' (if Beta distribution fit),
            optionally 'asr_log_linear_curve' (if log-linear curve fit), 'asr_naive', 'asr_combinatorial'.

        Example:
            >>> curve = model.scaling_curve()
            >>> plt.plot(curve['N'], curve['asr_anchored'], label='SABER')
            >>> plt.xscale('log')
        """
        self._check_fitted()

        if N_values is None:
            N_values = np.unique(np.logspace(0, 4, 50).astype(int))

        n_budget = int(self._n[0]) if np.all(self._n == self._n[0]) else int(np.mean(self._n))
        asr_at_n = self._distribution_fit_result.asr_at_n

        result = {"N": np.array(N_values)}

        if self._alpha is not None and self._beta is not None:
            result["asr_anchored"] = np.array([
                predict_anchored(N, n_budget, asr_at_n, self._alpha, correction=correction)
                for N in N_values
            ])
            result["asr_plugin"] = np.array([
                predict_plugin(N, self._alpha, self._beta, correction=correction)
                for N in N_values
            ])

        if self._a is not None and self._b is not None:
            result["asr_log_linear_curve"] = np.array([
                predict_log_linear_curve(N, self._a, self._b)
                for N in N_values
            ])

        if include_naive:
            result["asr_naive"] = np.array([
                predict_naive(N, self._k, self._n)
                for N in N_values
            ])

        if include_combinatorial:
            n_min = int(np.min(self._n))
            asr_comb = np.full(len(N_values), np.nan, dtype=np.float64)
            for i, N in enumerate(N_values):
                if N <= n_min:
                    asr_comb[i] = predict_combinatorial(int(N), self._k, self._n)
                else:
                    asr_comb[i] = None
            result["asr_combinatorial"] = asr_comb

        return result


    def summary(self) -> str:
        """
        Return a summary string of the fitted model.

        Returns:
            Human-readable summary.
        """
        self._check_fitted()

        lines = [
            "SABER Model Summary",
            "=" * 40,
            f"Queries (K):        {self._distribution_fit_result.n_queries}",
            f"Trials per query:   {self._distribution_fit_result.n_trials}",
            f"Fitting method:     {self.fit_method}",
            "",
            "Fitted Parameters:",
        ]
        lines.extend([
            f"  α (scaling rate): {self._alpha:.4f}",
            f"  β:                {self._beta:.4f}",
            f"  E[θ]:             {self._distribution_fit_result.mean_theta:.4f}",
        ])
        if self._curve_fit_result is not None:
            lines.extend([
                f"  a (log-linear):   {self._a:.4f}",
                f"  b (log-linear):   {self._b:.4f}",
            ])
        lines.extend([
            "",
            f"Observed ASR@n:     {self._distribution_fit_result.asr_at_n:.2%}",
            "",
            "Sample Predictions:",
        ])
        # Sample predictions depend on default method and fit type
        try:
            lines.append(f"  ASR@100:          {self.predict_asr(100):.2%}")
            lines.append(f"  ASR@1000:         {self.predict_asr(1000):.2%}")
        except (RuntimeError, ValueError):
            lines.append("  (sample predictions depend on chosen method)")
        return "\n".join(line for line in lines if line)


    def _check_fitted(self) -> None:
        """Raise error if model is not fitted."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")


    def _estimate_alpha_se(self) -> float:
        """
        Estimate standard error of α hat using observed Fisher information.

        This is a simplified approximation; for rigorous inference,
        use bootstrap or the full Fisher information matrix.
        Only valid when fitted with Beta distribution (alpha, beta).
        """
        if self._alpha is None or self._beta is None:
            raise RuntimeError("Alpha SE is only available for Beta distribution fit.")
        # Approximate SE using the observed information
        # This is based on Equation 87 in the paper
        n = int(self._n[0]) if np.all(self._n == self._n[0]) else self._n
        K = len(self._k) if isinstance(n, int) else None
        var_alpha, var_beta = var_alpha_beta_hat(self._alpha, self._beta, n, K)
        return np.sqrt(var_alpha)


    def __repr__(self) -> str:
        if self.is_fitted:
            s = f"SABER(α={self._alpha:.4f}, β={self._beta:.4f}, method='{self.method}'"
            if self._curve_fit_result is not None:
                s += f", curve_fitted=True"
            return s + ")"
        return f"SABER(method='{self.method}', unfitted)"
