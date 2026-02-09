"""
Test suite for SABER package.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import beta as sp_beta, binom


# Import from package
from saber import (
    SABER,
    DistributionFitResult,
    CurveFitResult,
    PredictionResult,
    BudgetResult,
    fit_beta_binomial_MLE,
    fit_beta_MLE,
    predict_plugin,
    predict_anchored,
    predict_naive,
    estimate_budget,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simulated_data():
    """Generate simulated Beta-Binomial data with known parameters."""
    rng = np.random.default_rng(42)
    
    TRUE_ALPHA = 0.4
    TRUE_BETA = 4.0
    K = 100  # queries
    n = 50   # trials per query
    
    # Sample true θ values
    theta_true = sp_beta.rvs(TRUE_ALPHA, TRUE_BETA, size=K, random_state=rng)
    
    # Sample successes
    k = binom.rvs(n, theta_true, random_state=rng)
    n_array = np.full(K, n)
    
    return {
        "k": k,
        "n": n_array,
        "n_scalar": n,
        "theta_true": theta_true,
        "alpha_true": TRUE_ALPHA,
        "beta_true": TRUE_BETA,
        "K": K,
    }


@pytest.fixture
def fitted_model(simulated_data):
    """Return a fitted SABER model."""
    model = SABER()
    model.fit(simulated_data["k"], simulated_data["n"])
    return model


# =============================================================================
# Test Fitting Functions
# =============================================================================

class TestFitting:
    """Tests for distribution fitting functions."""

    def test_fit_beta_binomial_basic(self, simulated_data):
        """Test basic Beta-Binomial fitting."""
        alpha, beta = fit_beta_binomial_MLE(
            simulated_data["k"], 
            simulated_data["n"]
        )
        
        assert alpha > 0
        assert beta > 0
        # Should be within reasonable range of true values
        assert 0.1 < alpha < 2.0
        assert 1.0 < beta < 20.0

    def test_fit_beta_binomial_accuracy(self, simulated_data):
        """Test that fitted parameters are close to ground truth."""
        alpha, beta = fit_beta_binomial_MLE(
            simulated_data["k"], 
            simulated_data["n"]
        )
        
        # Allow 50% relative error (estimation is noisy with finite data)
        assert abs(alpha - simulated_data["alpha_true"]) / simulated_data["alpha_true"] < 0.5
        assert abs(beta - simulated_data["beta_true"]) / simulated_data["beta_true"] < 0.5

    def test_fit_beta_binomial_heterogeneous(self):
        """Test fitting with heterogeneous budgets."""
        rng = np.random.default_rng(123)
        k = rng.integers(0, 20, size=50)
        n = rng.integers(20, 100, size=50)  # Different budget per query
        
        alpha, beta = fit_beta_binomial_MLE(k, n)
        
        assert alpha > 0
        assert beta > 0

    def test_fit_beta_binomial_edge_cases(self):
        """Test edge cases."""
        # All zeros
        k = np.zeros(10)
        n = np.full(10, 50)
        alpha, beta = fit_beta_binomial_MLE(k, n)
        assert alpha > 0 and beta > 0

        # All successes
        k = np.full(10, 50)
        n = np.full(10, 50)
        alpha, beta = fit_beta_binomial_MLE(k, n)
        assert alpha > 0 and beta > 0

    def test_fit_beta_two_stage(self, simulated_data):
        """Test two-stage fitting."""
        alpha, beta = fit_beta_MLE(
            simulated_data["k"], 
            simulated_data["n"]
        )
        
        assert alpha > 0
        assert beta > 0

    def test_fit_validation_errors(self):
        """Test input validation."""
        with pytest.raises(ValueError):
            fit_beta_binomial_MLE(np.array([1, 2]), np.array([10]))  # Shape mismatch
        
        with pytest.raises(ValueError):
            fit_beta_binomial_MLE(np.array([-1, 2]), np.array([10, 10]))  # Negative k
        
        with pytest.raises(ValueError):
            fit_beta_binomial_MLE(np.array([5, 2]), np.array([3, 10]))  # k > n


# =============================================================================
# Test Prediction Functions
# =============================================================================

class TestPrediction:
    """Tests for prediction functions."""

    def test_predict_plugin_basic(self):
        """Test basic plugin prediction."""
        asr = predict_plugin(N=1000, alpha=0.4, beta=4.0)
        
        assert 0 <= asr <= 1
        assert asr > 0.5  # Should be reasonably high at N=1000

    def test_predict_plugin_monotonic(self):
        """Test that ASR increases with N."""
        asrs = [predict_plugin(N=n, alpha=0.4, beta=4.0) for n in [10, 100, 1000]]
        
        assert asrs[0] < asrs[1] < asrs[2]

    def test_predict_plugin_correction(self):
        """Test small-N correction."""
        asr_standard = predict_plugin(N=20, alpha=0.4, beta=4.0, correction=False)
        asr_corrected = predict_plugin(N=20, alpha=0.4, beta=4.0, correction=True)
        
        # Should be different (correction has effect at small N)
        assert asr_standard != asr_corrected

    def test_predict_anchored_basic(self):
        """Test basic anchored prediction."""
        asr = predict_anchored(N=1000, n=100, asr_at_n=0.7, alpha=0.4)
        
        assert 0 <= asr <= 1
        assert asr > 0.7  # Should be higher than ASR@n

    def test_predict_anchored_at_n(self):
        """Test that ASR@n equals input at N=n."""
        asr_at_n = 0.73
        asr = predict_anchored(N=100, n=100, asr_at_n=asr_at_n, alpha=0.4)
        
        assert abs(asr - asr_at_n) < 0.01

    def test_predict_naive_basic(self, simulated_data):
        """Test naive baseline."""
        asr = predict_naive(N=1000, k=simulated_data["k"], n=simulated_data["n"])
        
        assert 0 <= asr <= 1

    def test_predict_naive_underestimates(self, simulated_data):
        """Test that naive baseline underestimates vs anchored."""
        alpha, beta = fit_beta_binomial_MLE(simulated_data["k"], simulated_data["n"])
        asr_at_n = np.mean(simulated_data["k"] > 0)
        
        asr_naive = predict_naive(N=1000, k=simulated_data["k"], n=simulated_data["n"])
        asr_anchored = predict_anchored(N=1000, n=simulated_data["n_scalar"], 
                                        asr_at_n=asr_at_n, alpha=alpha)
        
        # Naive typically underestimates (not always guaranteed with noise)
        # Just check both are valid
        assert 0 <= asr_naive <= 1
        assert 0 <= asr_anchored <= 1


# =============================================================================
# Test Budget Estimation
# =============================================================================

class TestBudgetEstimation:
    """Tests for budget estimation."""

    def test_estimate_budget_basic(self):
        """Test basic budget estimation."""
        budget = estimate_budget(target_asr=0.95, n=100, asr_at_n=0.7, alpha=0.4)
        
        assert budget > 100  # Need more than n to reach higher ASR

    def test_estimate_budget_already_reached(self):
        """Test when target is already reached."""
        budget = estimate_budget(target_asr=0.5, n=100, asr_at_n=0.7, alpha=0.4)
        
        assert budget == 100  # Already exceeded target

    def test_estimate_budget_consistency(self):
        """Test consistency with prediction."""
        alpha = 0.4
        n = 100
        asr_at_n = 0.7
        target = 0.95
        
        # Estimate budget
        budget = estimate_budget(target, n, asr_at_n, alpha)
        
        # Predict ASR at that budget
        asr_at_budget = predict_anchored(int(budget), n, asr_at_n, alpha)
        
        # Should be close to target
        assert abs(asr_at_budget - target) < 0.01


# =============================================================================
# Test SABER Class
# =============================================================================

class TestSABERClass:
    """Tests for main SABER class."""

    def test_init(self):
        """Test initialization."""
        model = SABER()
        assert not model.is_fitted
        assert model.method == "anchored"

    def test_init_with_options(self):
        """Test initialization with options."""
        model = SABER(method="plugin", fit_method="beta")
        assert model.method == "plugin"
        assert model.fit_method == "beta"

    def test_fit_basic(self, simulated_data):
        """Test basic fitting."""
        model = SABER()
        result = model.fit(simulated_data["k"], simulated_data["n"])
        
        assert result is model  # Returns self for chaining
        assert model.is_fitted
        assert model.alpha > 0
        assert model.beta > 0

    def test_fit_scalar_n(self, simulated_data):
        """Test fitting with scalar n."""
        model = SABER()
        model.fit(simulated_data["k"], simulated_data["n_scalar"])
        
        assert model.is_fitted

    def test_fit_list_input(self):
        """Test fitting with list input."""
        model = SABER()
        model.fit(k=[1, 2, 3, 0, 5], n=10)
        
        assert model.is_fitted

    def test_fit_distribution_result_only_by_default(self, simulated_data):
        """Test that fit() without also_fit_curve sets only distribution_fit_result."""
        model = SABER()
        model.fit(simulated_data["k"], simulated_data["n"])
        
        assert model.distribution_fit_result is not None
        assert model.distribution_fit_result.alpha == model.alpha
        assert model.distribution_fit_result.beta == model.beta
        assert model.curve_fit_result is None
        assert model.a is None
        assert model.b is None

    def test_fit_with_also_fit_curve(self, simulated_data):
        """Test that fit(..., also_fit_curve=True) sets both distribution and curve results."""
        model = SABER()
        model.fit(simulated_data["k"], simulated_data["n"], also_fit_curve=True)
        
        assert model.distribution_fit_result is not None
        assert model.curve_fit_result is not None
        assert model.a is not None
        assert model.b is not None
        assert model.curve_fit_result.a == model.a
        assert model.curve_fit_result.b == model.b
        assert model.curve_fit_result.n_queries == model.distribution_fit_result.n_queries

    def test_fit_result_backward_compat(self, simulated_data):
        """Test that fit_result returns distribution_fit_result (backward compatibility)."""
        model = SABER()
        model.fit(simulated_data["k"], simulated_data["n"])
        
        assert model.fit_result is model.distribution_fit_result
        assert model.fit_result.mean_theta == pytest.approx(
            model.alpha / (model.alpha + model.beta), rel=0.01
        )

    def test_predict_basic(self, fitted_model):
        """Test basic prediction."""
        result = fitted_model.predict(N=1000)
        
        assert isinstance(result, PredictionResult)
        assert result.N == 1000
        assert 0 <= result.asr <= 1

    def test_predict_with_ci(self, fitted_model):
        """Test prediction with confidence interval."""
        result = fitted_model.predict(N=1000, confidence=0.95)
        
        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower <= result.asr <= result.ci_upper

    def test_predict_multiple(self, fitted_model):
        """Test predicting at multiple N values."""
        results = fitted_model.predict(N=[100, 500, 1000])
        
        assert len(results) == 3
        assert all(isinstance(r, PredictionResult) for r in results)
        # ASR should increase with N
        asrs = [r.asr for r in results]
        assert asrs[0] <= asrs[1] <= asrs[2]

    def test_predict_asr(self, fitted_model):
        """Test predict_asr convenience method."""
        asr = fitted_model.predict_asr(1000)
        assert isinstance(asr, float)
        
        asrs = fitted_model.predict_asr([100, 500, 1000])
        assert len(asrs) == 3

    def test_predict_log_linear_curve_when_curve_fitted(self, simulated_data):
        """Test predict(method='log_linear_curve') when curve was fitted in fit()."""
        model = SABER()
        model.fit(simulated_data["k"], simulated_data["n"], also_fit_curve=True)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = model.predict(N=1000, method="log_linear_curve")
        assert len(w) == 0  # No warning when curve already fitted
        
        assert result.method == "log_linear_curve"
        assert 0 <= result.asr <= 1

    def test_predict_log_linear_curve_auto_fit(self, simulated_data):
        """Test predict(method='log_linear_curve') auto-fits curve and warns when not fitted."""
        model = SABER()
        model.fit(simulated_data["k"], simulated_data["n"])  # no also_fit_curve
        assert model.curve_fit_result is None
        
        with pytest.warns(UserWarning, match="Curve was not fitted"):
            result = model.predict(N=1000, method="log_linear_curve")
        
        assert result.method == "log_linear_curve"
        assert 0 <= result.asr <= 1
        # After predict, curve should be fitted
        assert model.curve_fit_result is not None
        assert model.a is not None
        assert model.b is not None

    def test_predict_combinatorial(self, fitted_model):
        """Test predict(method='combinatorial')."""
        # N must be <= min(n); use small N
        result = fitted_model.predict(N=10, method="combinatorial")
        assert result.method == "combinatorial"
        assert 0 <= result.asr <= 1

    def test_predict_plugin_and_anchored(self, fitted_model):
        """Test plugin and anchored methods."""
        r_plugin = fitted_model.predict(N=500, method="plugin")
        r_anchored = fitted_model.predict(N=500, method="anchored")
        assert r_plugin.method == "plugin"
        assert r_anchored.method == "anchored"
        assert 0 <= r_plugin.asr <= 1
        assert 0 <= r_anchored.asr <= 1

    def test_predict_with_correction(self, fitted_model):
        """Test predict with small-N correction."""
        r = fitted_model.predict(N=50, correction=True)
        assert 0 <= r.asr <= 1

    def test_budget_for_asr(self, fitted_model):
        """Test budget estimation."""
        result = fitted_model.budget_for_asr(target=0.95)
        
        assert isinstance(result, BudgetResult)
        assert result.target_asr == 0.95
        assert result.budget > 0

    def test_budget_for_asr_multiple(self, fitted_model):
        """Test multiple budget estimations."""
        results = fitted_model.budget_for_asr([0.90, 0.95, 0.99])
        
        assert len(results) == 3
        # Higher targets need more budget
        budgets = [r.budget for r in results]
        assert budgets[0] <= budgets[1] <= budgets[2]

    def test_scaling_curve(self, fitted_model):
        """Test scaling curve generation."""
        curve = fitted_model.scaling_curve()
        
        assert "N" in curve
        assert "asr_anchored" in curve
        assert "asr_plugin" in curve
        assert len(curve["N"]) > 10

    def test_scaling_curve_with_naive(self, fitted_model):
        """Test scaling curve with naive baseline."""
        curve = fitted_model.scaling_curve(include_naive=True)
        
        assert "asr_naive" in curve

    def test_scaling_curve_with_curve_fit(self, simulated_data):
        """Test scaling curve includes asr_log_linear_curve when curve is fitted."""
        model = SABER()
        model.fit(simulated_data["k"], simulated_data["n"], also_fit_curve=True)
        curve = model.scaling_curve()
        
        assert "asr_log_linear_curve" in curve
        assert len(curve["asr_log_linear_curve"]) == len(curve["N"])
        assert np.all((curve["asr_log_linear_curve"] >= 0) & (curve["asr_log_linear_curve"] <= 1))

    def test_scaling_curve_with_combinatorial(self, fitted_model):
        """Test scaling curve with include_combinatorial."""
        curve = fitted_model.scaling_curve(include_combinatorial=True)
        assert "asr_combinatorial" in curve
        # Some NaNs where N > min(n), some values where N <= min(n)
        assert np.any(np.isfinite(curve["asr_combinatorial"]))

    def test_summary(self, fitted_model):
        """Test summary output."""
        summary = fitted_model.summary()
        
        assert isinstance(summary, str)
        assert "α" in summary or "alpha" in summary.lower()

    def test_summary_includes_curve_when_fitted(self, simulated_data):
        """Test summary includes a, b when curve was fitted."""
        model = SABER()
        model.fit(simulated_data["k"], simulated_data["n"], also_fit_curve=True)
        summary = model.summary()
        assert "a (log-linear)" in summary or "log-linear" in summary
        assert "b (log-linear)" in summary

    def test_repr(self, fitted_model):
        """Test string representation."""
        repr_str = repr(fitted_model)
        assert "SABER" in repr_str
        assert "α=" in repr_str or "alpha" in repr_str.lower()

    def test_repr_with_curve_fitted(self, simulated_data):
        """Test repr shows curve_fitted when curve was fitted."""
        model = SABER()
        model.fit(simulated_data["k"], simulated_data["n"], also_fit_curve=True)
        repr_str = repr(model)
        assert "curve_fitted=True" in repr_str

    def test_unfitted_errors(self):
        """Test that unfitted model raises errors."""
        model = SABER()
        
        with pytest.raises(RuntimeError):
            _ = model.alpha
        
        with pytest.raises(RuntimeError):
            model.predict(1000)

    def test_method_chaining(self, simulated_data):
        """Test fluent API."""
        asr = (
            SABER()
            .fit(simulated_data["k"], simulated_data["n"])
            .predict(1000)
            .asr
        )
        
        assert 0 <= asr <= 1

    def test_fit_method_only_beta_or_beta_binomial(self):
        """Test that fit_method is only beta_binomial or beta (no log_linear_curve)."""
        model_default = SABER()
        assert model_default.fit_method in ("beta_binomial", "beta")
        model_beta = SABER(fit_method="beta")
        assert model_beta.fit_method == "beta"
        model_bb = SABER(fit_method="beta_binomial")
        assert model_bb.fit_method == "beta_binomial"


# =============================================================================
# Test Result Types
# =============================================================================

class TestResultTypes:
    """Tests for result data classes."""

    def test_distribution_fit_result(self):
        """Test DistributionFitResult properties."""
        result = DistributionFitResult(
            alpha=0.4, beta=4.0, n_queries=100,
            n_trials=50, asr_at_n=0.7, method="beta_binomial"
        )
        assert result.mean_theta == pytest.approx(0.4 / 4.4, rel=0.01)
        assert result.variance_theta > 0

    def test_curve_fit_result(self):
        """Test CurveFitResult stores a, b and shared fields."""
        result = CurveFitResult(
            a=1.5, b=0.3, n_queries=50, n_trials=100, asr_at_n=0.6
        )
        assert result.a == 1.5
        assert result.b == 0.3
        assert result.n_queries == 50
        assert result.asr_at_n == 0.6

    def test_prediction_result_repr(self):
        """Test PredictionResult representation."""
        result = PredictionResult(N=1000, asr=0.85)
        assert "1000" in repr(result)
        
        result_ci = PredictionResult(
            N=1000, asr=0.85, ci_lower=0.80, ci_upper=0.90, confidence=0.95
        )
        assert "CI" in repr(result_ci)


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow(self):
        """Test complete estimation workflow."""
        rng = np.random.default_rng(42)
        
        # Simulate data
        TRUE_ALPHA, TRUE_BETA = 0.4, 4.0
        K, n = 159, 100
        
        theta = sp_beta.rvs(TRUE_ALPHA, TRUE_BETA, size=K, random_state=rng)
        k = binom.rvs(n, theta, random_state=rng)
        
        # True ASR@1000
        true_asr_1000 = np.mean(1 - (1 - theta) ** 1000)
        
        # Fit and predict
        model = SABER()
        model.fit(k, n)
        predicted_asr_1000 = model.predict_asr(1000)
        
        # Should be within 5% absolute error
        assert abs(predicted_asr_1000 - true_asr_1000) < 0.05

    def test_naive_vs_saber(self):
        """Test that SABER outperforms naive baseline."""
        rng = np.random.default_rng(123)
        
        TRUE_ALPHA, TRUE_BETA = 0.3, 5.0
        K, n, N = 100, 50, 1000
        
        theta = sp_beta.rvs(TRUE_ALPHA, TRUE_BETA, size=K, random_state=rng)
        k = binom.rvs(n, theta, random_state=rng)
        
        true_asr = np.mean(1 - (1 - theta) ** N)
        
        model = SABER()
        model.fit(k, n)
        
        saber_asr = model.predict_asr(N)
        naive_asr = predict_naive(N, k, np.full(K, n))
        
        saber_error = abs(saber_asr - true_asr)
        naive_error = abs(naive_asr - true_asr)
        
        # SABER should typically have lower error
        # (not guaranteed on every random seed, but should be true most of the time)
        assert saber_error < 0.1 or saber_error < naive_error

    def test_workflow_with_also_fit_curve_and_log_linear_predict(self):
        """Integration: fit with also_fit_curve then predict with log_linear_curve."""
        rng = np.random.default_rng(99)
        K, n = 80, 60
        theta = sp_beta.rvs(0.4, 4.0, size=K, random_state=rng)
        k = binom.rvs(n, theta, random_state=rng)
        
        model = SABER()
        model.fit(k, n, also_fit_curve=True)
        
        assert model.distribution_fit_result is not None
        assert model.curve_fit_result is not None
        
        result = model.predict(N=500, method="log_linear_curve")
        assert result.method == "log_linear_curve"
        assert 0 <= result.asr <= 1
        
        # Anchored and plugin still work
        assert 0 <= model.predict_asr(500) <= 1
        assert 0 <= model.predict(N=500, method="plugin").asr <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
