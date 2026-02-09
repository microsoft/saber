"""
Example usage of the SABER package.

This script demonstrates the main features of SABER for 
predicting LLM adversarial risk under Best-of-N sampling: 
fitting, prediction (anchored, plugin, naive, combinatorial, log-linear curve), 
budget estimation, scaling curves, and uncertainty quantification.
"""

import numpy as np
from scipy.stats import beta as sp_beta, binom

from saber import SABER


def main():
    print("=" * 60)
    print("SABER Example: Predicting LLM Adversarial Risk")
    print("=" * 60)

    # =========================================================================
    # Simulate jailbreak evaluation data
    # =========================================================================
    print("\n1. Simulating Jailbreak Data")
    print("-" * 60)

    # In practice, you would collect this from actual jailbreak evaluations:
    # - Run N attempts per harmful query
    # - Count how many succeed (k)

    # For this example, we simulate with known ground truth
    rng = np.random.default_rng(42)

    TRUE_ALPHA = 0.4   # Controls scaling speed
    TRUE_BETA = 4.0    # Controls distribution shape
    K = 159            # Number of harmful queries (like HarmBench)
    n = 100            # Measurement budget per query

    # Sample "true" per-query vulnerabilities from Beta distribution
    theta_true = sp_beta.rvs(TRUE_ALPHA, TRUE_BETA, size=K, random_state=rng)

    # Sample observed successes from Binomial
    k = binom.rvs(n, theta_true, random_state=rng)

    print(f"   Queries (K):          {K}")
    print(f"   Trials per query (n): {n}")
    print(f"   Total attempts:       {K * n:,}")
    print(f"   Observed successes:   {k.sum():,}")

    # =========================================================================
    # Fit SABER model
    # =========================================================================
    print("\n2. Fitting SABER Model")
    print("-" * 60)

    model = SABER()
    model.fit(k, n)

    print(f"   Fitted α: {model.alpha:.4f} (true: {TRUE_ALPHA})")
    print(f"   Fitted β: {model.beta:.4f} (true: {TRUE_BETA})")
    print(f"   E[θ]:     {model.fit_result.mean_theta:.4f}")

    # Optional: also fit log-linear curve for method="log_linear_curve"
    model.fit(k, n, also_fit_curve=True)
    print(f"   (Also fitted curve: a={model.a:.4f}, b={model.b:.4f})")

    # =========================================================================
    # Predict ASR at large N
    # =========================================================================
    print("\n3. Predicting ASR@N")
    print("-" * 60)

    # Single prediction
    result = model.predict(N=1000)
    true_asr_1000 = np.mean(1 - (1 - theta_true) ** 1000)

    print(f"   ASR@1000 prediction: {result.asr:.2%}")
    print(f"   ASR@1000 true:       {true_asr_1000:.2%}")
    print(f"   Absolute error:      {abs(result.asr - true_asr_1000):.2%}")

    # With confidence interval
    result_ci = model.predict(N=1000, confidence=0.95)
    print(f"\n   95% CI: [{result_ci.ci_lower:.2%}, {result_ci.ci_upper:.2%}]")

    # Other prediction methods (curve already fitted above)
    r_plugin = model.predict(N=1000, method="plugin")
    r_curve = model.predict(N=1000, method="log_linear_curve")
    r_naive = model.predict(N=1000, method="naive")
    print(f"\n   Plugin:   {r_plugin.asr:.2%}  |  Log-linear: {r_curve.asr:.2%}  |  Naive: {r_naive.asr:.2%}")

    # Combinatorial only valid for N <= min(n)
    r_comb = model.predict(N=50, method="combinatorial")
    print(f"   Combinatorial ASR@50: {r_comb.asr:.2%}")

    # With small-N correction (for anchored/plugin at small N)
    r_corrected = model.predict(N=50, correction=True)
    print(f"   Anchored@50 (with correction): {r_corrected.asr:.2%}")

    # Multiple predictions
    print("\n   Scaling predictions:")
    print(f"   {'N':>8} | {'Predicted':>12} | {'True':>12} | {'Error':>10}")
    print(f"   {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")

    for N in [20, 50, 100, 200, 500, 1000, 2000]:
        pred = model.predict_asr(N)
        true = np.mean(1 - (1 - theta_true) ** N)
        err = abs(pred - true)
        print(f"   {N:>8} | {pred:>11.2%} | {true:>11.2%} | {err:>9.2%}")

    # =========================================================================
    # Budget estimation
    # =========================================================================
    print("\n4. Budget Estimation")
    print("-" * 60)

    targets = [0.90, 0.95, 0.99]
    print(f"   {'Target ASR':>12} | {'Estimated N':>12}")
    print(f"   {'-'*12}-+-{'-'*12}")

    for target in targets:
        budget = model.budget_for_asr(target)
        print(f"   {target:>11.0%} | {budget.budget:>12.0f}")

    # =========================================================================
    # Scaling curve (optional keys: asr_log_linear_curve, asr_combinatorial)
    # =========================================================================
    print("\n5. Scaling Curve")
    print("-" * 60)

    curve = model.scaling_curve(include_naive=True, include_combinatorial=True)
    print(f"   Keys: {list(curve.keys())}")
    print(f"   N range: {curve['N'].min()} .. {curve['N'].max()} ({len(curve['N'])} points)")

    # =========================================================================
    # Compare with naive baseline
    # =========================================================================
    print("\n6. SABER vs Naive Baseline")
    print("-" * 60)

    from saber import predict_naive

    N = 1000
    true_asr = np.mean(1 - (1 - theta_true) ** N)
    saber_asr = model.predict_asr(N)
    naive_asr = predict_naive(N, k, np.full(K, n))

    saber_err = abs(saber_asr - true_asr)
    naive_err = abs(naive_asr - true_asr)
    reduction = (naive_err - saber_err) / naive_err * 100

    print(f"   True ASR@1000:   {true_asr:.2%}")
    print(f"   SABER:           {saber_asr:.2%} (error: {saber_err:.2%})")
    print(f"   Naive baseline:  {naive_asr:.2%} (error: {naive_err:.2%})")
    print(f"   Error reduction: {reduction:.1f}%")

    # =========================================================================
    # Uncertainty: asymptotic variance of (α̂, β̂)
    # =========================================================================
    print("\n7. Uncertainty (Var(α̂), Var(β̂))")
    print("-" * 60)

    from saber.uncertainty import var_alpha_beta_hat

    var_alpha, var_beta = var_alpha_beta_hat(alpha=model.alpha, beta=model.beta, n=n, K=K)
    print(f"   Var(α̂): {var_alpha:.6f}  (SE: {var_alpha**0.5:.4f})")
    print(f"   Var(β̂): {var_beta:.6f}  (SE: {var_beta**0.5:.4f})")

    # =========================================================================
    # Model summary
    # =========================================================================
    print("\n8. Model Summary")
    print("-" * 60)
    print(model.summary())

    # =========================================================================
    # Fluent API example
    # =========================================================================
    print("\n9. One-liner API")
    print("-" * 60)
    asr = SABER().fit(k, n).predict(1000).asr
    print(f"   SABER().fit(k, n).predict(1000).asr = {asr:.2%}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
