Advanced Usage
==============

Heterogeneous Budgets
---------------------

SABER supports different trial counts per query:

.. code-block:: python

   import numpy as np
   from saber import SABER

   k = np.array([3, 5, 0, 2, 8])
   n = np.array([50, 100, 75, 100, 200])  # Different budgets

   model = SABER()
   model.fit(k, n)

Model Selection
---------------

.. code-block:: python

   # SABER-Anchored (default, recommended)
   model = SABER(method="anchored")

   # SABER-Plugin (uses β directly)
   model = SABER(method="plugin")

   # Override for specific prediction
   result = model.predict(N=1000, method="naive")
   result = model.predict(N=1000, method="combinatorial")   # N must be <= min(n)
   result = model.predict(N=1000, method="log_linear_curve")  # auto-fits curve if not fitted

   # Small-N Correction
   result = model.predict(N=1000, method="anchored", correction=True)

Log-Linear Curve Fit
--------------------

The default fit is always the Beta distribution (α, β). You can additionally fit a log-linear curve:

.. code-block:: python

   # Fit both distribution and curve
   model = SABER()
   model.fit(k, n, also_fit_curve=True)

   # Or: fit only distribution; curve will be auto-fitted on first use
   model.fit(k, n)
   result = model.predict(N=1000, method="log_linear_curve")

Low-Level API
-------------

For advanced usage, access the underlying functions directly:

.. code-block:: python

   from saber import fit_beta_binomial, predict_anchored, estimate_budget

   # Fit distribution
   alpha, beta = fit_beta_binomial(k, n)

   # Predict ASR
   asr = predict_anchored(N=1000, n=100, asr_at_n=0.73, alpha=alpha)

   # Estimate budget
   budget = estimate_budget(target_asr=0.95, n=100, asr_at_n=0.73, alpha=alpha)

Scaling Curves
--------------

Generate data for plotting:

.. code-block:: python

   import matplotlib.pyplot as plt

   curve = model.scaling_curve(
       include_naive=True,
       include_combinatorial=True,
       correction=False
   )

   plt.figure(figsize=(8, 5))
   plt.plot(curve['N'], curve['asr_anchored'] * 100, label='SABER-Anchored')
   plt.plot(curve['N'], curve['asr_plugin'] * 100, label='SABER-Plugin')
   if 'asr_log_linear_curve' in curve:
       plt.plot(curve['N'], curve['asr_log_linear_curve'] * 100, label='Log-linear curve')
   plt.plot(curve['N'], curve['asr_naive'] * 100, '--', label='Naive Baseline')
   plt.xscale('log')
   plt.xlabel('Number of Attempts (N)')
   plt.ylabel('ASR@N (%)')
   plt.legend()
   plt.title('ASR Scaling Curve')
   plt.show()

Estimator Variants
------------------

.. list-table::
   :widths: 25 40 35
   :header-rows: 1

   * - Variant
     - Equation
     - Best For
   * - **SABER-Anchored**
     - ``1 - (1-ASR@n)·(n/N)^α``
     - Most cases (default)
   * - **SABER-Plugin**
     - ``1 - Γ(α+β)/Γ(β)·N^(-α)``
     - When ASR@n unavailable
   * - **SABER-Fitting**
     - ``exp(-a·N^(-b))``
     - Alternative scaling model
   * - **Naive Baseline**
     - ``(1/K) Σ (1-(1-θ)^N)``
     - Comparison only
   * - **Combinatorial**
     - ``1 - (1/K) Σ (C(n-k,N)/C(n,N))``
     - N ≤ min(n) only
