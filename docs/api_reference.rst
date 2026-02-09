API Reference
=============

SABER Class
-----------

.. py:class:: SABER(method="anchored", fit_method="beta_binomial")

   Main class for SABER risk estimation.

   :param method: Prediction method - ``"anchored"``, ``"plugin"``, ``"naive"``, ``"combinatorial"``, or ``"log_linear_curve"``
   :type method: str
   :param fit_method: Fitting method - ``"beta_binomial"`` or ``"beta"``
   :type fit_method: str

Methods
^^^^^^^

.. py:method:: fit(k, n, also_fit_curve=False)

   Fit Sample-ASR distribution to data.

   :param k: Array of successes per query
   :param n: Number of trials (int or array)
   :param also_fit_curve: Whether to also fit log-linear curve
   :type also_fit_curve: bool
   :returns: self (for method chaining)

.. py:method:: predict(N, confidence=None, method=None, correction=False)

   Predict ASR@N.

   :param N: Target number of attempts (int or list)
   :param confidence: Confidence level for interval (e.g., 0.95)
   :type confidence: float, optional
   :param method: Override prediction method
   :type method: str, optional
   :param correction: Apply small-N correction
   :type correction: bool
   :returns: PredictionResult or list of PredictionResult

.. py:method:: predict_asr(N)

   Get just the ASR value(s).

   :param N: Target number of attempts
   :returns: ASR value(s)

.. py:method:: budget_for_asr(target, confidence=None)

   Estimate budget for target ASR (requires Beta fit).

   :param target: Target ASR (float or list)
   :param confidence: Confidence level for interval
   :type confidence: float, optional
   :returns: BudgetResult or list of BudgetResult

.. py:method:: scaling_curve(include_naive=True, include_combinatorial=True, correction=False)

   Generate scaling curve data for plotting.

   :param include_naive: Include naive baseline
   :type include_naive: bool
   :param include_combinatorial: Include combinatorial baseline
   :type include_combinatorial: bool
   :param correction: Apply small-N correction
   :type correction: bool
   :returns: Dictionary with curve data

.. py:method:: summary()

   Print model summary.

Properties
^^^^^^^^^^

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Property
     - Description
   * - ``alpha``
     - Fitted α parameter (Beta distribution)
   * - ``beta``
     - Fitted β parameter (Beta distribution)
   * - ``a``
     - Fitted log-linear parameter a (if curve fitted)
   * - ``b``
     - Fitted log-linear parameter b (if curve fitted)
   * - ``distribution_fit_result``
     - ``DistributionFitResult`` (always set after ``fit``)
   * - ``curve_fit_result``
     - ``CurveFitResult`` or ``None``
   * - ``fit_result``
     - Same as ``distribution_fit_result`` (backward compatibility)
   * - ``is_fitted``
     - Whether model is fitted (Beta distribution)

Result Types
------------

PredictionResult
^^^^^^^^^^^^^^^^

Returned by ``predict()``.

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``N``
     - int
     - Number of attempts
   * - ``asr``
     - float
     - Predicted attack success rate
   * - ``ci_lower``
     - float
     - Lower confidence bound
   * - ``ci_upper``
     - float
     - Upper confidence bound
   * - ``confidence``
     - float
     - Confidence level
   * - ``method``
     - str
     - Prediction method used

BudgetResult
^^^^^^^^^^^^

Returned by ``budget_for_asr()``.

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``target_asr``
     - float
     - Target ASR
   * - ``budget``
     - float
     - Estimated number of attempts needed
   * - ``ci_lower``
     - float
     - Lower confidence bound
   * - ``ci_upper``
     - float
     - Upper confidence bound
   * - ``confidence``
     - float
     - Confidence level

DistributionFitResult
^^^^^^^^^^^^^^^^^^^^^

Beta distribution fit result.

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``alpha``
     - float
     - Fitted α parameter
   * - ``beta``
     - float
     - Fitted β parameter
   * - ``n_queries``
     - int
     - Number of queries
   * - ``n_trials``
     - int/array
     - Number of trials per query
   * - ``asr_at_n``
     - float
     - Observed ASR at n
   * - ``method``
     - str
     - Fit method used
   * - ``mean_theta``
     - float
     - Mean of theta distribution
   * - ``variance_theta``
     - float
     - Variance of theta distribution

CurveFitResult
^^^^^^^^^^^^^^

Log-linear curve fit result.

.. list-table::
   :widths: 20 20 60
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``a``
     - float
     - Fitted parameter a
   * - ``b``
     - float
     - Fitted parameter b
   * - ``n_queries``
     - int
     - Number of queries
   * - ``n_trials``
     - int/array
     - Number of trials per query
   * - ``asr_at_n``
     - float
     - Observed ASR at n
   * - ``method``
     - str
     - Fit method used

Low-Level Functions
-------------------

.. py:function:: fit_beta_binomial(k, n)

   Fit Beta-Binomial distribution to success/trial data.

   :param k: Array of successes per query
   :param n: Number of trials (int or array)
   :returns: ``(alpha, beta)`` tuple

.. py:function:: predict_anchored(N, n, asr_at_n, alpha)

   Predict ASR using anchored method.

   :param N: Target number of attempts
   :param n: Observed number of trials
   :param asr_at_n: Observed ASR at n trials
   :param alpha: Fitted α parameter
   :returns: Predicted ASR (float)

.. py:function:: estimate_budget(target_asr, n, asr_at_n, alpha)

   Estimate budget needed for target ASR.

   :param target_asr: Target attack success rate
   :param n: Observed number of trials
   :param asr_at_n: Observed ASR at n trials
   :param alpha: Fitted α parameter
   :returns: Estimated number of attempts needed (float)
