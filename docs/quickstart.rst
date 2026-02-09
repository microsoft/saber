Quick Start
===========

Basic Example
-------------

.. code-block:: python

   import numpy as np
   from saber import SABER

   # Your jailbreak evaluation data:
   # k[i] = number of successful jailbreaks for query i
   # n[i] = number of attempts for query i
   k = np.array([3, 5, 0, 2, 8, 1, 4, 0, 6, 2])
   n = 100  # 100 attempts per query

   # Fit and predict
   model = SABER()
   model.fit(k, n)

   # Predict ASR at N=1000 attempts
   result = model.predict(N=1000)
   print(f"ASR@1000 = {result.asr:.2%}")

   # With confidence interval
   result = model.predict(N=1000, confidence=0.95)
   print(f"ASR@1000 = {result.asr:.2%} [{result.ci_lower:.2%}, {result.ci_upper:.2%}]")

Core Workflow
-------------

.. code-block:: python

   from saber import SABER

   # 1. Collect jailbreak data
   k = [...]  # successes per query
   n = 100    # trials per query

   # 2. Fit the model
   model = SABER()
   model.fit(k, n)

   # 3. Predict ASR at target budget
   asr_1000 = model.predict(N=1000).asr

Fluent API
----------

.. code-block:: python

   # One-liner
   asr = SABER().fit(k, n).predict(1000).asr

   # Method chaining
   model = SABER()
   model.fit(k, n)
   print(model.summary())

Multiple Predictions
--------------------

.. code-block:: python

   # Predict at multiple N values
   results = model.predict(N=[100, 500, 1000, 5000])
   for r in results:
       print(f"ASR@{r.N} = {r.asr:.2%}")

   # Or get just the values
   asrs = model.predict_asr([100, 500, 1000])

Budget Estimation
-----------------

.. code-block:: python

   # How many attempts for 95% success rate?
   result = model.budget_for_asr(target=0.95)
   print(f"Need {result.budget:.0f} attempts for 95% ASR")

   # Multiple targets
   results = model.budget_for_asr([0.90, 0.95, 0.99])
