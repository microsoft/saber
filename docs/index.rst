SABER Documentation
====================

**S**\ caling-**A**\ ware **B**\ est-of-N **E**\ stimation of **R**\ isk

A Python package for predicting large-scale adversarial risk in Large Language Models under Best-of-N sampling.

**Paper**: https://arxiv.org/pdf/2601.22636

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quickstart
   api_reference
   advanced_usage

Overview
--------

Standard LLM safety evaluations use single-shot (ASR@1) metrics, but real attackers can exploit parallel sampling to repeatedly probe models. SABER provides a principled statistical framework to:

- **Predict** ASR@N at large budgets from small measurements
- **Estimate** how many attempts are needed to reach a target success rate
- **Quantify** uncertainty in adversarial risk predictions

Installation
------------

.. code-block:: bash

   pip install saber-risk

Or from source:

.. code-block:: bash

   git clone https://github.com/microsoft/saber
   cd saber
   pip install -e .

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
