"""
Uncertainty quantification for Beta-Binomial MLE (α hat, β hat).

Provides Fisher information and asymptotic variance of (α, β) estimates
under the SABER hierarchical model: θ_i ~ Beta(α, β), k_i | θ_i ~ Binomial(n_i, θ_i).
"""

import numpy as np
from scipy.special import polygamma
from typing import Union, Tuple


def trigamma(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Second derivative of log Gamma (digamma'), ψ'(x).

    Parameters
    ----------
    x : float or np.ndarray
        Argument(s).

    Returns
    -------
    float or np.ndarray
        ψ'(x) = polygamma(1, x).
    """
    return polygamma(1, x)


def E_trigamma_alpha_plus_ks(
    alpha: float,
    beta: float,
    n: Union[int, np.ndarray],
    num_simulations: int = 100000,
    ) -> Union[float, np.ndarray]:
    """
    Monte Carlo estimate of E[ψ'(α + k)] under Beta-Binomial(α, β, n).

    Used in the Fisher information for α. Draws θ ~ Beta(α, β), k ~ Binomial(n, θ),
    then averages ψ'(α + k).

    Parameters
    ----------
    alpha, beta : float
        Beta prior parameters.
    n : int or np.ndarray
        Trial count (per query). If array, one value per query.
    num_simulations : int
        Number of Monte Carlo samples.

    Returns
    -------
    float or np.ndarray
        Estimated E[ψ'(α + k)]; scalar if n is int, else array.
    """
    ps = np.random.beta(alpha, beta, size=num_simulations)

    ks = []
    for p_i in ps:
        s = np.random.binomial(n, p_i)
        ks.append(s)

    return np.mean(trigamma(alpha + np.array(ks)), axis=0)


def E_trigamma_beta_plus_n_minus_ks(
    alpha: float,
    beta: float,
    n: Union[int, np.ndarray],
    num_simulations: int = 100000,
    ) -> Union[float, np.ndarray]:
    """
    Monte Carlo estimate of E[ψ'(β + n - k)] under Beta-Binomial(α, β, n).

    Used in the Fisher information for β. Draws θ ~ Beta(α, β), k ~ Binomial(n, θ),
    then averages ψ'(β + n - k).

    Parameters
    ----------
    alpha, beta : float
        Beta prior parameters.
    n : int or np.ndarray
        Trial count (per query). If array, one value per query.
    num_simulations : int
        Number of Monte Carlo samples.

    Returns
    -------
    float or np.ndarray
        Estimated E[ψ'(β + n - k)]; scalar if n is int, else array.
    """
    ps = np.random.beta(alpha, beta, size=num_simulations)

    ks = []
    for p_i in ps:
        s = np.random.binomial(n, p_i)
        ks.append(s)
    return np.mean(trigamma(beta + n - np.array(ks)), axis=0)


def fisher_information_matrix_sample_level(
    alpha: float,
    beta: float,
    n: int,
    num_simulations: int = 100000,
    ) -> np.ndarray:
    """
    Fisher information matrix for (α, β) from a single sample (one query with n trials).

    I(α, β) is 2x2; entries use trigamma and Monte Carlo expectations over k ~ Beta-Binomial.
    For K i.i.d. queries, corpus-level information is K * I(α, β).

    Parameters
    ----------
    alpha, beta : float
        Beta prior parameters.
    n : int
        Number of trials for this sample.
    num_simulations : int
        Monte Carlo sample size for the expectations.

    Returns
    -------
    np.ndarray
        Shape (2, 2); [[I_αα, I_αβ], [I_βα, I_ββ]].
    """
    I11 = trigamma(alpha)-trigamma(alpha+beta) + trigamma(alpha+beta+n) - E_trigamma_alpha_plus_ks(alpha, beta, n, num_simulations)
    I22 = trigamma(beta)-trigamma(alpha+beta) + trigamma(alpha+beta+n) - E_trigamma_beta_plus_n_minus_ks(alpha, beta, n, num_simulations)
    I12 = -trigamma(alpha+beta) + trigamma(alpha+beta+n)
    I21 = -trigamma(alpha+beta) + trigamma(alpha+beta+n)
    return np.array([[I11, I12], [I21, I22]])


def _fisher_information_matrix_homogeneous(
    alpha: float,
    beta: float,
    n: int,
    K: int,
    num_simulations: int = 100000,
    ) -> np.ndarray:
    """
    Fisher information matrix for (α, β) with homogeneous budget: K queries, each with n trials.

    Equal to K * I_sample(α, β, n).
    """
    A = fisher_information_matrix_sample_level(alpha, beta, n, num_simulations)
    return A * K


def _fisher_information_matrix_heterogeneous(
    alpha: float,
    beta: float,
    n: np.ndarray,
    num_simulations: int = 100000,
    ) -> np.ndarray:
    """
    Fisher information matrix for (α, β) with heterogeneous budgets n_i per query.

    Sums the per-query Fisher information; n is array of shape (K,) with n_i for each query.
    """
    I11 = trigamma(alpha)-trigamma(alpha+beta) + trigamma(alpha+beta+n) - E_trigamma_alpha_plus_ks(alpha, beta, n, num_simulations)
    I22 = trigamma(beta)-trigamma(alpha+beta) + trigamma(alpha+beta+n) - E_trigamma_beta_plus_n_minus_ks(alpha, beta, n, num_simulations)
    I12 = -trigamma(alpha+beta) + trigamma(alpha+beta+n)
    I21 = -trigamma(alpha+beta) + trigamma(alpha+beta+n)
    
    I11 = I11.sum()
    I22 = I22.sum()
    I12 = I12.sum()
    I21 = I21.sum()
    return np.array([[I11, I12], [I21, I22]])


def fisher_information_matrix_corpus_level(
    alpha: float,
    beta: float,
    n: Union[int, np.ndarray],
    K: Union[int, None] = None,
    num_simulations: int = 100000,
    ) -> np.ndarray:
    """
    Fisher information matrix for (α, β) at corpus level (all queries).

    Parameters
    ----------
    alpha, beta : float
        Beta prior parameters.
    n : int or np.ndarray
        If int: common trial count; then K must be provided.
        If array: trial count per query, shape (K,).
    K : int or None
        Number of queries. Required when n is int; must be None or match len(n) when n is array.
    num_simulations : int
        Monte Carlo sample size for expectations.

    Returns
    -------
    np.ndarray
        Shape (2, 2); corpus-level Fisher information [[I_αα, I_αβ], [I_βα, I_ββ]].

    Raises
    ------
    ValueError
        If n is int and K is None, or if n is array and K is not None and len(n) != K.
    """
    if isinstance(n, int) and K is None:
        raise ValueError("K must be provided if n is an integer")
    elif isinstance(n, np.ndarray) and K is not None and len(n) != K:
        raise ValueError("n and K must have the same length, or K must be None, if n is an array")

    if isinstance(n, int):
        return _fisher_information_matrix_homogeneous(alpha, beta, n, K, num_simulations)
    else:
        return _fisher_information_matrix_heterogeneous(alpha, beta, n, num_simulations)


def var_alpha_beta_hat(
    alpha: float,
    beta: float,
    n: Union[int, np.ndarray],
    K: Union[None, int, np.ndarray] = None,
    num_simulations: int = 100000,
    ) -> Tuple[float, float]:
    """
    Asymptotic variance of α hat and β hat (MLE under Beta-Binomial model).

    Returns (Var(α hat), Var(β hat)) from the inverse of the Fisher information matrix.
    Uses Monte Carlo to approximate the required expectations in the information matrix.

    Parameters
    ----------
    alpha, beta : float
        True (or fitted) Beta prior parameters.
    n : int or np.ndarray
        Trial count(s). int for homogeneous; array of shape (K,) for heterogeneous.
    K : None, int, or np.ndarray
        Number of queries. Required when n is int. If n is int and K is array,
        interpreted as weights/counts per group (special case).
    num_simulations : int
        Monte Carlo sample size for Fisher information expectations.

    Returns
    -------
    tuple of (float, float)
        (Var(α hat), Var(β hat)).

    Raises
    ------
    ValueError
        If K is array but n is not int.
    """
    if isinstance(K, np.ndarray):
        if not isinstance(n, int):
            raise ValueError("n must be an integer if K is an array")
        I = _fisher_information_matrix_homogeneous(alpha, beta, n, 1, num_simulations)
        I_norm = (I[0,0]*I[1,1]-I[0,1]*I[1,0]) * K
    else:
        I = fisher_information_matrix_corpus_level(alpha, beta, n, K, num_simulations)
        I_norm = I[0,0]*I[1,1]-I[0,1]*I[1,0]
    V11 = I[1,1]/I_norm
    V22 = I[0,0]/I_norm
    return V11, V22


if __name__ == "__main__":
    alpha = 0.5
    beta = 2.0
    n = 100
    K = 2
    print(var_alpha_beta_hat(alpha, beta, n, K))

    print(var_alpha_beta_hat(alpha, beta, np.array([n, n]), K=None))
    print(var_alpha_beta_hat(alpha, beta, n, K=np.array([1, 2])))