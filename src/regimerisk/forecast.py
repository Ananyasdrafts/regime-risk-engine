"""Baseline risk forecaster: RiskMetrics EWMA volatility with Gaussian VaR/ES.

Deliberately boring. The point of the project is to measure *when* a standard
model's risk estimates go wrong, not to build a better volatility model.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def ewma_vol(returns: np.ndarray, lam: float = 0.94, init_window: int = 20) -> np.ndarray:
    """One-step-ahead volatility forecast: sigma_hat[t] uses returns up to t-1.

    The first ``init_window`` returns seed the variance; those days fall inside
    the evaluation warm-up and are never scored.
    """
    T = len(returns)
    sigma2 = np.empty(T)
    sigma2[0] = np.var(returns[:init_window])
    for t in range(1, T):
        sigma2[t] = lam * sigma2[t - 1] + (1.0 - lam) * returns[t - 1] ** 2
    return np.sqrt(sigma2)


def gaussian_var(sigma_hat: np.ndarray, alpha: float) -> np.ndarray:
    """VaR as a positive loss: P(r_t < -VaR_t) = alpha under N(0, sigma_hat^2)."""
    return -norm.ppf(alpha) * sigma_hat


def gaussian_es(sigma_hat: np.ndarray, alpha: float) -> np.ndarray:
    """Expected shortfall E[-r | r < -VaR] under N(0, sigma_hat^2)."""
    return sigma_hat * norm.pdf(norm.ppf(alpha)) / alpha
