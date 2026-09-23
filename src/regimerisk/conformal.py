"""Adaptive conformal VaR/ES and a calibration-monitoring abstention rule.

Conformal step
--------------
Nonconformity score is the standardised loss s_t = -r_t / sigma_hat_t. The
VaR multiplier q_t is the finite-sample conformal quantile of the last
``window`` scores at level 1 - alpha_t, and alpha_t follows the Adaptive
Conformal Inference update of Gibbs & Candes (2021):

    alpha_{t+1} = alpha_t + gamma * (alpha - breach_t)

so a run of breaches pushes alpha_t down (wider VaR) and a run of quiet days
pushes it up. ES is sigma_hat_t times the mean calibration score at or beyond q_t.

Abstention
----------
Before issuing day t's forecast we look at the last ``window`` realised breaches.
Under correct calibration their count is Binomial(window, alpha). We abstain if
the count is implausibly high, or if ACI has had to move alpha_t far from its
nominal value (which also catches the over-conservative case after vol falls).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConformalRisk:
    var: np.ndarray  # positive loss threshold, NaN before start
    es: np.ndarray
    breach: np.ndarray  # 1.0 / 0.0, NaN before start
    alpha_t: np.ndarray  # ACI working miscoverage level used on day t


def adaptive_conformal_var(
    returns: np.ndarray,
    sigma_hat: np.ndarray,
    alpha: float,
    window: int = 250,
    gamma: float = 0.005,
    start: int = 250,
) -> ConformalRisk:
    T = len(returns)
    scores = -returns / sigma_hat
    var, es, breach, alpha_path = (np.full(T, np.nan) for _ in range(4))

    a = alpha
    for t in range(max(start, window), T):
        cal = scores[t - window : t]
        n = len(cal)
        level = np.clip(np.ceil((n + 1) * (1.0 - a)) / n, 0.0, 1.0)
        q = np.quantile(cal, level, method="higher")
        var[t] = sigma_hat[t] * q
        es[t] = sigma_hat[t] * cal[cal >= q].mean()
        breach[t] = float(scores[t] > q)
        alpha_path[t] = a
        a += gamma * (alpha - breach[t])

    return ConformalRisk(var, es, breach, alpha_path)


def abstention_flags(
    breach: np.ndarray,
    alpha: float,
    alpha_t: np.ndarray | None = None,
    window: int = 50,
    k_enter: int = 6,
    drift: float = 0.03,
) -> np.ndarray:
    """True on days where the forecast should be withheld.

    Uses only information available before day t (breaches up to t-1 and the
    ACI level alpha_t, itself a function of breaches up to t-1). Hysteresis: once
    abstaining, resume only when the breach count is back to at most
    expected + 1 and alpha_t is back inside the drift band.
    """
    T = len(breach)
    b = np.nan_to_num(breach, nan=0.0)
    trailing = np.concatenate([[0.0], np.cumsum(b)])
    k_exit = round(alpha * window) + 1
    started = ~np.isnan(breach)

    flags = np.zeros(T, dtype=bool)
    on = False
    for t in range(T):
        if not started[t]:
            continue
        count = trailing[t] - trailing[max(0, t - window)]
        d = 0.0 if alpha_t is None else abs(alpha_t[t] - alpha)
        if on:
            on = not (count <= k_exit and d <= drift)
        else:
            on = count >= k_enter or d > drift
        flags[t] = on
    return flags
