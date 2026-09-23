"""Standard VaR backtests: Kupiec POF, Christoffersen independence, Basel traffic light."""
from __future__ import annotations

import numpy as np
from scipy.special import xlogy
from scipy.stats import chi2


def _binom_loglik(x, n, p):
    return xlogy(n - x, 1.0 - p) + xlogy(x, p)


def kupiec_lr(x, n, alpha: float):
    """Proportion-of-failures likelihood ratio (vectorised over x, n)."""
    x = np.asarray(x, dtype=float)
    n = np.asarray(n, dtype=float)
    phat = np.where(n > 0, x / np.maximum(n, 1), 0.0)
    return -2.0 * (_binom_loglik(x, n, alpha) - _binom_loglik(x, n, phat))


def kupiec_pof(breach: np.ndarray, alpha: float) -> dict:
    b = breach[~np.isnan(breach)]
    lr = float(kupiec_lr(b.sum(), len(b), alpha))
    return {"breaches": int(b.sum()), "n": len(b), "rate": float(b.mean()),
            "lr": lr, "p_value": float(chi2.sf(lr, 1))}


def christoffersen_ind(breach: np.ndarray) -> dict:
    """Test that a breach today does not change the probability of one tomorrow."""
    b = breach[~np.isnan(breach)].astype(int)
    prev, cur = b[:-1], b[1:]
    n00 = np.sum((prev == 0) & (cur == 0))
    n01 = np.sum((prev == 0) & (cur == 1))
    n10 = np.sum((prev == 1) & (cur == 0))
    n11 = np.sum((prev == 1) & (cur == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(len(cur), 1)
    ll_null = _binom_loglik(n01 + n11, len(cur), p)
    ll_alt = _binom_loglik(n01, n00 + n01, p01) + _binom_loglik(n11, n10 + n11, p11)
    lr = float(-2.0 * (ll_null - ll_alt))
    return {"lr": lr, "p_value": float(chi2.sf(lr, 1)), "p_breach_after_breach": float(p11)}


def _trailing_counts(breach: np.ndarray, window: int):
    """Breach count and number of scored days in the window ending at t (inclusive)."""
    valid = ~np.isnan(breach)
    cb = np.concatenate([[0.0], np.cumsum(np.where(valid, breach, 0.0))])
    cn = np.concatenate([[0], np.cumsum(valid)])
    idx = np.arange(len(breach)) + 1
    lo = np.maximum(idx - window, 0)
    return cb[idx] - cb[lo], cn[idx] - cn[lo]


def basel_flags(breach99: np.ndarray, window: int = 250, yellow: int = 5) -> np.ndarray:
    """True once the trailing 250-day 99% VaR breach count reaches the Basel yellow zone.

    The flag for day t uses breaches up to and including t (the backtest is run
    after the close), so it is compared against abstention decisions for t+1.
    """
    count, n = _trailing_counts(breach99, window)
    return (n >= window) & (count >= yellow)


def rolling_kupiec_flags(breach: np.ndarray, alpha: float, window: int = 250,
                         level: float = 0.05) -> np.ndarray:
    """True when a trailing-window Kupiec test rejects AND breaches are too many."""
    count, n = _trailing_counts(breach, window)
    lr = kupiec_lr(count, n, alpha)
    return (n >= window) & (chi2.sf(lr, 1) < level) & (count > alpha * n)
