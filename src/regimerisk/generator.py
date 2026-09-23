"""Synthetic return generator with ground-truth-labelled regime shifts.

Returns follow a regime-switching GARCH(1,1) with standardised Student-t
innovations:

    r_t      = mu_k + sigma_t * z_t,           z_t ~ t(nu_k) scaled to unit variance
    sigma2_t = omega_k + alpha_k * eps2_{t-1} + beta_k * sigma2_{t-1}
    omega_k  = sigma_bar_k^2 * (1 - alpha_k - beta_k)

where k is the regime active at t. At a regime change the GARCH state is
rescaled so that sigma2 / sigma_bar_k^2 is preserved, which makes the change
in volatility level happen exactly at the labelled timestamp rather than
leaking in over the GARCH half-life. Every simulated path carries the true
conditional (mu_t, sigma_t, nu_t), so the true probability of any VaR breach
can be computed exactly downstream.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Regime:
    name: str
    ann_vol: float
    nu: float
    alpha: float
    beta: float
    mu: float  # daily drift (log-return)

    @property
    def daily_vol(self) -> float:
        return self.ann_vol / np.sqrt(TRADING_DAYS)

    @property
    def omega(self) -> float:
        return self.daily_vol**2 * (1.0 - self.alpha - self.beta)


CALM, ELEVATED, CRISIS, FAT_TAIL, LIQUIDITY = range(5)

REGIMES: dict[int, Regime] = {
    CALM: Regime("calm", 0.12, 8.0, 0.05, 0.93, 0.0003),
    ELEVATED: Regime("elevated", 0.22, 6.0, 0.08, 0.90, 0.0),
    CRISIS: Regime("crisis", 0.45, 4.0, 0.12, 0.86, -0.0010),
    FAT_TAIL: Regime("fat_tail", 0.12, 3.0, 0.05, 0.93, 0.0003),
    LIQUIDITY: Regime("liquidity", 0.70, 3.0, 0.15, 0.80, -0.0050),
}

# One-off gap move on the first day of a liquidity shock, in calm daily vols.
LIQUIDITY_JUMP_SIGMAS = -6.0

# (start_day, regime) pairs for the fixed scenario used in the figures.
FIXED_SCHEDULE: list[tuple[int, int]] = [
    (0, CALM),
    (1000, ELEVATED),
    (1400, CRISIS),
    (1600, CALM),
    (2000, LIQUIDITY),
    (2010, CALM),
    (2300, FAT_TAIL),
    (2700, CALM),
]
FIXED_T = 3000


@dataclass
class SimulatedPath:
    returns: np.ndarray
    prices: np.ndarray
    sigma: np.ndarray  # true conditional vol of r_t given the past
    mu: np.ndarray  # true conditional mean
    nu: np.ndarray  # true innovation degrees of freedom
    regime: np.ndarray
    shifts: pd.DataFrame  # columns: t, from_regime, to_regime, kind


def std_t(nu: float, size, rng: np.random.Generator) -> np.ndarray:
    """Student-t draws rescaled to unit variance (requires nu > 2)."""
    return rng.standard_t(nu, size) * np.sqrt((nu - 2.0) / nu)


def schedule_to_labels(schedule: list[tuple[int, int]], T: int) -> np.ndarray:
    regime = np.empty(T, dtype=int)
    starts = [s for s, _ in schedule] + [T]
    for (start, k), end in zip(schedule, starts[1:]):
        regime[start:end] = k
    return regime


def classify_shift(src: int, dst: int) -> str:
    """Label a regime change by what it does to the return distribution."""
    if dst == LIQUIDITY:
        return "liquidity_shock"
    if src == LIQUIDITY:
        return "liquidity_exit"
    ratio = REGIMES[dst].daily_vol / REGIMES[src].daily_vol
    if abs(np.log(ratio)) < 0.05:
        return "tail_heavier" if REGIMES[dst].nu < REGIMES[src].nu else "tail_lighter"
    return "vol_up" if ratio > 1 else "vol_down"


def extract_shifts(regime: np.ndarray) -> pd.DataFrame:
    idx = np.flatnonzero(np.diff(regime)) + 1
    rows = [
        (int(t), int(regime[t - 1]), int(regime[t]), classify_shift(regime[t - 1], regime[t]))
        for t in idx
    ]
    return pd.DataFrame(rows, columns=["t", "from_regime", "to_regime", "kind"])


def random_schedule(
    T: int,
    rng: np.random.Generator,
    warmup: int = 500,
    min_dur: int = 150,
    max_dur: int = 400,
    p_liquidity: float = 0.2,
) -> list[tuple[int, int]]:
    """Random regime sequence: calm warm-up, then regimes of 150-400 days with
    occasional 5-15 day liquidity shocks that revert to the prior regime."""
    schedule = [(0, CALM)]
    t, cur = warmup, CALM
    while t < T - min_dur:
        if rng.random() < p_liquidity:
            schedule.append((t, LIQUIDITY))
            t += int(rng.integers(5, 16))
            schedule.append((t, cur))
        else:
            cur = int(rng.choice([k for k in (CALM, ELEVATED, CRISIS, FAT_TAIL) if k != cur]))
            schedule.append((t, cur))
        t += int(rng.integers(min_dur, max_dur + 1))
    return schedule


def simulate(
    schedule: list[tuple[int, int]], T: int, seed: int | None = None, s0: float = 100.0
) -> SimulatedPath:
    rng = np.random.default_rng(seed)
    regime = schedule_to_labels(schedule, T)
    onsets = {s for s, k in schedule if k == LIQUIDITY}

    returns = np.empty(T)
    sigma = np.empty(T)
    mu = np.array([REGIMES[k].mu for k in regime])
    nu = np.array([REGIMES[k].nu for k in regime])

    sigma2 = REGIMES[regime[0]].daily_vol ** 2
    eps_prev = 0.0
    for t in range(T):
        p = REGIMES[regime[t]]
        if t > 0:
            if regime[t] != regime[t - 1]:
                # Carry the GARCH state across the break in relative terms.
                q = REGIMES[regime[t - 1]]
                sigma2 *= p.daily_vol**2 / q.daily_vol**2
                eps_prev *= p.daily_vol / q.daily_vol
            sigma2 = p.omega + p.alpha * eps_prev**2 + p.beta * sigma2
        sigma[t] = np.sqrt(sigma2)
        eps = sigma[t] * std_t(p.nu, None, rng)
        if t in onsets:
            eps += LIQUIDITY_JUMP_SIGMAS * REGIMES[CALM].daily_vol
            # The jump is a known-in-advance event only to the simulator; for the
            # "true" conditional distribution we leave sigma/mu untouched so the
            # jump shows up as an unforecastable tail event.
        returns[t] = p.mu + eps
        eps_prev = eps

    prices = s0 * np.exp(np.cumsum(returns))
    return SimulatedPath(returns, prices, sigma, mu, nu, regime, extract_shifts(regime))


def fixed_scenario(seed: int = 7) -> SimulatedPath:
    return simulate(FIXED_SCHEDULE, FIXED_T, seed=seed)


def monte_carlo(n_paths: int, T: int = 3000, seed: int = 0) -> list[SimulatedPath]:
    master = np.random.default_rng(seed)
    paths = []
    for _ in range(n_paths):
        sub_seed = int(master.integers(2**32))
        sched = random_schedule(T, np.random.default_rng(sub_seed))
        paths.append(simulate(sched, T, seed=sub_seed + 1))
    return paths
