"""Run the pipeline on simulated paths and score it against ground truth.

Because every path carries the true conditional distribution of r_t, each VaR
forecast has an exactly computable *true breach probability*
p_t = P(r_t < -VaR_t | past). Calibration error is measured on p_t directly,
rather than on noisy realised breach counts.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .backtest import basel_flags, rolling_kupiec_flags
from .conformal import abstention_flags, adaptive_conformal_var
from .forecast import ewma_vol, gaussian_es, gaussian_var
from .generator import SimulatedPath

EVAL_START = 500  # first scored day: EWMA + 250-day calibration window are warm
HORIZON = 150  # days after a shift in which detection is counted
PRE = 50  # days before a shift shown in the event study
STEADY_STATE = 250  # days since last shift after which a regime counts as settled
EVENT_KINDS = ["vol_up", "liquidity_shock", "tail_lighter", "vol_down", "tail_heavier"]
DETECTORS = ("abstain", "basel", "kupiec")
SEVERE = 2.0  # expected excess 95% breaches over the horizon that make a shift "harmful"


@dataclass(frozen=True)
class Config:
    alpha: float = 0.05  # VaR level monitored by abstention (95% VaR)
    alpha_tail: float = 0.01  # 99% VaR, used for the Basel traffic light
    alpha_es: float = 0.025  # 97.5% ES (FRTB)
    lam: float = 0.94
    cal_window: int = 250
    gamma: float = 0.005
    abst_window: int = 50
    k_enter: int = 6
    drift: float = 0.03


DEFAULT_CONFIG = Config()


def true_breach_prob(var, mu, sigma, nu):
    """P(r_t < -VaR_t) under the true standardised Student-t conditional law."""
    x = (-var - mu) / (sigma * np.sqrt((nu - 2.0) / nu))
    return student_t.cdf(x, nu)


def true_es(alpha, mu, sigma, nu):
    """True expected shortfall (positive loss) of mu + sigma * std_t(nu)."""
    q = student_t.ppf(1.0 - alpha, nu)
    es_t = (nu + q**2) / (nu - 1.0) * student_t.pdf(q, nu) / alpha
    return -mu + sigma * np.sqrt((nu - 2.0) / nu) * es_t


def lagged(flags: np.ndarray) -> np.ndarray:
    """Shift an end-of-day flag so it applies to the next day's forecast."""
    return np.concatenate([[False], flags[:-1]])


def run_path(path: SimulatedPath, cfg: Config = DEFAULT_CONFIG) -> dict[str, np.ndarray]:
    r = path.returns
    sigma_hat = ewma_vol(r, cfg.lam)
    out: dict[str, np.ndarray] = {"sigma_hat": sigma_hat}

    for tag, a in [("95", cfg.alpha), ("99", cfg.alpha_tail), ("975", cfg.alpha_es)]:
        naive_var = gaussian_var(sigma_hat, a)
        naive_breach = (r < -naive_var).astype(float)
        naive_breach[: cfg.cal_window] = np.nan
        conf = adaptive_conformal_var(r, sigma_hat, a, cfg.cal_window, cfg.gamma, cfg.cal_window)
        out[f"naive_var{tag}"] = naive_var
        out[f"naive_breach{tag}"] = naive_breach
        out[f"naive_p{tag}"] = true_breach_prob(naive_var, path.mu, path.sigma, path.nu)
        out[f"conf_var{tag}"] = conf.var
        out[f"conf_breach{tag}"] = conf.breach
        out[f"conf_p{tag}"] = true_breach_prob(conf.var, path.mu, path.sigma, path.nu)
        out[f"conf_alpha{tag}"] = conf.alpha_t
        if tag == "975":
            out["conf_es975"] = conf.es
            out["naive_es975"] = gaussian_es(sigma_hat, a)
            out["true_es975"] = true_es(a, path.mu, path.sigma, path.nu)

    out["abstain"] = abstention_flags(out["conf_breach95"], cfg.alpha, out["conf_alpha95"],
                                      cfg.abst_window, cfg.k_enter, cfg.drift)
    out["basel"] = lagged(basel_flags(out["conf_breach99"]))
    out["kupiec"] = lagged(rolling_kupiec_flags(out["conf_breach95"], cfg.alpha))
    out["basel_naive"] = lagged(basel_flags(out["naive_breach99"]))
    return out


def days_since_shift(path: SimulatedPath) -> np.ndarray:
    T = len(path.returns)
    last = np.zeros(T, dtype=int)
    for s in path.shifts["t"]:
        last[s:] = s
    return np.arange(T) - last


# --------------------------------------------------------------------- events
def _first(mask: np.ndarray) -> float:
    idx = np.flatnonzero(mask)
    return float(idx[0]) if len(idx) else np.nan


def _excess_before(excess: np.ndarray, lag: float) -> float:
    """Cumulative expected excess breaches accrued before detection (whole horizon if none)."""
    n = len(excess) if np.isnan(lag) else int(lag)
    return float(excess[n - 1]) if n > 0 else 0.0


def shift_events(path: SimulatedPath, res: dict, cfg: Config = DEFAULT_CONFIG):
    """One row per regime shift plus aligned traces from -PRE to +HORIZON days."""
    T = len(path.returns)
    shifts = path.shifts.reset_index(drop=True)
    rows, traces = [], []
    for i, sh in shifts.iterrows():
        if sh.kind == "liquidity_exit" or sh.t < EVAL_START:
            continue
        s = int(sh.t)
        nxt = i + 2 if sh.kind == "liquidity_shock" else i + 1  # skip the shock's own exit
        end = min(s + HORIZON, int(shifts.t[nxt]) if nxt < len(shifts) else T, T)
        start = max(s - PRE, int(shifts.t[i - 1]) if i > 0 else 0)
        if sh.kind != "liquidity_shock" and i > 0 and shifts.kind[i - 1] == "liquidity_exit":
            start = max(start, int(shifts.t[i - 1]))

        win = slice(s, end)
        excess = np.cumsum(res["conf_p95"][win] - cfg.alpha)
        lag = {k: _first(res[k][win]) for k in DETECTORS}

        rows.append({
            "t": s, "kind": sh.kind, "from": int(sh.from_regime), "to": int(sh.to_regime),
            "horizon": end - s,
            # A detector already firing at onset is carrying over an earlier
            # alarm; those events are excluded from that detector's lag stats.
            **{f"pre_{k}": bool(res[k][s]) for k in DETECTORS},
            **{f"lag_{k}": v for k, v in lag.items()},
            "excess_before_abstain": _excess_before(excess, lag["abstain"]),
            "excess_before_basel": _excess_before(excess, lag["basel"]),
            "excess_total": float(excess[-1]),
            "p95_conf_first20": float(np.mean(res["conf_p95"][s:min(s + 20, end)])),
            "p95_naive_first20": float(np.mean(res["naive_p95"][s:min(s + 20, end)])),
        })

        trace = {}
        offsets = np.arange(-PRE, HORIZON)
        valid = (s + offsets >= start) & (s + offsets < end)
        idx = np.clip(s + offsets, 0, T - 1)
        for key in ("naive_p95", "conf_p95", "naive_p99", "conf_p99"):
            trace[key] = np.where(valid, res[key][idx], np.nan)
        trace["abstain"] = np.where(valid, res["abstain"][idx].astype(float), np.nan)
        trace["es_ratio_conf"] = np.where(valid, (res["conf_es975"] / res["true_es975"])[idx], np.nan)
        trace["es_ratio_naive"] = np.where(valid, (res["naive_es975"] / res["true_es975"])[idx], np.nan)
        trace["kind"] = sh.kind
        traces.append(trace)
    return pd.DataFrame(rows), traces


def summarise_events(events: pd.DataFrame) -> pd.DataFrame:
    """Per-shift-kind detection statistics.

    For each detector, shifts where it was already firing at onset are excluded.
    The ``all_harmful`` row pools every shift with >= SEVERE expected excess 95%
    breaches over the horizon, whatever its kind.
    """
    out = []
    for kind in ["all_harmful"] + EVENT_KINDS:
        ev = events[events.excess_total >= SEVERE] if kind == "all_harmful" else events[events.kind == kind]
        if ev.empty:
            continue
        row = {"kind": kind, "n": len(ev),
               "harmful_share": float((ev.excess_total >= SEVERE).mean()),
               "p95_naive_first20": ev.p95_naive_first20.mean(),
               "p95_conf_first20": ev.p95_conf_first20.mean()}
        for k in DETECTORS:
            e = ev[~ev[f"pre_{k}"]]
            lag = e[f"lag_{k}"]
            row[f"{k}_detect_rate"] = float(lag.notna().mean())
            row[f"{k}_median_lag"] = float(lag.median()) if lag.notna().any() else np.nan
            if k != "kupiec":
                row[f"excess_before_{k}"] = float(e[f"excess_before_{k}"].median())
        e = ev[~ev.pre_abstain & ~ev.pre_basel]
        la, lb = e.lag_abstain, e.lag_basel
        row["abstain_first_rate"] = float(((la < lb) | (la.notna() & lb.isna())).mean())
        both = la.notna() & lb.notna()
        row["median_lead_vs_basel"] = float((lb - la)[both].median()) if both.any() else np.nan
        out.append(row)
    return pd.DataFrame(out)


def event_curves(traces, events: pd.DataFrame, kinds=EVENT_KINDS) -> dict:
    """Mean true-breach-probability traces and cumulative detection curves per shift kind."""
    curves = {}
    for kind in kinds:
        tr = [t for t in traces if t["kind"] == kind]
        ev = events[events.kind == kind]
        c = {"n": len(tr)}
        for key in ("naive_p95", "conf_p95"):
            c[key] = np.nanmean([t[key] for t in tr], axis=0)
        for det in DETECTORS:
            lags = ev.loc[~ev[f"pre_{det}"], f"lag_{det}"].to_numpy()
            c[f"detect_{det}"] = np.array([np.mean(lags <= k) for k in range(HORIZON)])
        curves[kind] = c
    return curves


# ---------------------------------------------------------- pooled day-level
def steady_state_mask(path: SimulatedPath) -> np.ndarray:
    T = len(path.returns)
    return (np.arange(T) >= EVAL_START) & (days_since_shift(path) >= STEADY_STATE)


def transition_mask(path: SimulatedPath, kinds=("vol_up", "liquidity_shock"), days: int = 50):
    """Days within ``days`` of a shift of the given kinds."""
    T = len(path.returns)
    m = np.zeros(T, dtype=bool)
    for _, sh in path.shifts.iterrows():
        if sh.kind in kinds and sh.t >= EVAL_START:
            m[sh.t : sh.t + days] = True
    return m


def coverage_by_level(paths, levels, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Realised VaR coverage vs nominal, overall / steady-state / post-shift."""
    rows = []
    for lvl in levels:
        a = 1.0 - lvl
        acc = {m: {"all": [], "steady": [], "transition": []} for m in ("naive", "conformal")}
        for path in paths:
            r = path.returns
            sh = ewma_vol(r, cfg.lam)
            covered = {
                "naive": (r >= -gaussian_var(sh, a)).astype(float),
                "conformal": 1.0 - adaptive_conformal_var(r, sh, a, cfg.cal_window, cfg.gamma,
                                                          cfg.cal_window).breach,
            }
            masks = {"all": np.arange(len(r)) >= EVAL_START,
                     "steady": steady_state_mask(path), "transition": transition_mask(path)}
            for m, c in covered.items():
                for region, mask in masks.items():
                    acc[m][region].append(c[mask])
        for m, regions in acc.items():
            rows.append({"nominal": lvl, "model": m,
                         **{region: float(np.concatenate(v).mean()) for region, v in regions.items()}})
    return pd.DataFrame(rows)


def oracle_risk_coverage(results, key: str, fractions, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Upper bound: abstain on exactly the days with the highest true breach probability."""
    p = np.concatenate([r[key][EVAL_START:] for r in results])
    order = np.sort(p)  # ascending: answering the lowest-p days first
    rows = []
    for f in fractions:
        kept = order[: round(f * len(order))]
        rows.append({"answered": f, "danger_rate": float((kept > 2 * cfg.alpha).mean())})
    return pd.DataFrame(rows)


def risk_coverage(paths, results, k_grid, cfg: Config = DEFAULT_CONFIG) -> pd.DataFrame:
    """Sweep the abstention threshold: fraction of days answered vs error on those days."""
    rows = []
    for model in ("conformal", "naive"):
        for k in list(k_grid) + [None]:  # None = never abstain
            answered, err, danger = [], [], []
            for path, res in zip(paths, results):
                breach = res["conf_breach95"] if model == "conformal" else res["naive_breach95"]
                p = res["conf_p95"] if model == "conformal" else res["naive_p95"]
                if k is None:
                    flags = np.zeros(len(p), dtype=bool)
                else:
                    alpha_t = res["conf_alpha95"] if model == "conformal" else None
                    flags = abstention_flags(breach, cfg.alpha, alpha_t, cfg.abst_window, k, cfg.drift)
                ev = np.arange(len(p)) >= EVAL_START
                keep = ev & ~flags
                answered.append(keep.sum() / ev.sum())
                err.append(np.abs(p[keep] - cfg.alpha))
                danger.append(p[keep] > 2 * cfg.alpha)
            rows.append({"model": model, "k_enter": k, "answered": float(np.mean(answered)),
                         "mean_abs_err": float(np.concatenate(err).mean()),
                         "danger_rate": float(np.concatenate(danger).mean())})
    return pd.DataFrame(rows)
