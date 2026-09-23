"""Figures. Colour follows the entity across every figure:
conformal / abstention = blue, naive Gaussian = orange, Basel = aqua, Kupiec = yellow."""
from __future__ import annotations

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate import HORIZON, PRE
from .generator import REGIMES, SimulatedPath

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
REGIME_SHADE = {"calm": None, "elevated": "#ecebe7", "crisis": "#d9d8d3",
                "fat_tail": "#ecebe7", "liquidity": "#c3c2bc"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelsize": 9.5, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.5,
    "font.family": "sans-serif",
})


def _shade_regimes(ax, path: SimulatedPath, label: bool = False):
    starts = [0] + list(path.shifts["t"]) + [len(path.regime)]
    for a, b in pairwise(starts):
        name = REGIMES[path.regime[a]].name
        color = REGIME_SHADE[name]
        if color:
            ax.axvspan(a, b, color=color, lw=0, zorder=0,
                       hatch="////" if name == "fat_tail" else None, ec="#d9d8d3" if name == "fat_tail" else None)
        if label and name != "calm":
            ax.text((a + b) / 2, 1.01, name.replace("_", "-"), transform=ax.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=7.5, color=INK2)


def _shade_flags(ax, flags, color=BLUE, alpha=0.12):
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ax.axvspan(a, b, color=color, alpha=alpha, lw=0, zorder=0.5)


def sanity_plot(path: SimulatedPath, res: dict, fname: str):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1, 1]})
    t = np.arange(len(path.returns))
    ax = axes[0]
    _shade_regimes(ax, path, label=True)
    ax.plot(t, path.prices, color=INK, lw=1.1)
    ax.set_ylabel("price")
    ax.set_title("Synthetic price path with ground-truth regimes", pad=16)

    ax = axes[1]
    _shade_regimes(ax, path)
    ax.plot(t, path.returns * 100, color=INK2, lw=0.6)
    ax.set_ylabel("daily return (%)")

    ax = axes[2]
    _shade_regimes(ax, path)
    ann = np.sqrt(252) * 100
    ax.plot(t, path.sigma * ann, color=INK, lw=1.2, label="true conditional vol")
    ax.plot(t, res["sigma_hat"] * ann, color=ORANGE, lw=1.2, label="EWMA forecast (λ = 0.94)")
    ax.set_ylabel("annualised vol (%)")
    ax.set_xlabel("trading day")
    ax.legend(loc="upper left")
    for s in path.shifts["t"]:
        for a in axes:
            a.axvline(s, color=MUTED, lw=0.6, ls=":")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def var_band_plot(path: SimulatedPath, res: dict, fname: str, window=slice(900, 2900)):
    """Returns vs 99% VaR and 97.5% ES, with abstention periods shaded."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    t = np.arange(len(path.returns))[window]
    _shade_regimes(ax, path, label=True)
    _shade_flags(ax, res["abstain"])
    ax.plot(t, path.returns[window] * 100, color=MUTED, lw=0.6, label="daily return")
    ax.plot(t, -res["naive_var99"][window] * 100, color=ORANGE, lw=1.2, label="naive Gaussian 99% VaR")
    ax.plot(t, -res["conf_var99"][window] * 100, color=BLUE, lw=1.2, label="conformal 99% VaR")
    b = window.start + np.flatnonzero(res["conf_breach99"][window] == 1)
    ax.scatter(b, path.returns[b] * 100, s=16, color=BLUE, ec=SURFACE, lw=1, zorder=3,
               label="conformal 99% breach")
    ax.set_xlim(window.start, window.stop)
    ax.set_ylabel("daily return (%)")
    ax.set_xlabel("trading day  (blue shading = abstaining)")
    ax.set_title("Fixed scenario: 99% VaR forecasts and abstention periods", pad=16)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def coverage_plot(cov, path: SimulatedPath, res: dict, fname: str, roll: int = 20):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})
    lo, hi = cov.nominal.min() - 0.01, 1.0
    a1.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="perfect calibration")
    for model, color in [("naive", ORANGE), ("conformal", BLUE)]:
        c = cov[cov.model == model]
        a1.plot(c.nominal, c["all"], color=color, marker="o", ms=5, mec=SURFACE, label=f"{model}, all days")
        a1.plot(c.nominal, c.transition, color=color, marker="o", ms=5, mec=SURFACE, ls=":",
                label=f"{model}, ≤50 days after vol-up/shock")
    a1.set_xlabel("nominal VaR coverage")
    a1.set_ylabel("realised coverage")
    a1.set_title("Coverage vs nominal (Monte Carlo)", pad=16)
    a1.legend(loc="lower right", fontsize=7.5)

    t = np.arange(len(path.returns))
    _shade_regimes(a2, path, label=True)
    _shade_flags(a2, res["abstain"])
    for key, color, lab in [("naive_p95", ORANGE, "naive"), ("conf_p95", BLUE, "conformal")]:
        sm = np.convolve(res[key], np.ones(roll) / roll, mode="full")[: len(t)]
        sm[: 500 + roll] = np.nan
        a2.plot(t, sm, color=color, lw=1.3, label=lab)
    a2.axhline(0.05, color=INK2, lw=0.9, ls="--")
    a2.text(510, 0.052, "nominal 5%", fontsize=7.5, color=INK2, va="bottom")
    a2.set_xlim(500, len(t))
    a2.set_ylabel(f"true P(breach of 95% VaR), {roll}-day mean")
    a2.set_xlabel("trading day  (blue shading = abstaining)")
    a2.set_title("True breach probability, fixed scenario", pad=16)
    a2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def risk_coverage_plot(rc, oracle, fname: str):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(oracle.answered, oracle.danger_rate, color=MUTED, ls="--", lw=1.2)
    ax.text(0.76, oracle.danger_rate.iloc[0] + 0.001, "oracle abstention\n(knows true breach prob.)",
            fontsize=7.5, color=INK2, va="bottom")
    for model, color, lab in [("conformal", BLUE, "conformal + abstention"),
                              ("naive", ORANGE, "naive + abstention")]:
        c = rc[rc.model == model].sort_values("answered")
        ax.plot(c.answered, c.danger_rate, color=color, marker="o", ms=5, mec=SURFACE, label=lab)
        base = c[c.k_enter.isna()].danger_rate.iloc[0]
        ax.axhline(base, color=color, lw=0.8, ls=":", alpha=0.8)
        for _, row in c.dropna(subset=["k_enter"]).iterrows():
            if row.k_enter in (4, 6, 8):
                ax.annotate(f"k={int(row.k_enter)}", (row.answered, row.danger_rate), fontsize=7,
                            color=INK2, xytext=(0, -12), textcoords="offset points", ha="center")
    fig.text(0.01, 0.01, "Dotted lines: never abstain (what random abstention would also give). "
             "k = breaches in 50 days that trigger abstention.", fontsize=7, color=INK2)
    ax.set_xlim(0.73, 1.005)
    ax.set_ylim(0, None)
    ax.set_xlabel("fraction of days a forecast is issued")
    ax.set_ylabel("share of issued forecasts with\ntrue breach prob. > 2× nominal")
    ax.set_title("Risk-coverage trade-off (95% VaR)")
    ax.legend(loc="center left", bbox_to_anchor=(0, 0.62))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(fname, dpi=150)
    plt.close(fig)


def event_study_plot(curves: dict, kinds, fname: str, alpha: float = 0.05):
    titles = {"vol_up": "Vol up (calm→elevated→crisis)", "liquidity_shock": "Liquidity shock",
              "tail_lighter": "Tails lighten (t3 → t8)", "vol_down": "Vol down",
              "tail_heavier": "Tails fatten (t8 → t3)"}
    fig, axes = plt.subplots(2, len(kinds), figsize=(4 * len(kinds), 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})
    off = np.arange(-PRE, HORIZON)
    for j, kind in enumerate(kinds):
        c = curves[kind]
        ax = axes[0, j]
        for key, color, lab in [("naive_p95", ORANGE, "naive"), ("conf_p95", BLUE, "conformal")]:
            ax.plot(off, c[key], color=color, label=lab)
        ax.axhline(alpha, color=INK2, lw=0.9, ls="--")
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(f"{titles[kind]}  (n={c['n']})", fontsize=10)
        if j == 0:
            ax.set_ylabel("mean true P(breach), 95% VaR")
            ax.legend(loc="upper right")
            ax.text(HORIZON - 2, alpha, "nominal", fontsize=7.5, color=INK2, ha="right", va="bottom")

        ax = axes[1, j]
        for det, color, lab in [("abstain", BLUE, "abstention (50-day, 95%)"),
                                ("basel", AQUA, "Basel traffic light (250-day, 99%)"),
                                ("kupiec", YELLOW, "Kupiec POF (250-day, 95%)")]:
            ax.step(np.arange(HORIZON), c[f"detect_{det}"], where="post", color=color, label=lab)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_xlabel("days since true regime shift")
        if j == 0:
            ax.set_ylabel("share of shifts detected")
            ax.legend(loc="upper left", fontsize=7.5)
    fig.suptitle("Calibration breakdown around injected regime shifts (Monte Carlo)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
