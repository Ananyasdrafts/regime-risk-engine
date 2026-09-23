"""Regenerate every figure in docs/images/.

    python scripts/make_figures.py

The fixed-scenario figures are re-simulated (a fraction of a second); the Monte
Carlo figures read artifacts/figure_inputs.pkl from scripts/run_experiment.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from regimerisk import plots
from regimerisk.evaluate import run_path
from regimerisk.generator import fixed_scenario

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images"
INPUTS = ROOT / "artifacts" / "figure_inputs.pkl"


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    if not INPUTS.exists():
        raise SystemExit("run scripts/run_experiment.py first")
    with open(INPUTS, "rb") as f:
        d = pickle.load(f)

    path = fixed_scenario()
    res = run_path(path)
    plots.sanity_plot(path, res, IMAGES / "fixed_scenario.png")
    plots.var_band_plot(path, res, IMAGES / "var_bands.png")
    plots.coverage_plot(d["coverage"], path, res, IMAGES / "coverage.png")
    plots.risk_coverage_plot(d["risk_coverage"], d["oracle"], IMAGES / "risk_coverage.png")
    plots.event_study_plot(d["curves"], ["vol_up", "liquidity_shock", "tail_lighter"],
                           IMAGES / "event_study.png", d["alpha"])
    print(f"wrote figures to {IMAGES.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
