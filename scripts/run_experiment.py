"""Run the Monte Carlo experiment and write every number the README quotes.

    python scripts/run_experiment.py [--paths 200] [--seed 1]

Writes docs/experiment_results.csv (per-shift detection table), docs/results.json
(headline numbers, backtests, ES, sweeps) and artifacts/ (inputs for make_figures.py).
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from regimerisk.backtest import christoffersen_ind, kupiec_pof
from regimerisk.evaluate import (
    EVAL_START,
    PRE,
    SEVERE,
    Config,
    coverage_by_level,
    event_curves,
    oracle_risk_coverage,
    risk_coverage,
    run_path,
    shift_events,
    steady_state_mask,
    summarise_events,
)
from regimerisk.generator import monte_carlo

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ARTIFACTS = ROOT / "artifacts"


def backtest_table(results):
    """Pooled breach rates plus the share of paths on which each test rejects at 5%."""
    rows = []
    for model in ("naive", "conf"):
        for tag, a in (("95", 0.05), ("99", 0.01), ("975", 0.025)):
            per_path = [r[f"{model}_breach{tag}"][EVAL_START:] for r in results]
            rows.append({
                "model": "conformal" if model == "conf" else "naive", "level": 1 - a,
                "breach_rate": kupiec_pof(np.concatenate(per_path), a)["rate"],
                "kupiec_reject_share": float(np.mean([kupiec_pof(b, a)["p_value"] < 0.05
                                                      for b in per_path])),
                "christoffersen_reject_share": float(np.mean([christoffersen_ind(b)["p_value"] < 0.05
                                                              for b in per_path])),
            })
    return pd.DataFrame(rows)


def es_table(paths, results):
    rows = []
    for model in ("naive", "conf"):
        ratio_all, ratio_steady = [], []
        for p, r in zip(paths, results):
            ratio = r[f"{model}_es975"] / r["true_es975"]
            ratio_all.append(ratio[EVAL_START:])
            ratio_steady.append(ratio[steady_state_mask(p)])
        ratio_all = np.concatenate(ratio_all)
        rows.append({"model": "conformal" if model == "conf" else "naive",
                     "es_over_true_median_all": float(np.nanmedian(ratio_all)),
                     "es_over_true_median_steady": float(np.nanmedian(np.concatenate(ratio_steady))),
                     "share_days_es_understated_20pct": float(np.nanmean(ratio_all < 0.8))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    cfg = Config()

    print(f"simulating {args.paths} paths ...")
    paths = monte_carlo(args.paths, seed=args.seed)
    results = [run_path(p, cfg) for p in paths]
    ev_list, traces = [], []
    for i, (p, r) in enumerate(zip(paths, results)):
        e, tr = shift_events(p, r, cfg)
        ev_list.append(e.assign(path=i))
        traces += tr
    events = pd.concat(ev_list, ignore_index=True)
    summary = summarise_events(events)
    curves = event_curves(traces, events)

    steady = [steady_state_mask(p) for p in paths]
    false_alarm = {k: float(np.mean(np.concatenate([r[k][m] for r, m in zip(results, steady)])))
                   for k in ("abstain", "basel", "kupiec")}

    print("coverage and risk-coverage sweeps ...")
    cov = coverage_by_level(paths[:60], [0.80, 0.85, 0.90, 0.95, 0.975, 0.99], cfg)
    rc = risk_coverage(paths, results, [3, 4, 5, 6, 7, 8, 10, 12], cfg)
    oracle = oracle_risk_coverage(results, "conf_p95", np.linspace(0.75, 1.0, 26), cfg)
    backtests = backtest_table(results)
    es = es_table(paths, results)

    harmful = summary.set_index("kind").loc["all_harmful"]
    rc_c = rc[rc.model == "conformal"]
    no_abstain = rc_c[rc_c.k_enter.isna()].iloc[0]
    k_default = rc_c[rc_c.k_enter == cfg.k_enter].iloc[0]
    vol_up_p = curves["vol_up"]["conf_p95"]
    p_all = np.concatenate([r["conf_p95"][EVAL_START:] for r in results])
    flag_all = np.concatenate([r["abstain"][EVAL_START:] for r in results])
    headline = {
        "n_paths": args.paths,
        "n_shifts_scored": len(events),
        "n_harmful_shifts": int(harmful.n),
        "harmful_definition": f">= {SEVERE} expected excess 95% VaR breaches within the horizon",
        "abstention_detect_rate_harmful": float(harmful.abstain_detect_rate),
        "abstention_median_lag_harmful": float(harmful.abstain_median_lag),
        "basel_detect_rate_harmful": float(harmful.basel_detect_rate),
        "basel_median_lag_harmful": float(harmful.basel_median_lag),
        "kupiec_detect_rate_harmful": float(harmful.kupiec_detect_rate),
        "abstention_first_rate_harmful": float(harmful.abstain_first_rate),
        "abstention_median_lead_vs_basel_days": float(harmful.median_lead_vs_basel),
        "false_alarm_share_steady_state": false_alarm,
        "vol_up_peak_true_p95": float(np.nanmax(vol_up_p)),
        "vol_up_days_until_true_p95_below_6pct": np.argmax(vol_up_p[PRE:] < 0.06).item(),
        "mean_true_p95_on_abstained_days": float(p_all[flag_all].mean()),
        "mean_true_p95_on_issued_days": float(p_all[~flag_all].mean()),
        "danger_rate_no_abstention": float(no_abstain.danger_rate),
        "danger_rate_default_k": float(k_default.danger_rate),
        "answered_default_k": float(k_default.answered),
    }

    out = {
        "config": asdict(cfg),
        "headline": headline,
        "backtests": backtests.to_dict(orient="records"),
        "expected_shortfall": es.to_dict(orient="records"),
        "coverage_by_level": cov.to_dict(orient="records"),
        "risk_coverage": rc.to_dict(orient="records"),
    }
    (DOCS / "results.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    summary.round(4).to_csv(DOCS / "experiment_results.csv", index=False)
    with open(ARTIFACTS / "figure_inputs.pkl", "wb") as f:
        pickle.dump({"coverage": cov, "risk_coverage": rc, "oracle": oracle, "curves": curves,
                     "alpha": cfg.alpha}, f)

    pd.set_option("display.width", 200)
    print("\n== headline ==")
    print(json.dumps(headline, indent=2))
    print("\n== detection by shift kind ==")
    print(summary.round(3).T.to_string())
    print("\n== backtests ==")
    print(backtests.round(4).to_string(index=False))
    print("\n== expected shortfall ==")
    print(es.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
