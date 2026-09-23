import numpy as np
from scipy.stats import t as student_t

from regimerisk.evaluate import (
    Config,
    run_path,
    shift_events,
    summarise_events,
    true_breach_prob,
    true_es,
)
from regimerisk.generator import fixed_scenario, std_t


def test_true_breach_prob_at_true_quantile_equals_alpha():
    nu, mu, sigma = 4.0, 0.001, 0.02
    var = -(mu + sigma * np.sqrt((nu - 2) / nu) * student_t.ppf(0.01, nu))
    assert np.isclose(true_breach_prob(var, mu, sigma, nu), 0.01)


def test_true_es_matches_monte_carlo():
    nu, mu, sigma = 4.0, 0.0, 0.02
    r = mu + sigma * std_t(nu, 2_000_000, np.random.default_rng(0))
    losses = np.sort(-r)[::-1][: int(0.025 * len(r))]
    assert np.isclose(true_es(0.025, mu, sigma, nu), losses.mean(), rtol=0.02)


def test_pipeline_on_fixed_scenario():
    path = fixed_scenario()
    res = run_path(path, Config())
    assert np.isnan(res["conf_var95"][:250]).all() and not np.isnan(res["conf_var95"][250:]).any()
    events, traces = shift_events(path, res)
    assert set(events.kind) == {"vol_up", "vol_down", "liquidity_shock", "tail_heavier",
                                "tail_lighter"}
    assert len(traces) == len(events)
    assert not summarise_events(events).empty
