import numpy as np

from regimerisk.backtest import kupiec_pof
from regimerisk.conformal import abstention_flags, adaptive_conformal_var
from regimerisk.generator import CALM, REGIMES, simulate


def test_conformal_var_is_calibrated_in_a_stationary_regime():
    path = simulate([(0, CALM)], 6000, seed=4)
    sigma_hat = np.full(6000, REGIMES[CALM].daily_vol)
    res = adaptive_conformal_var(path.returns, sigma_hat, alpha=0.05)
    assert abs(kupiec_pof(res.breach, 0.05)["rate"] - 0.05) < 0.01
    assert np.all(res.es[250:] >= res.var[250:])


def test_aci_widens_var_after_a_run_of_breaches():
    rng = np.random.default_rng(0)
    r = rng.standard_normal(600) * 0.01
    r[400:420] = -0.05  # a burst of losses far beyond the calibrated quantile
    res = adaptive_conformal_var(r, np.full(600, 0.01), alpha=0.05)
    assert res.alpha_t[421] < res.alpha_t[399]
    assert res.var[450] > 2 * res.var[399]
    assert res.breach[410:420].sum() == 0  # once widened, the burst stops breaching


def test_abstention_triggers_on_breach_cluster():
    breach = np.zeros(300)
    breach[200:210] = 1.0
    flags = abstention_flags(breach, alpha=0.05, k_enter=6)
    assert not flags[:205].any() and flags[206]


def test_abstention_is_quiet_when_calibrated():
    breach = (np.random.default_rng(1).random(20000) < 0.05).astype(float)
    assert abstention_flags(breach, alpha=0.05, k_enter=6).mean() < 0.1


def test_abstention_uses_only_past_breaches():
    breach = np.zeros(100)
    breach[50:56] = 1.0
    flags = abstention_flags(breach, alpha=0.05, k_enter=6)
    assert not flags[55] and flags[56]
