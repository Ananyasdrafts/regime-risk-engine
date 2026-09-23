import numpy as np

from regimerisk.backtest import basel_flags, christoffersen_ind, kupiec_lr, kupiec_pof


def test_kupiec_lr_is_zero_at_nominal_rate():
    assert np.isclose(kupiec_lr(25, 2500, 0.01), 0.0)


def test_kupiec_rejects_doubled_breach_rate():
    breach = np.zeros(2500)
    breach[::50] = 1.0  # 2% breaches against a 1% model
    assert kupiec_pof(breach, 0.01)["p_value"] < 0.01


def test_christoffersen_detects_clustering():
    rng = np.random.default_rng(0)
    iid = (rng.random(2000) < 0.05).astype(float)
    clustered = np.zeros(2000)
    for s in range(0, 2000, 200):
        clustered[s : s + 10] = 1.0  # same 5% rate, all in bursts
    assert christoffersen_ind(iid)["p_value"] > 0.01
    assert christoffersen_ind(clustered)["p_value"] < 1e-6


def test_basel_yellow_zone_needs_five_breaches_in_250_days():
    breach = np.zeros(400)
    breach[[260, 280, 300, 320]] = 1.0
    assert not basel_flags(breach).any()
    breach[340] = 1.0
    flags = basel_flags(breach)
    assert not flags[339] and flags[340]
