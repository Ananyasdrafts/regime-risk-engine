import numpy as np
from scipy.stats import kurtosis

from regimerisk.generator import (
    CALM,
    CRISIS,
    ELEVATED,
    FAT_TAIL,
    FIXED_SCHEDULE,
    FIXED_T,
    LIQUIDITY,
    REGIMES,
    classify_shift,
    monte_carlo,
    simulate,
)


def test_labels_and_shifts_match_schedule():
    path = simulate(FIXED_SCHEDULE, FIXED_T, seed=1)
    assert list(path.shifts["t"]) == [s for s, _ in FIXED_SCHEDULE[1:]]
    for start, k in FIXED_SCHEDULE:
        assert path.regime[start] == k


def test_vol_level_switches_at_labelled_timestamp():
    # Realised vol should match each regime's target. Crisis (t4 + persistent
    # GARCH) has a very heavy-tailed sample variance, so pool across seeds.
    sched = [(0, CALM), (5000, CRISIS)]
    paths = [simulate(sched, 10000, seed=s) for s in range(20)]
    for k, sl in [(CALM, slice(500, 5000)), (CRISIS, slice(5000, 10000))]:
        realised = np.sqrt(np.mean([p.returns[sl].var() for p in paths]))
        assert abs(realised / REGIMES[k].daily_vol - 1) < 0.1
    # The true conditional vol jumps at the boundary, not gradually.
    ratio = REGIMES[CRISIS].daily_vol / REGIMES[CALM].daily_vol
    assert np.allclose(paths[0].sigma[5000] / paths[0].sigma[4999], ratio, rtol=0.1)


def test_fat_tail_regime_has_same_vol_heavier_tails():
    calm = simulate([(0, CALM)], 20000, seed=3)
    fat = simulate([(0, FAT_TAIL)], 20000, seed=3)
    z_calm = (calm.returns - calm.mu) / calm.sigma
    z_fat = (fat.returns - fat.mu) / fat.sigma
    assert abs(z_calm.std() - 1) < 0.05 and abs(z_fat.std() - 1) < 0.1
    assert kurtosis(z_fat) > kurtosis(z_calm) + 1


def test_shift_classification():
    assert classify_shift(CALM, ELEVATED) == "vol_up"
    assert classify_shift(CRISIS, CALM) == "vol_down"
    assert classify_shift(CALM, FAT_TAIL) == "tail_heavier"
    assert classify_shift(FAT_TAIL, CALM) == "tail_lighter"
    assert classify_shift(CALM, LIQUIDITY) == "liquidity_shock"
    assert classify_shift(LIQUIDITY, CALM) == "liquidity_exit"


def test_monte_carlo_paths_have_shifts_after_calm_warmup():
    paths = monte_carlo(3, seed=5)
    assert all(len(p.shifts) >= 4 for p in paths)
    assert all((p.regime[:500] == CALM).all() for p in paths)
