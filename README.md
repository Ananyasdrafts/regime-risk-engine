# Regime Risk Engine

> **When does a risk model stop being trustworthy after a regime shift, and can adaptive
> calibration with abstention detect that failure earlier than standard backtesting?**

**Short answer:** most regime shifts do little harm. The ones that do can make the model
badly miscalibrated right away, and it takes roughly one to two months (20 to 40
trading days) to recover depending on what changed. Abstention catches more harmful
shifts within 150 days than standard backtesting, but it also fires more often, and
usually not before the model has already recovered on its own.

`status: v1 complete`

You can't answer this on real data. You don't know when the regime changed, and you don't
know what the true chance of a big loss was yesterday, so a backtest can only count
breaches after the fact. So I built a synthetic market where I know both: every shift is
labelled, and every day's true breach probability can be computed exactly. Then I can
watch a Value-at-Risk model go wrong day by day and time every alarm against the truth.

200 simulated markets, 3,000 trading days each, 1,791 regime shifts. Every number is in
[docs/results.json](docs/results.json), and all days are trading days. "Recovered" means
the true chance of breaching the 95% VaR, averaged over every shift of that kind, is back
under 6%.

## when does the model stop being trustworthy?

- **only some shifts matter.** 324 of the 1,791 shifts (18%) did real damage, meaning at
  least two more expected 95% VaR breaches than there should have been. Shifts where
  volatility falls or tails get heavier mostly make the model too cautious, not
  dangerous.
- **the harmful ones bite immediately.** When volatility jumps, the true chance of
  breaching the 95% VaR hits 24% on day one, nearly 5x what it should be.
- **for how long depends on what changed.** After a volatility jump the model fixes
  itself in 22 days (the naive model takes 35). When the tails get lighter at the same
  volatility, the conformal model is still calibrated on 250 days of the old heavy-tailed
  losses. It peaks at an 8% true breach chance against the naive model's 6.3%, takes 40
  days to recover against the naive model's 13, and stays worse than naive for 120 of the
  150 days after. After a single liquidity shock it stays overly cautious for about 190
  days, because that one loss sits in its calibration window.
- **and a passing backtest won't tell you.** Conformal improves long-run tail
  calibration. The Gaussian model's 99% VaR gets breached 1.98% of the time and fails
  Kupiec on 98.5% of paths, conformal gets 0.99% and passes on every path. It's still
  badly wrong for one to two months after harmful shifts. Averaged over years, those
  months disappear.

## can abstention catch it earlier than backtesting?

- **it catches far more than the backtests.** Of the 324 shifts that did real damage, abstention caught 51%
  within 150 days. The Basel traffic light caught 7.5% and a rolling Kupiec test 1.9%.
  When both fired, abstention was a median 25 days ahead of Basel, and 54 days ahead on
  the tail-lightening case.
- **it's not a like-for-like race, though.** Abstention looks at the last 50 days, the
  backtests at 250, and abstention raises more false alarms (it's on for 8.6% of days in
  settled regimes, Basel for 2.9%). Part of its lead just comes from firing more often.
  A fair comparison would match the false-alarm rates first.
- **but usually not before the damage.** After a volatility jump abstention needs about
  23 days to be sure (median), roughly the time the model takes to fix itself. So it
  tends to go off just as the problem is going away. Sitting out 9% of days barely changed how many
  badly wrong forecasts went out (5.39% to 5.34%).
- **the reason is how little each day tells you.** Abstention, like every backtest,
  counts breaches. That's one bit a day, and at 95% the bit is almost always zero, so any
  alarm built on it needs weeks of evidence. Whether that's too slow depends on the
  forecaster. This EWMA recovers in about a month; a slower model would leave an
  alarm more room to fire first.

![Event study: true breach probability around each shift, and how fast each alarm fires](docs/images/event_study.png)

The top row answers the first question: every shift lined up on the day it happened,
with the true breach probability around it. The bottom row answers the second: how many
shifts each alarm has caught by each day after.

## what I built

- **a synthetic market with labelled regimes.** Returns follow a regime-switching
  GARCH(1,1) with fat-tailed Student-t shocks. Five regimes: calm, elevated, crisis, a
  fat-tail regime with calm-level volatility but much heavier tails, and a short
  liquidity shock that opens with a 6-sigma drop. Every path keeps its true volatility
  and tail shape, so I can compute exactly how likely any VaR number was to be breached.
- **a deliberately boring baseline.** RiskMetrics EWMA volatility with Gaussian VaR and
  Expected Shortfall. It's simple on purpose: the experiment tests calibration and
  monitoring, not forecasting, so keeping the forecaster fixed and plain means any
  difference comes from the layers on top of it.
- **adaptive conformal VaR and ES** on top of it (ACI, Gibbs and Candès 2021). It
  calibrates the tail from the last 250 standardised losses and nudges its own target
  after every breach or quiet day.
- **an abstention rule.** Hold back the forecast when the last 50 days had implausibly
  many 95% VaR breaches (6 or more when 2.5 are expected, p ≈ 0.04), or when ACI has had
  to drift far from its nominal level.
- **the backtests a bank would actually run**, as the thing to beat: Kupiec
  proportion-of-failures, Christoffersen independence, and the Basel traffic light.

![A fixed scenario: price, returns, and true vs forecast volatility across regimes](docs/images/fixed_scenario.png)

## results

**tail calibration**

| | Gaussian EWMA | + adaptive conformal |
|---|---:|---:|
| 99% VaR breach rate (target 1%) | 1.98% | 0.99% |
| paths where Kupiec rejects the 99% VaR | 98.5% | 0% |
| predicted / true Expected Shortfall at 97.5% (median) | 0.82 | 1.05 |
| days where ES is understated by more than 20% | 44.7% | 17.0% |

ACI pins the long-run breach rate by design, so passing Kupiec is expected here, not
impressive. That's also why a passing backtest says little about whether the model is
right day to day.

**catching harmful shifts** (at least two extra expected 95% VaR breaches)

| alarm | caught within 150 days | median delay |
|---|---:|---:|
| abstention (50-day window) | 51% | 56 days |
| Basel traffic light (250-day window) | 7.5% | 88.5 days |
| Kupiec test (250-day window) | 1.9% | 68 days |

False alarms, share of days in settled regimes: abstention 8.6%, Basel 2.9%, Kupiec 0.4%.
The per-shift-type breakdown is in [docs/experiment_results.csv](docs/experiment_results.csv).

![99% VaR from each model on the fixed scenario, with abstention periods shaded](docs/images/var_bands.png)

A single liquidity shock (day 2000) keeps the conformal 99% VaR at about 5x its usual
level for 189 days, because that one loss stays in the calibration window.

## what I learned

- The right bar for an alarm isn't "faster than Basel", it's "faster than the model fixes
  itself". Beating the backtest was easy. Beating the model's own recovery is the hard,
  useful part, and counting breaches didn't get there against this model.
- "Is the model calibrated?" is really two questions: on average, and right now. Most
  backtests only answer the first. Ground truth is what let me separate them.
- Calibration has memory, and memory has a cost. The window that makes conformal robust
  is the same thing that makes it slow to forget an old regime.

## limitations

Everything here is simulated. That's the point, since ground truth is what makes the
measurement possible, but it means the results are properties of this simulated regime
process, not estimates of how often these things happen in real markets. It's one asset
with no correlation between assets, and the forecaster is EWMA only, so the timing
results are relative to how fast EWMA recovers. The abstention vs backtest comparison
also isn't matched on false-alarm rate yet.

## run it

```bash
pip install -e ".[dev]"
python scripts/run_experiment.py     # 200 simulated markets, about 40 seconds
python scripts/make_figures.py       # regenerates docs/images/
```

Tested and runs in CI. Each piece is a small module under `src/regimerisk/`: `generator`
(the synthetic market), `forecast` (EWMA), `conformal` (conformal VaR/ES and abstention),
`backtest` (Kupiec, Christoffersen, Basel) and `evaluate` (true breach probabilities and
the event study). The build log, including the dead ends, is in
[docs/build-notes.md](docs/build-notes.md).

## what's next

- **a monitor that uses every day, not one bit a day.** Watching the whole standardised
  residual (a rolling mean of z², or a check that forecast quantiles are uniform) should
  have far more power than counting breaches. The real question is whether anything can
  fire before the model heals itself, compared against the backtests at the same
  false-alarm rate this time.
- **a forecaster that heals at a different speed.** A refit GARCH(1,1) would change the
  window where early detection is even possible.
- **real returns.** No ground truth there, but I can check whether the same lag shows up
  around known stress periods like March 2020.
- **more than one asset.** Correlation regimes, where diversification disappears in a
  crisis, are where portfolio VaR really breaks.
