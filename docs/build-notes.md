# Build notes

An honest log of how this came together, what broke, and what I changed my mind about.

## the starting question

I wanted to carry the idea from MedMaps and Vigil (models that know when they can't be
trusted) into market risk. The plan was simple: simulate markets with regime shifts at
known times, put a conformal wrapper and an abstention rule around a standard VaR model,
and show the abstention catching the shift before a real backtest would.

The one decision I'm most glad I made early: score everything against the *true* breach
probability. Every simulated day carries its true volatility and tail shape, so for any
VaR number I can compute exactly how likely it was to be breached. Without that, I'd be
judging a noisy alarm with another noisy count.

## what broke along the way

- **My volatility test failed for the crisis regime.** Realised volatility came out 26%
  below target. It wasn't a bug: crisis returns have t(4) tails on top of a very
  persistent GARCH, so the sample variance is itself extremely heavy-tailed and one path
  tells you little. Averaged over seeds the generator is unbiased (0.96 to 0.99 of
  target). The test now pools variance across 20 seeds.
- **"Tail-only" shifts were two opposite things.** I'd grouped every shift that changes
  tail shape without changing volatility. Splitting them showed they behave in opposite
  ways: fattening the tails makes the 95% VaR too *cautious*, lightening them makes the
  conformal VaR too *tight*. That second case turned into one of the main findings.
- **I measured "recovered" two different ways.** For volatility jumps I'd counted days
  until the true breach chance fell under 6%, for lighter tails until it fell under about
  5%, which made the tail case look twice as slow (80 days instead of 40). Now there's one
  threshold, set in `run_experiment.py`, and every number in the README is written by the
  script instead of read off a plot.
- **I was flattering the Basel test.** Its 250-day window is often already in the yellow
  zone when a shift happens, so it looked like it detected instantly. Now each alarm is
  only scored on shifts where it wasn't already firing.

## the result I didn't expect

I expected abstention to catch shifts early. Against the regulatory backtests it does
(51% of harmful shifts caught vs 7.5% for Basel), but the risk-coverage curve came out
flat, and the days abstention flagged were slightly *safer* than the days it didn't.

Before believing that, I checked it wasn't a bug by lining up the true breach probability
and the abstention flag around every volatility spike. The mechanism is clean. The model
is badly wrong for about 22 trading days (24% breach chance on day one), the EWMA heals
itself on roughly an 11-day half-life, and a 50-day breach count needs about 23 trading
days of evidence. By the time the alarm is sure, the problem is mostly gone, and ACI has by
then overcorrected into being slightly too cautious.

## what I'd do differently

The breach count was the obvious monitor because it's what backtests use, and that's
exactly its weakness: one bit per day. The next version would monitor the full
standardised residual, which uses every observation. The ground-truth setup makes that
comparison easy to run honestly, which is really the point of the project.
