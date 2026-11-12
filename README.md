# Chart forecaster for pumpfun-top5-forecast-01

`main.py` (Python 3, standard library only) reads the house's marks in
`$MAKEFUN_INPUT_DIR` and writes one `forecast.json` to `$MAKEFUN_OUTPUT_DIR`:
per-place probabilities for the five largest pump.fun coins at the answer
reads, an ordered top 5, and per-place market-cap size distributions.

## Approach

Nothing is hard-coded to one contest snapshot: every parameter is measured
from the history in the input, so the same program runs on a 2-hour
rehearsal and on an 11-hour history whose reads sit 7-13 hours past the cut.

1. **Dynamics from the marks.** For each venue (bonding curve, graduated
   pool) the program estimates drift, volatility and a sudden-collapse
   ("dump") hazard of log10 market cap from consecutive-mark changes.
   Coins that disappear from a full 200-row chart are counted as censored
   steps down to just under that chart's floor - without them, every
   statistic is biased toward survivors. Estimates are shrunk toward
   conservative priors, so thin samples stay sane.
2. **Incumbents as random walks.** Each plausible coin starts from its
   last observed cap (discounted for staleness) and evolves with its own
   recent slope - shrunk by the amount of data behind it - fading into an
   age- and venue-dependent prior drift. Excess cap above the chart's #5
   level bleeds back, curve coins that cross the measured graduation cap
   switch to pool dynamics, and dumps strike at the measured hazard.
3. **The unborn and the uncharted.** Launches keep arriving at the rate
   the history's `eligible` counter implies. A measured fraction of them
   are contenders that reach at least half of the #5 cap; simulated as a
   Poisson stream (widened to also stand for pre-cut coins that never
   charted), each rockets to a peak drawn from the history's contender
   peaks within tens of minutes, then decays with a measured half-life.
   These coins are never named - their mass stays in each place's unnamed
   share, which is exactly how the scorer pays when one of them takes a
   place.
4. **Frequencies become the forecast.** Thousands of seeded simulations
   rank the whole field at every answer read. Place frequencies (with a
   mild shrink toward the unnamed share, since the parameters are
   themselves estimates) become `probabilities`; stagnant mid-chart coins
   that cannot hold a place are named at p=0 so the unnamed share is not
   diluted by them; the simulated cap distribution at each place, mixed
   with a small uniform component, becomes `sizes`; the modal ranking
   becomes `top5`. A fixed seed and sorted iteration make two runs of the
   program byte-identical.
