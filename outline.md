# Strategy Outline

---

Important:
* Simulate both puts and calls opened on every day
* Iteratively simulate positions and their PL
* Allow for fractional trading of SPY
* ATM strikes, 30-day expiries

---

### Strategy steps

We will simulate both long and short gamma: We will simulate long position on call and put strangles and short positions on the same.

We will open a zero-delta straddle, either as a long or short position. Then, we will rebalance daily to return to a delta of zero. We assume that we can trade the underlying, in this case chosen as SPY, in precise fractions. For our purposes we will open a position at-the-money.

We will test individual contracts with expiry of 2 weeks, **1 month**, and 2 months (we will initially focus on 1 month expiry). We will close all positions one week prior to expiry.
* We choose the closest 30 day exp, that is, approximately 20 rows in the data.
* We will also choose different expiries and strikes every day.

We will then analyze individual trades as well as aggregating all of the daily opened contracts, comparing using puts and calls, and compare strategies where we have limits (thereby not opening new positions on certain days)

Depending on whether our calculated IV is greater or less than the IV in the data, we will take a long or short gamma position.


---

### Considerations
* Daily rebalance: SPY price
* Option bid/ask price - using the midpoint of best bid/best offer data
* Thresholds, depending on our IV vs the given IV (calculation on WRDS website), to either sell options to go short gamma or to buy options to go long gamma
* Metrics - theoretical value metrics for the market (over or underpriced)
* Papers to see how often a hedge is necessary
* Simulate positions for strikes at every expiration.
* We will assume a constant, scalable contract sizing (ie. a beginning value of one contract each)
* We will analyze the strategy with respect to individual contract combinations, which we can then consolidate. We will set the capital arbitrarily, and then we can modify our strategy parameters (ie. IV for position side, RSI/regression) to decide which trades were made out of the simulations.
* Trading cost assumptions/leverage/borrowing costs




---

### To-do

Confirm data - are the missing values in calls contained under puts? - Nope
Graph of ATM deltas for expiry, IV vs given IV, something else
Benchmark - hf returns or just hold the market - Figure out later
Calculate black scholes to fill in; Consider IV reversal as a closing strat? - Will figure out how to fill in, stick with basic closing strategy
