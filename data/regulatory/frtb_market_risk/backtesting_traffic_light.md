# FRTB / Basel VaR back-testing traffic-light (representative structure)

> **PLACEHOLDER / NOT VERBATIM.** Hand-authored summary of the publicly
> known structure of the Basel supervisory back-testing framework, read
> with BCBS d457. Not a reproduction of any regulator's exact published
> tables; the implementation in `tools/statistical_tests.py` derives the
> zones from the cumulative binomial probability rather than copying a
> table.

## Zones

Over a rolling 250-business-day window, the number of days on which the loss
exceeded the 99% one-day VaR is counted. The result maps to:

| Zone   | Meaning                                    | Multiplier plus-factor |
| ------ | ------------------------------------------- | ---------------------- |
| green  | consistent with a sound model              | 0.00                  |
| amber  | more exceptions than expected; investigate | 0.40 - 0.85 (by count)|
| red    | model deficiency presumed                  | supervisory add-on / SA fallback |

The generalisation used here classifies any `(observations, expected_rate)`
pair by whether the cumulative binomial probability of at most the observed
number of exceptions is below 0.95 (green), below 0.9999 (amber) or above
(red). At n=250, p=0.01 this reproduces the familiar green 0-4 / amber 5-9 /
red 10+ bands.

## FRTB extension

Under FRTB, back-testing is performed at the **desk** level and at **both**
the 97.5% and 99% confidence levels; the worst outcome drives escalation.
Persistent amber/red at desk level, together with a P&L-attribution
failure, removes internal-model eligibility for that desk.
