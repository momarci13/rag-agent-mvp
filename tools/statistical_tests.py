"""Statistical backtesting tests shared by the risk-validation gates.

Replaces v1's ad hoc flat exception-rate/count thresholds with the standard
Kupiec (1995) Proportion-of-Failures likelihood-ratio test, and a
CDF-threshold traffic-light zone classification in the same spirit as
Basel's VaR backtesting approach.

This module also carries the credit-side calibration tests (Jeffreys and
exact-binomial PD tests, an LGD/CCF mean-drift test) and the FRTB market-risk
P&L-attribution zone classification used by the deeper EBA/ECB-aligned gates.

These are textbook implementations, not a reproduction of any specific
regulator's exact published tables -- see the DRAFT/illustrative disclaimers
in agents/risk_schemas.py.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from scipy import stats


def kupiec_pof_test(exceptions: int, observations: int, expected_rate: float) -> dict:
    """Kupiec's Proportion-of-Failures likelihood-ratio test.

    Tests whether the observed exception rate is statistically consistent
    with ``expected_rate`` under a binomial model. Returns the LR statistic
    (chi-squared, 1 df) and its p-value; a small p-value combined with
    ``observed_rate > expected_rate`` indicates the observed exception rate
    is significantly worse than tolerated, not just numerically higher.
    """
    if observations <= 0:
        raise ValueError("observations must be positive")
    if not 0.0 < expected_rate < 1.0:
        raise ValueError("expected_rate must be in (0, 1)")
    if not 0 <= exceptions <= observations:
        raise ValueError("exceptions must be between 0 and observations")

    n, x, p = observations, exceptions, expected_rate
    observed_rate = x / n

    def _log_term(count: int, prob: float) -> float:
        # count == 0 contributes 0 to the log-likelihood regardless of prob,
        # avoiding a log(0) evaluation in the degenerate all-pass/all-fail case.
        if count == 0:
            return 0.0
        return count * math.log(prob)

    log_l_null = _log_term(x, p) + _log_term(n - x, 1.0 - p)
    log_l_alt = _log_term(x, observed_rate) + _log_term(n - x, 1.0 - observed_rate)
    lr_stat = max(-2.0 * (log_l_null - log_l_alt), 0.0)
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    return {
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "observed_rate": observed_rate,
        "expected_rate": p,
        "exceptions": x,
        "observations": n,
    }


def traffic_light_zone(exceptions: int, observations: int, expected_rate: float) -> Literal["green", "yellow", "red"]:
    """Classify a backtesting result into a green/yellow/red zone using the
    same cumulative-binomial-probability methodology Basel's VaR
    traffic-light approach is derived from (CDF thresholds at 95% / 99.99%),
    generalized to any (observations, expected_rate) pair rather than only
    n=250, p=0.01."""
    if observations <= 0:
        raise ValueError("observations must be positive")
    if not 0.0 < expected_rate < 1.0:
        raise ValueError("expected_rate must be in (0, 1)")

    cumulative_prob = float(stats.binom.cdf(exceptions, observations, expected_rate))
    if cumulative_prob < 0.95:
        return "green"
    if cumulative_prob < 0.9999:
        return "yellow"
    return "red"


# Basel VaR back-testing "plus factor" add-on to the internal-model multiplier
# for a 250-day / 99% window (green 0-4, yellow 5-9, red 10+). Red returns
# None: the internal model can no longer be used for that book without a
# supervisory add-on / standardised-approach fallback.
_PLUS_FACTOR_BY_EXCEPTIONS: dict[int, float] = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}


def backtesting_plus_factor(exceptions: int, zone: str) -> float | None:
    """Return the multiplier plus-factor implied by a back-testing result.

    ``zone`` is a value from :func:`traffic_light_zone`. Green -> 0.0, yellow
    -> the Basel stepwise add-on by exception count, red -> ``None`` (model
    disallowed / capital add-on determined by the supervisor).
    """
    if zone == "green":
        return 0.0
    if zone == "red":
        return None
    return _PLUS_FACTOR_BY_EXCEPTIONS.get(exceptions, 0.85)


def jeffreys_test(observed_defaults: int, obligors: int, predicted_pd: float) -> dict:
    """One-sided Jeffreys PD calibration test for a single rating grade.

    Uses the Jeffreys (Beta(0.5, 0.5)) prior; the posterior after observing
    ``observed_defaults`` defaults among ``obligors`` obligors is
    ``Beta(x + 0.5, n - x + 0.5)``. The p-value is the posterior probability
    that the true default rate is at or below ``predicted_pd``; a *small*
    p-value means the grade's realised default rate is significantly above the
    assigned PD (calibration too optimistic).
    """
    if obligors < 0 or observed_defaults < 0 or observed_defaults > obligors:
        raise ValueError("need 0 <= observed_defaults <= obligors")
    if not 0.0 <= predicted_pd <= 1.0:
        raise ValueError("predicted_pd must be in [0, 1]")
    observed_rate = observed_defaults / obligors if obligors else 0.0
    if obligors == 0:
        p_value = 1.0
    else:
        pd_clamped = min(max(predicted_pd, 1e-9), 1.0 - 1e-9)
        a = observed_defaults + 0.5
        b = obligors - observed_defaults + 0.5
        p_value = float(stats.beta.cdf(pd_clamped, a, b))
    return {
        "test": "jeffreys",
        "p_value": p_value,
        "observed_rate": observed_rate,
        "predicted_pd": predicted_pd,
        "obligors": obligors,
        "observed_defaults": observed_defaults,
    }


def binomial_pd_test(observed_defaults: int, obligors: int, predicted_pd: float) -> dict:
    """Exact one-sided binomial PD calibration test (portfolio or grade level).

    p-value = P(X >= observed_defaults) for X ~ Binomial(obligors,
    predicted_pd). Small p-value ⇒ realised defaults significantly exceed the
    number implied by the assigned PD.
    """
    if obligors < 0 or observed_defaults < 0 or observed_defaults > obligors:
        raise ValueError("need 0 <= observed_defaults <= obligors")
    if not 0.0 <= predicted_pd <= 1.0:
        raise ValueError("predicted_pd must be in [0, 1]")
    observed_rate = observed_defaults / obligors if obligors else 0.0
    if obligors == 0:
        p_value = 1.0
    else:
        p_value = float(stats.binom.sf(observed_defaults - 1, obligors, predicted_pd))
    return {
        "test": "binomial",
        "p_value": p_value,
        "observed_rate": observed_rate,
        "predicted_pd": predicted_pd,
        "obligors": obligors,
        "observed_defaults": observed_defaults,
    }


def normal_mean_test(
    observed_mean: float,
    predicted_mean: float,
    n: int | None,
    sample_std: float | None = None,
    *,
    tolerance: float = 0.10,
) -> dict:
    """Test whether a realised LGD/CCF mean is worse than predicted.

    With ``n >= 2`` and a positive ``sample_std`` a one-sided t-test is used
    (``H1: observed_mean > predicted_mean``). Otherwise it falls back to an
    absolute tolerance band: ``within_tolerance`` is ``abs(diff) <= tolerance``
    and ``p_value`` is ``None``.

    ``underestimation`` is ``observed_mean > predicted_mean`` (the model would
    have under-provisioned).
    """
    diff = observed_mean - predicted_mean
    underestimation = diff > 0.0
    if n is not None and n >= 2 and sample_std is not None and sample_std > 0.0:
        t_stat = diff / (sample_std / math.sqrt(n))
        p_value = float(stats.t.sf(t_stat, df=n - 1))
        return {
            "test": "t_test",
            "p_value": p_value,
            "t_statistic": float(t_stat),
            "observed_mean": observed_mean,
            "predicted_mean": predicted_mean,
            "n": n,
            "underestimation": underestimation,
            "within_tolerance": abs(diff) <= tolerance,
            "tolerance": tolerance,
        }
    return {
        "test": "tolerance_band",
        "p_value": None,
        "t_statistic": None,
        "observed_mean": observed_mean,
        "predicted_mean": predicted_mean,
        "n": n,
        "underestimation": underestimation,
        "within_tolerance": abs(diff) <= tolerance,
        "tolerance": tolerance,
    }


def pla_statistics(
    hypothetical_pnl: Sequence[float], risk_theoretical_pnl: Sequence[float]
) -> dict:
    """Spearman rank correlation and two-sample Kolmogorov-Smirnov statistic
    between the hypothetical and risk-theoretical P&L series (FRTB PLA test)."""
    if len(hypothetical_pnl) < 2 or len(risk_theoretical_pnl) < 2:
        raise ValueError("need at least 2 observations in each P&L series")
    spearman = float(stats.spearmanr(hypothetical_pnl, risk_theoretical_pnl).statistic)
    ks = float(stats.ks_2samp(hypothetical_pnl, risk_theoretical_pnl).statistic)
    return {"spearman": spearman, "ks": ks}


def pla_test(
    spearman: float | None,
    ks: float | None,
    *,
    spearman_green: float = 0.80,
    spearman_amber: float = 0.70,
    ks_green: float = 0.09,
    ks_amber: float = 0.12,
) -> Literal["green", "amber", "red"] | None:
    """FRTB P&L-attribution zone. ``None`` when the statistics are missing.

    green: Spearman >= ``spearman_green`` and KS <= ``ks_green``.
    amber: Spearman >= ``spearman_amber`` and KS <= ``ks_amber``.
    red:   otherwise -- the desk cannot use the internal model approach.
    """
    if spearman is None or ks is None:
        return None
    if spearman >= spearman_green and ks <= ks_green:
        return "green"
    if spearman >= spearman_amber and ks <= ks_amber:
        return "amber"
    return "red"
