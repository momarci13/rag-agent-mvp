"""Unit tests for tools/statistical_tests.py against known reference values.

The traffic_light_zone bands at n=250, p=0.01 reproduce Basel's published
VaR backtesting traffic-light table (green 0-4, yellow 5-9, red 10+) as a
by-product of the correct CDF-threshold methodology -- verified below.
"""

import pytest

from tools.statistical_tests import (
    backtesting_plus_factor,
    binomial_pd_test,
    jeffreys_test,
    kupiec_pof_test,
    normal_mean_test,
    pla_statistics,
    pla_test,
    traffic_light_zone,
)


def test_kupiec_pof_test_matches_the_null_when_observed_equals_expected():
    result = kupiec_pof_test(exceptions=25, observations=2500, expected_rate=0.01)
    assert result["observed_rate"] == pytest.approx(0.01)
    assert result["lr_statistic"] == pytest.approx(0.0, abs=1e-6)
    assert result["p_value"] == pytest.approx(1.0, abs=1e-6)


def test_kupiec_pof_test_flags_a_clear_excess():
    result = kupiec_pof_test(exceptions=25, observations=250, expected_rate=0.01)
    assert result["observed_rate"] == pytest.approx(0.10)
    assert result["p_value"] < 0.001


def test_kupiec_pof_test_handles_zero_exceptions():
    result = kupiec_pof_test(exceptions=0, observations=250, expected_rate=0.01)
    assert result["observed_rate"] == 0.0
    assert result["lr_statistic"] > 0
    assert 0.0 <= result["p_value"] <= 1.0


def test_kupiec_pof_test_handles_all_exceptions():
    result = kupiec_pof_test(exceptions=250, observations=250, expected_rate=0.01)
    assert result["observed_rate"] == 1.0
    assert result["p_value"] < 1e-10


def test_kupiec_pof_test_validates_inputs():
    with pytest.raises(ValueError):
        kupiec_pof_test(exceptions=1, observations=0, expected_rate=0.01)
    with pytest.raises(ValueError):
        kupiec_pof_test(exceptions=1, observations=10, expected_rate=1.5)
    with pytest.raises(ValueError):
        kupiec_pof_test(exceptions=20, observations=10, expected_rate=0.01)


# n=250, p=0.01 traffic-light bands per Basel's published VaR backtesting table.
_BASEL_250_1PCT_ZONES = {
    0: "green", 1: "green", 2: "green", 3: "green", 4: "green",
    5: "yellow", 6: "yellow", 7: "yellow", 8: "yellow", 9: "yellow",
    10: "red", 11: "red", 14: "red",
}


@pytest.mark.parametrize("exceptions,expected_zone", list(_BASEL_250_1PCT_ZONES.items()))
def test_traffic_light_zone_matches_basel_250_1pct_table(exceptions, expected_zone):
    assert traffic_light_zone(exceptions, 250, 0.01) == expected_zone


def test_traffic_light_zone_validates_inputs():
    with pytest.raises(ValueError):
        traffic_light_zone(1, 0, 0.01)
    with pytest.raises(ValueError):
        traffic_light_zone(1, 10, 1.0)


# ---------------- backtesting_plus_factor ----------------

@pytest.mark.parametrize(
    "exceptions,zone,expected",
    [
        (2, "green", 0.0),
        (5, "yellow", 0.40),
        (6, "yellow", 0.50),
        (7, "yellow", 0.65),
        (8, "yellow", 0.75),
        (9, "yellow", 0.85),
        (12, "red", None),
    ],
)
def test_backtesting_plus_factor(exceptions, zone, expected):
    assert backtesting_plus_factor(exceptions, zone) == expected


# ---------------- jeffreys_test / binomial_pd_test ----------------

def test_jeffreys_test_well_calibrated_grade_is_not_flagged():
    # 5 defaults / 1000 obligors == exactly the assigned PD of 0.5%.
    result = jeffreys_test(observed_defaults=5, obligors=1000, predicted_pd=0.005)
    assert result["observed_rate"] == pytest.approx(0.005)
    assert result["p_value"] > 0.10


def test_jeffreys_test_optimistic_grade_is_flagged():
    # 50 defaults / 1000 obligors == 5%, ten times the assigned 0.5% PD.
    result = jeffreys_test(observed_defaults=50, obligors=1000, predicted_pd=0.005)
    assert result["p_value"] < 0.01


def test_jeffreys_test_zero_obligors_is_inconclusive():
    result = jeffreys_test(observed_defaults=0, obligors=0, predicted_pd=0.01)
    assert result["p_value"] == 1.0
    assert result["observed_rate"] == 0.0


def test_binomial_pd_test_agrees_with_jeffreys_on_direction():
    ok = binomial_pd_test(observed_defaults=5, obligors=1000, predicted_pd=0.005)
    bad = binomial_pd_test(observed_defaults=50, obligors=1000, predicted_pd=0.005)
    assert ok["p_value"] > 0.10
    assert bad["p_value"] < 0.01


def test_pd_tests_validate_inputs():
    with pytest.raises(ValueError):
        jeffreys_test(observed_defaults=11, obligors=10, predicted_pd=0.01)
    with pytest.raises(ValueError):
        binomial_pd_test(observed_defaults=1, obligors=10, predicted_pd=1.5)


# ---------------- normal_mean_test ----------------

def test_normal_mean_test_t_test_flags_lgd_underestimation():
    result = normal_mean_test(observed_mean=0.45, predicted_mean=0.30, n=100, sample_std=0.20)
    assert result["test"] == "t_test"
    assert result["underestimation"] is True
    assert result["p_value"] < 0.01


def test_normal_mean_test_t_test_no_flag_when_observed_below_predicted():
    result = normal_mean_test(observed_mean=0.28, predicted_mean=0.30, n=100, sample_std=0.20)
    assert result["underestimation"] is False
    assert result["p_value"] > 0.10


def test_normal_mean_test_falls_back_to_tolerance_band_without_std():
    within = normal_mean_test(observed_mean=0.32, predicted_mean=0.30, n=None)
    outside = normal_mean_test(observed_mean=0.45, predicted_mean=0.30, n=None)
    assert within["test"] == "tolerance_band" and within["within_tolerance"] is True
    assert outside["within_tolerance"] is False and outside["p_value"] is None


# ---------------- pla_test / pla_statistics ----------------

@pytest.mark.parametrize(
    "spearman,ks,expected",
    [
        (0.90, 0.05, "green"),
        (0.80, 0.09, "green"),
        (0.79, 0.09, "amber"),
        (0.75, 0.12, "amber"),
        (0.70, 0.121, "red"),
        (0.60, 0.30, "red"),
    ],
)
def test_pla_test_zone_boundaries(spearman, ks, expected):
    assert pla_test(spearman, ks) == expected


def test_pla_test_none_when_statistics_missing():
    assert pla_test(None, 0.05) is None
    assert pla_test(0.9, None) is None


def test_pla_statistics_perfectly_aligned_series():
    stats_out = pla_statistics([1, 2, 3, 4, 5, 6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert stats_out["spearman"] == pytest.approx(1.0)
    assert stats_out["ks"] == pytest.approx(0.0)
    with pytest.raises(ValueError):
        pla_statistics([1.0], [1.0])
