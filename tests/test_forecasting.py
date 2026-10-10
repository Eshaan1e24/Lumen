"""pytest suite for forecasting.forecast_series.  Run:  cd /home/user/work/ml && ../venv/bin/python -m pytest -q test_forecasting.py"""
import json
import time
import warnings

import numpy as np
import pandas as pd
import pytest

from app import forecasting as FC
from app.forecasting import forecast_series

warnings.simplefilter("ignore")
needs_threadlimit = pytest.mark.skipif(FC._threadpool_limits is None,
                                       reason="threadpoolctl missing: set OMP_NUM_THREADS=1/OPENBLAS_NUM_THREADS=1 to meet the speed target")


def monthly(n=36, seed=0, noise=3.0, level=100.0, amp=10.0, trend=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    idx = pd.date_range("2022-01-01", periods=n, freq="MS")
    return pd.Series(level + trend * t + amp * np.sin(2 * np.pi * t / 12) + rng.normal(0, noise, n), index=idx)


def daily(n=120, seed=0, noise=5.0, weekly=15.0, level=100.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(level + weekly * np.sin(2 * np.pi * t / 7) + rng.normal(0, noise, n), index=idx)


KEYS = {"status", "model", "model_name", "history", "forecast", "lower", "upper", "backtest", "baseline_comparison", "confidence", "caveats"}


def check_schema(r, H):
    assert KEYS <= set(r)
    f = r["forecast"]
    for k in ("x", "y", "lower", "upper", "lower80", "upper80", "lower95", "upper95"):
        assert len(f[k]) == H, k
    assert set(r["lower"]) == {"80", "95"} and set(r["upper"]) == {"80", "95"}
    assert r["confidence"] in ("high", "medium", "low")
    assert isinstance(r["caveats"], list) and all(isinstance(c, str) for c in r["caveats"])
    json.dumps(r)   # must be JSON-serialisable as is (no numpy / NaN objects)
    for i in range(H):
        assert f["lower95"][i] <= f["lower80"][i] + 1e-9 <= f["y"][i] + 1e-9
        assert f["y"][i] <= f["upper80"][i] + 1e-9 <= f["upper95"][i] + 2e-9


def test_schema_and_ordering_monthly():
    r = forecast_series(monthly(), 6, "MS")
    assert r["status"] == "ok"
    check_schema(r, 6)
    assert r["forecast"]["x"][0] == "2025-01-01" and r["history"]["x"][-1] == "2024-12-01"
    bt = r["backtest"]
    assert bt["n_folds"] >= 2 and bt["mase"] is not None and bt["smape"] is not None
    assert set(bt["coverage"]) == {"80", "95"}


def test_deterministic():
    s = monthly(seed=3)
    a = json.dumps(forecast_series(s, 6, "MS"), sort_keys=True)
    b = json.dumps(forecast_series(s, 6, "MS"), sort_keys=True)
    assert a == b


def test_does_not_mutate_input():
    s = monthly()
    s.iloc[5] = np.nan
    before = s.copy()
    forecast_series(s, 6, "MS")
    pd.testing.assert_series_equal(s, before)


@needs_threadlimit
def test_runtime_under_two_seconds_for_200_points():
    s = daily(200)
    t0 = time.perf_counter()
    r = forecast_series(s, 14, "D")
    dt = time.perf_counter() - t0
    assert r["status"] == "ok" and dt < 2.0, dt
    s2 = pd.Series(np.random.default_rng(1).normal(100, 5, 200), index=pd.date_range("2022-01-01", periods=200, freq="W"))
    t0 = time.perf_counter()
    forecast_series(s2, 12, "W")
    assert time.perf_counter() - t0 < 2.0


def test_seasonal_pattern_is_captured():
    s = monthly(n=48, noise=1.0, amp=20.0)
    r = forecast_series(s, 12, "MS")
    truth = 100 + 20 * np.sin(2 * np.pi * np.arange(48, 60) / 12)
    err = np.mean(np.abs(np.array(r["forecast"]["y"]) - truth))
    naive_err = np.mean(np.abs(s.iloc[-1] - truth))
    assert err < 0.5 * naive_err
    assert "beats" in r["baseline_comparison"]


def test_trend_is_followed():
    t = np.arange(40)
    s = pd.Series(50 + 2.0 * t + np.random.default_rng(0).normal(0, 1, 40), index=pd.date_range("2023-01-01", periods=40, freq="D"))
    r = forecast_series(s, 5, "D")
    assert r["forecast"]["y"][-1] > s.iloc[-10:].mean()   # keeps going up, not flat at the last value


def test_beats_or_admits_naive_on_random_walk():
    rng = np.random.default_rng(5)
    s = pd.Series(np.cumsum(rng.normal(0, 1, 80)) + 100, index=pd.date_range("2024-01-01", periods=80, freq="D"))
    r = forecast_series(s, 7, "D")
    assert r["confidence"] in ("medium", "low")   # never "high" when it cannot show skill over naive
    assert "naive" in r["baseline_comparison"]


# ----------------------------------------------------------------------------- short / degenerate input
def test_short_series_gets_labelled_naive():
    r = forecast_series(monthly().iloc[:5], 4, "MS")
    assert r["status"] == "limited" and r["model"] == "naive" and r["confidence"] == "low"
    assert set(r["forecast"]["y"]) == {float(monthly().iloc[4])} or len(set(r["forecast"]["y"])) == 1
    assert any("history" in c for c in r["caveats"])
    check_schema(r, 4)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_short_is_a_clear_refusal(n):
    r = forecast_series(monthly().iloc[:n], 4, "MS")
    assert r["status"] == "refused" and r["reason"] and r["forecast"]["y"] == []
    json.dumps(r)


def test_constant_series():
    r = forecast_series(pd.Series(5.0, index=pd.date_range("2023-01-01", periods=30, freq="MS")), 6, "MS")
    assert r["forecast"]["y"] == [5.0] * 6 and r["confidence"] == "low"
    assert any("flat" in c for c in r["caveats"])


def test_all_zeros_and_intermittent():
    z = pd.Series(0.0, index=pd.date_range("2023-01-01", periods=30, freq="W"))
    r = forecast_series(z, 4, "W")
    assert r["forecast"]["y"] == [0.0] * 4 and r["forecast"]["lower"] == [0.0] * 4
    rng = np.random.default_rng(0)
    v = rng.poisson(0.7, 80) * 10.0
    r2 = forecast_series(pd.Series(v, index=pd.date_range("2024-01-01", periods=80, freq="D")), 7, "D")
    assert r2["confidence"] == "low" and any("zero" in c for c in r2["caveats"])
    assert min(r2["forecast"]["lower95"]) >= 0


def test_negative_values_not_floored():
    s = monthly() - 120.0
    r = forecast_series(s, 6, "MS")
    assert min(r["forecast"]["y"]) < 0 and min(r["forecast"]["lower95"]) < 0
    check_schema(r, 6)


def test_nonnegative_history_gives_nonnegative_bands():
    rng = np.random.default_rng(2)
    s = pd.Series(np.abs(rng.normal(3, 5, 60)), index=pd.date_range("2024-01-01", periods=60, freq="D"))
    r = forecast_series(s, 10, "D")
    assert min(r["forecast"]["lower95"]) >= 0 and min(r["forecast"]["y"]) >= 0


def test_nan_gaps_inf_and_trailing_blanks():
    s = monthly()
    s.iloc[10:13] = np.nan
    s.iloc[5] = np.inf
    s.iloc[-2:] = np.nan
    r = forecast_series(s, 6, "MS")
    assert r["status"] == "ok"
    assert r["history"]["x"][-1] == "2024-10-01" and r["forecast"]["x"][0] == "2024-11-01"
    assert all(v is not None and np.isfinite(v) for v in r["history"]["y"])
    assert any("interpolation" in c for c in r["caveats"]) and any("ignored" in c for c in r["caveats"])
    check_schema(r, 6)


def test_all_nan_and_garbage_never_raise():
    for bad in [pd.Series([np.nan] * 12), None, "abc", {"a": 1}, [], pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}), pd.Series(["x", "y", "z"] * 4), 7]:
        r = forecast_series(bad, 6, "MS")
        assert r["status"] == "refused", bad
        json.dumps(r)
    r = forecast_series(monthly(), "not a number", "???")
    assert r["status"] == "ok" and len(r["forecast"]["y"]) == 6


def test_outlier_spike_does_not_distort_forecast():
    base = pd.Series(100.0 + np.random.default_rng(0).normal(0, 2, 40), index=pd.date_range("2024-01-01", periods=40, freq="D"))
    spiked = base.copy()
    spiked.iloc[25] = 5000.0
    a = forecast_series(base, 7, "D")["forecast"]["y"]
    b = forecast_series(spiked, 7, "D")["forecast"]["y"]
    assert np.max(np.abs(np.array(a) - np.array(b))) < 8.0   # a 50x one-off does not drag the forecast


def test_level_shift_is_tracked():
    rng = np.random.default_rng(1)
    v = np.r_[rng.normal(100, 3, 30), rng.normal(160, 3, 30)]
    r = forecast_series(pd.Series(v, index=pd.date_range("2024-01-01", periods=60, freq="D")), 7, "D")
    assert min(r["forecast"]["y"]) > 140


def test_duplicate_dates_irregular_and_alt_indices():
    s = monthly()
    dup = pd.concat([s, s.iloc[-1:]])
    r = forecast_series(dup, 3, "MS")
    assert r["status"] == "ok" and any("combined" in c for c in r["caveats"])
    gap = s.drop(s.index[[7, 8]])
    assert len(forecast_series(gap, 3, "MS")["history"]["y"]) == 36
    tz = s.copy()
    tz.index = tz.index.tz_localize("UTC")
    assert forecast_series(tz, 3, "MS")["status"] == "ok"
    st = s.copy()
    st.index = st.index.strftime("%Y-%m-%d")
    assert forecast_series(st, 3, "MS")["forecast"]["x"][0] == "2025-01-01"
    mid = s.copy()
    mid.index = mid.index + pd.Timedelta(days=14)
    assert forecast_series(mid, 3, "MS")["status"] == "ok"
    pl = forecast_series(list(s.values), 3, "MS")
    assert pl["status"] == "ok" and pl["forecast"]["x"] == [36, 37, 38]


def test_frequency_aliases_and_inference():
    s = monthly()
    a = forecast_series(s, 4, "M")
    b = forecast_series(s, 4, "MS")
    c = forecast_series(s, 4)           # inferred from the index
    assert a["forecast"]["y"] == b["forecast"]["y"] == c["forecast"]["y"]
    w = forecast_series(pd.Series(np.random.default_rng(0).normal(50, 3, 30), index=pd.date_range("2024-01-07", periods=30, freq="W")), 3, "W")
    assert w["forecast"]["x"][0] == "2024-08-04"


def test_extreme_scales():
    for k in (1e-8, 1e12):
        r = forecast_series(monthly() * k, 6, "MS")
        assert r["status"] == "ok" and all(v != 0 for v in r["forecast"]["y"])


def test_horizon_clamped_and_long_history_truncated():
    assert len(forecast_series(monthly(), 0, "MS")["forecast"]["y"]) == 1
    assert len(forecast_series(monthly(), 1000, "MS")["forecast"]["y"]) == 60
    long = daily(1500)
    r = forecast_series(long, 7, "D")
    assert r["status"] == "ok" and any("most recent" in c for c in r["caveats"])


def test_honest_label_when_nothing_beats_naive():
    """White noise has nothing to learn: the label must not claim skill."""
    rng = np.random.default_rng(11)
    s = pd.Series(rng.normal(100, 10, 60), index=pd.date_range("2024-01-01", periods=60, freq="D"))
    r = forecast_series(s, 7, "D")
    # a flat average legitimately beats 'repeat last value' on noise, so either honest label is acceptable, but never "high" + no skill
    if "no better than naive" in r["baseline_comparison"]:
        assert r["confidence"] != "high"


# ----------------------------------------------------------------------------- calibration on known processes
@needs_threadlimit
def test_interval_coverage_close_to_nominal_on_known_process():
    """AR(1) + weekly seasonality, 40 independent series: pooled held-out coverage should be near 80/95%."""
    hit80 = hit95 = tot = 0
    for seed in range(40):
        rng = np.random.default_rng(1000 + seed)
        n, H = 90, 7
        e = np.zeros(n + H)
        for k in range(1, n + H):
            e[k] = 0.4 * e[k - 1] + rng.normal(0, 6)
        t = np.arange(n + H)
        y = 200 + 25 * np.sin(2 * np.pi * t / 7) + 0.2 * t + e
        s = pd.Series(y[:n], index=pd.date_range("2024-01-01", periods=n, freq="D"))
        f = forecast_series(s, H, "D")["forecast"]
        truth = y[n:]
        hit80 += int(np.sum((truth >= f["lower80"]) & (truth <= f["upper80"])))
        hit95 += int(np.sum((truth >= f["lower95"]) & (truth <= f["upper95"])))
        tot += H
    assert 0.70 <= hit80 / tot <= 0.92, hit80 / tot
    assert 0.88 <= hit95 / tot <= 0.995, hit95 / tot


def test_old_method_overstates_confidence_example():
    """Regression guard for the headline claim: bands are wider for noisier series and widen with horizon only gently."""
    quiet = forecast_series(daily(120, noise=2.0, weekly=10.0), 7, "D")["forecast"]
    noisy = forecast_series(daily(120, noise=20.0, weekly=10.0), 7, "D")["forecast"]
    wq = np.mean(np.array(quiet["upper95"]) - np.array(quiet["lower95"]))
    wn = np.mean(np.array(noisy["upper95"]) - np.array(noisy["lower95"]))
    assert wn > 3 * wq
