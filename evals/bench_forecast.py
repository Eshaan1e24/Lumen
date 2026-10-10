"""New forecaster vs the original Holt-Winters, on synthetic business series, last-6-periods holdout.   python -m evals.bench_forecast
Also checks the forecaster's SELF-ASSESSMENT: do its confidence labels and its "beats naive" verdicts match what happens on the hidden periods?
Series: monthly/weekly/daily, trend + seasonality + noise + occasional level shifts, 18-120 periods. For each series the last 6
periods are hidden; each method forecasts them. Reported: median MASE (error relative to a seasonal-naive forecast; lower is better,
1.0 = as good as seasonal naive) and how often the stated 95% range contained the truth (should be near 95%)."""
import warnings, sys
import numpy as np, pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from app.forecasting import forecast_series
warnings.simplefilter("ignore")
H = 6


def make(seed):
    rng = np.random.default_rng(seed); freq, m = [("MS", 12), ("W", 52), ("D", 7)][seed % 3]
    n = int(rng.integers(18, 121)) if freq != "D" else int(rng.integers(60, 200))
    t = np.arange(n); level = rng.uniform(50, 500)
    y = level * (1 + rng.uniform(-.004, .01) * t) * (1 + rng.uniform(0, .5) * np.sin(2 * np.pi * t / m + rng.uniform(0, 6)) * (rng.random() < .6))
    y = y * (1 + rng.normal(0, rng.uniform(.03, .25), n))
    if rng.random() < .3: y[int(n * rng.uniform(.4, .8)):] *= rng.uniform(.7, 1.4)         # a level shift
    return pd.Series(np.clip(y, 0, None), index=pd.date_range("2022-01-03", periods=n, freq=freq)), freq, m


def old_hw(s, m):
    seasonal = "add" if len(s) >= 2 * m else None
    try: fit = ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal=seasonal, seasonal_periods=m if seasonal else None).fit()
    except Exception: fit = ExponentialSmoothing(s, trend="add").fit()
    fc = fit.forecast(H).values; sd = float(np.std(fit.resid)); w = 1.96 * sd * np.sqrt(np.arange(1, H + 1))
    return fc, np.maximum(fc - w, 0), fc + w


def main(n_series=90):
    res = {"old": [], "new": []}; cov = {"old": [0, 0], "new": [0, 0]}; by_conf = {}; verdicts = []; naive_mase = []
    for seed in range(n_series):
        s, freq, m = make(seed); train, test = s.iloc[:-H], s.iloc[-H:].values
        lag = m if len(train) > m else 1
        scale = np.mean(np.abs(train.values[lag:] - train.values[:-lag])) or 1.0
        fo, lo, hi = old_hw(train, m); r = forecast_series(train, horizon=H, freq=freq, agg="sum")
        if r["status"] == "refused": continue
        fn, ln, hn = (np.array(r["forecast"][k]) for k in ("y", "lower95", "upper95"))
        nv = np.mean(np.abs(test - train.values[-1])) / scale; naive_mase.append(nv)
        by_conf.setdefault(r.get("confidence"), []).append(np.mean(np.abs(test - fn)) / scale)
        verdicts.append((r.get("baseline", {}).get("vs_naive_pct"), np.mean(np.abs(test - fn)) / scale < nv))
        for name, f, l, h in (("old", fo, lo, hi), ("new", fn, ln, hn)):
            res[name].append(np.mean(np.abs(test - f)) / scale); cov[name][0] += int(((test >= l) & (test <= h)).sum()); cov[name][1] += H
    n = len(res["old"]); old, new = np.array(res["old"]), np.array(res["new"])
    print(f"{n} synthetic series, holdout = last {H} periods")
    print(f"median MASE   old Holt-Winters {np.median(old):.3f}   new {np.median(new):.3f}   (lower is better)")
    print(f"mean MASE     old {old.mean():.3f}   new {new.mean():.3f}")
    print(f"new beats old on {np.mean(new < old):.0%} of series")
    print(f"95% range contained the truth: old {cov['old'][0] / cov['old'][1]:.1%}   new {cov['new'][0] / cov['new'][1]:.1%}  (target 95%)")
    print(f"naive (repeat the last value): median MASE {np.median(naive_mase):.3f}; new forecaster beats naive on {np.mean(new < np.array(naive_mase)):.0%} of series")
    print("Confidence label -> median MASE actually achieved on the hidden periods (lower is better):")
    for k in ("high", "medium", "low"):
        if k in by_conf: print(f"  {k:7} n={len(by_conf[k]):3}  median MASE {np.median(by_conf[k]):.3f}")
    claimed = [(v, ok) for v, ok in verdicts if v is not None and v > 0]
    if claimed: print(f"When Lumen said 'beats naive' ({len(claimed)} series), the hidden periods agreed in {np.mean([ok for _, ok in claimed]):.0%} of them")


if __name__ == "__main__": main(int(sys.argv[1]) if len(sys.argv) > 1 else 90)
