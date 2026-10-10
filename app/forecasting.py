"""forecasting.py - a small, honest forecaster for business time series.

Public API
----------
    forecast_series(s, horizon=6, freq=None, *, agg="sum") -> dict

What it does (all of it deterministic; no random numbers anywhere):
  1. Cleans the input (NaN/inf, duplicate dates, gaps, trailing blanks, outlier spikes for *fitting only*).
  2. Builds a few candidate models: naive, seasonal naive, simple exponential smoothing, damped-trend
     smoothing, Theta, seasonal exponential smoothing (raw and log scale) and simple averages of them.
  3. Chooses among them by rolling-origin time-series cross-validation on the series itself (not AIC),
     with a parsimony rule so a complex model must clearly beat a simple one.
  4. Builds 80% and 95% prediction intervals from that model's own out-of-sample backtest errors
     (empirical, horizon-aware) instead of a textbook sd*sqrt(h).
  5. Reports how it did against naive / seasonal-naive in the same backtest, a high/medium/low
     confidence label, and plain-English caveats.

Dependencies: numpy, pandas, statsmodels (scipy comes with it). Written against pandas 3 / statsmodels 0.15.
It never raises: on bad input it returns status "refused" with a reason, and on short history it falls
back to a labelled naive forecast.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

try:  # optional: statsmodels' tiny linear algebra is ~10-50x slower when BLAS spawns many threads
    from threadpoolctl import threadpool_limits as _threadpool_limits
except Exception:  # pragma: no cover
    _threadpool_limits = None

try:  # statsmodels is only needed for the smoothing / Theta candidates
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    from statsmodels.tsa.forecasting.theta import ThetaModel
    _HAVE_SM = True
except Exception:  # pragma: no cover - exercised only when statsmodels is missing
    ETSModel = ThetaModel = None
    _HAVE_SM = False

__all__ = ["forecast_series", "CFG"]

Z80 = 1.2815515655446004   # two-sided 80% normal quantile
Z95 = 1.959963984540054    # two-sided 95% normal quantile

# Tunable knobs (the benchmark harness overrides these for ablations; the defaults are what ships).
CFG = dict(
    max_folds=5,            # rolling origins used for model choice and interval calibration
    min_train=6,            # smallest training window allowed in a fold
    min_seasonal_folds=3,   # seasonal-ETS candidates need at least this many folds with >= 2 full cycles
    max_ets_season=24,      # no seasonal smoothing for longer cycles (52-week ETS is slow and over-parameterised)
    winsorize=True,         # clip extreme one-off spikes before FITTING (never before scoring)
    winsor_k=4.0,           # robust-sigma multiple
    max_fit_points=400,     # only the most recent points are used for fitting
    log_candidates=True,    # multiplicative seasonality via log scale
    candidates=None,        # optional whitelist of candidate ids (ablations)
    seasonal_damped=False,  # also try damped-trend seasonal smoothing
    extra_components=[],    # additional components (e.g. "drift") to backtest
    combos={"combo_trend": ["theta", "damped", "ses"],
            "combo_all": ["theta", "damped", "ses", "ets_s", "ets_s_log", "snaive"]},
    select="default",       # "default": keep cfg["default"] unless a challenger wins by cfg["margin"]; "argmin": best CV with parsimony
    default="combo_all",    # equal-weight average of every eligible statistical model
    margin=0.10,            # a single model / sub-average must beat the default's CV error by 10% to replace it
    parsimony=0.03,         # (argmin mode only)
    # prediction intervals
    interval_b=0.10,        # prior horizon-growth exponent: sigma_h ~ h^b
    interval_b_weight=0.5,  # weight on the b estimated from the backtest (rest on the prior)
    interval_a=0.5,         # finite-sample widening (1 + a/folds)
)

_SEASON = {"D": 7, "W": 52, "MS": 12, "ME": 12, "QS": 4, "QE": 4, "YS": 1, "YE": 1, "h": 24, "B": 5}
_UNIT = {"D": "day", "W": "week", "MS": "month", "ME": "month", "QS": "quarter", "QE": "quarter",
         "YS": "year", "YE": "year", "h": "hour", "B": "business day"}
_ALIAS = {"M": "ME", "A": "YE", "Y": "YE", "Q": "QE", "H": "h", "T": "min", "S": "s", "AS": "YS"}

_NAMES = {
    "naive": "Naive (repeat the last value)",
    "snaive": "Seasonal naive (repeat the last cycle)",
    "ses": "Simple exponential smoothing",
    "damped": "Damped-trend exponential smoothing",
    "theta": "Theta method",
    "ets_s": "Seasonal exponential smoothing",
    "ets_s_log": "Seasonal exponential smoothing (multiplicative, log scale)",
    "ets_sd": "Seasonal exponential smoothing with damped trend",
    "ets_sd_log": "Seasonal exponential smoothing with damped trend (log scale)",
    "drift": "Naive with drift",
    "snaive_growth": "Seasonal naive with year-over-year growth",
}
_COMPLEXITY = {"naive": 0, "snaive": 1, "snaive_growth": 1.5, "drift": 2, "ses": 2, "theta": 3, "damped": 4, "combo": 5, "combo_trend": 5, "combo_all": 5,
               "ets_s": 6, "ets_s_log": 7, "ets_sd": 8, "ets_sd_log": 9, "combo_seasonal": 8}


# ----------------------------------------------------------------------------- input handling
def _to_off(freq):
    if freq is None:
        return None
    f = str(freq).strip()
    f = _ALIAS.get(f, f)
    try:
        return to_offset(f)
    except Exception:
        return None


def _base(off) -> str:
    return off.name.split("-")[0] if off is not None else ""


def _period_alias(off):
    b = _base(off)
    return {"D": "D", "B": "B", "W": off.freqstr if b == "W" else None, "MS": "M", "ME": "M", "QS": "Q", "QE": "Q",
            "YS": "Y", "YE": "Y", "h": "h"}.get(b)


def _guess_offset(idx: pd.DatetimeIndex):
    try:
        f = pd.infer_freq(idx) if len(idx) >= 3 else None
        if f:
            return to_offset(f)
    except Exception:
        pass
    if len(idx) < 2:
        return None
    d = float(np.median(np.diff(idx.values).astype("timedelta64[s]").astype(float))) / 86400.0
    if d <= 0:
        return None
    if d < 0.2:
        return to_offset("h")
    if d <= 1.5:
        return to_offset("D")
    if 5 <= d <= 9:
        return to_offset("W")
    if 26 <= d <= 33:
        return to_offset("MS")
    if 85 <= d <= 95:
        return to_offset("QS")
    return None


def _prepare(s, freq, agg):
    """Return (info, None) or (None, refusal_reason). info has y, labels, off, m, notes, counts."""
    notes: list[str] = []
    if isinstance(s, pd.DataFrame):
        if s.shape[1] != 1:
            return None, "Expected a single series, got a table with several columns."
        s = s.iloc[:, 0]
    if not isinstance(s, pd.Series):
        try:
            s = pd.Series(s)
        except Exception:
            return None, "The input could not be read as a series of numbers."
    if len(s) == 0:
        return None, "The series is empty."
    if pd.api.types.is_datetime64_any_dtype(s.dtype) or pd.api.types.is_timedelta64_dtype(s.dtype):
        return None, "The values are dates, not numbers."
    vals = pd.to_numeric(s, errors="coerce").astype("float64")
    vals = vals.replace([np.inf, -np.inf], np.nan)
    n_bad = int(vals.isna().sum() - s.isna().sum())
    if n_bad > 0:
        notes.append(f"{n_bad} value(s) were not numbers and were treated as missing.")

    idx = vals.index
    if isinstance(idx, pd.PeriodIndex):
        idx = idx.to_timestamp()
    elif not isinstance(idx, pd.DatetimeIndex) and (pd.api.types.is_object_dtype(idx.dtype)
                                                    or pd.api.types.is_string_dtype(idx.dtype)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idx = pd.DatetimeIndex(pd.to_datetime(idx, errors="raise", format="mixed"))
        except Exception:
            idx = pd.RangeIndex(len(vals))
    if isinstance(idx, pd.DatetimeIndex):
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        vals.index = idx
        keep = ~idx.isna()
        if not keep.all():
            vals = vals[keep]
            notes.append("Rows with a missing date were dropped.")
        if not vals.index.is_monotonic_increasing:
            vals = vals.sort_index()
    else:
        vals.index = pd.RangeIndex(len(vals))
    if len(vals) == 0:
        return None, "The series is empty."

    off = _to_off(freq)
    datelike = isinstance(vals.index, pd.DatetimeIndex)
    if datelike and off is None:
        off = _guess_offset(vals.index)
        if off is None:
            notes.append("The date spacing is irregular, so the series was treated as evenly spaced observations.")
            datelike = False
    if datelike and _base(off) not in _SEASON:   # unsupported alias (e.g. "SMS"): fall back to the spacing we can see
        g = _guess_offset(vals.index)
        off = g if (g is not None and _base(g) in _SEASON) else None
        if off is None:
            datelike = False

    labels = None
    if datelike:
        idx = vals.index
        regular = False
        try:
            regular = (not idx.has_duplicates) and idx.equals(pd.date_range(idx[0], periods=len(idx), freq=off))
        except Exception:
            regular = False
        if not regular:
            pa = _period_alias(off)
            if pa is None:
                datelike = False
            else:
                grp = vals.groupby(idx.to_period(pa))
                vals_p = getattr(grp, agg if agg in ("sum", "mean", "last", "max", "min") else "sum")()
                dup = int(len(vals) - len(vals_p))
                full = pd.period_range(vals_p.index.min(), vals_p.index.max(), freq=pa)
                if len(full) > 20000:
                    return None, "The date range is too long for this frequency."
                vals_p = vals_p.reindex(full)
                how = "end" if _base(off) in ("W", "ME", "QE", "YE") else "start"
                lab = full.to_timestamp(how=how).normalize()
                vals = pd.Series(vals_p.to_numpy(dtype="float64"), index=lab)
                if dup > 0:
                    notes.append(f"{dup} observation(s) shared a period with another and were combined ({agg}).")
                n_gap_new = int(vals.isna().sum())
                if n_gap_new:
                    notes.append(f"{n_gap_new} period(s) had no data.")
    if datelike:
        labels = list(vals.index)

    y = vals.to_numpy(dtype="float64")
    # trim leading / trailing blanks
    ok = ~np.isnan(y)
    if not ok.any():
        return None, "The series has no usable numbers."
    first, last = int(np.argmax(ok)), int(len(y) - 1 - np.argmax(ok[::-1]))
    if last < len(y) - 1:
        notes.append(f"The last {len(y) - 1 - last} period(s) have no data and were ignored; the forecast starts after the last real value.")
    y = y[first:last + 1]
    if labels is not None:
        labels = labels[first:last + 1]
    n_interp = int(np.isnan(y).sum())
    n_real = int(len(y) - n_interp)
    if n_interp:
        yy = pd.Series(y).interpolate(method="linear", limit_area="inside").to_numpy()
        y = yy
        notes.append(f"{n_interp} missing period(s) in the middle were filled by straight-line interpolation.")
    m = _SEASON.get(_base(off), 1) if (datelike or freq is not None) else 1
    if not datelike and off is not None and _base(off) in _SEASON:
        m = _SEASON[_base(off)]
    return dict(y=y, labels=labels, off=off if datelike else None, m=int(m), notes=notes,
                n_interp=n_interp, n_real=n_real, unit=_UNIT.get(_base(off), "period")), None


def _winsorize(y: np.ndarray, m: int, k: float) -> tuple[np.ndarray, int]:
    """Clip extreme one-off spikes (both signs) around a rolling-median level. Fitting copy only."""
    n = len(y)
    if n < 12:
        return y, 0
    w = int(min(max(5, m if m > 1 else 5), max(5, n // 3)))
    if w % 2 == 0:
        w += 1
    base = pd.Series(y).rolling(w, center=True, min_periods=max(3, w // 2)).median().to_numpy()
    r = y - base
    mad = np.nanmedian(np.abs(r - np.nanmedian(r))) * 1.4826
    if not np.isfinite(mad) or mad <= 0:
        return y, 0
    lo, hi = base - k * mad, base + k * mad
    out = np.clip(y, lo, hi)
    return out, int(np.sum(out != y))


# ----------------------------------------------------------------------------- candidate models
def _chk(fc, H):
    fc = np.asarray(fc, dtype="float64")
    if fc.shape != (H,) or not np.all(np.isfinite(fc)):
        raise ValueError("bad forecast")
    return fc


def _ets(y, H, m, trend, damped, seasonal, log=False):
    z = np.log(y) if log else y
    mod = ETSModel(z, error="add", trend=trend, damped_trend=bool(damped and trend), seasonal=seasonal,
                   seasonal_periods=m if seasonal else None)
    res = mod.fit(disp=False, maxiter=300)
    fc = np.asarray(res.forecast(H), dtype="float64")
    if log:
        if np.max(fc) > 50:
            raise ValueError("log-scale blow-up")
        fc = np.exp(fc)
    return _chk(fc, H)


def _comp_forecast(name, y, H, m):
    n = len(y)
    if name == "naive":
        return np.full(H, y[-1])
    if name == "snaive":
        return y[-m:][np.arange(H) % m]
    if name == "snaive_growth":
        # last cycle repeated, scaled by year-over-year growth measured on the most recent k periods (additive if not all positive)
        k = int(min(m, n - m))
        if k < 3:
            raise ValueError("not enough history for a growth estimate")
        base = y[-m:][np.arange(H) % m]
        rec, prev = y[-k:], y[-k - m:-m]
        cyc = np.arange(H) // m + 1
        if np.min(y[-k - m:]) > 0 and prev.sum() > 0:
            g = float(np.clip(rec.sum() / prev.sum(), 0.5, 2.0))
            return base * g ** cyc
        d = float(rec.mean() - prev.mean())
        return base + d * cyc
    if name == "drift":
        return y[-1] + (y[-1] - y[0]) / max(n - 1, 1) * np.arange(1, H + 1)
    if name == "ses":
        return _ets(y, H, m, None, False, None)
    if name == "damped":
        return _ets(y, H, m, "add", True, None)
    if name == "theta":
        ds = bool(m > 1 and n >= 2 * m)
        res = ThetaModel(y, period=m if ds else None, deseasonalize=ds).fit()
        return _chk(res.forecast(H), H)
    if name == "ets_s":
        return _ets(y, H, m, None, False, "add")
    if name == "ets_s_log":
        return _ets(y, H, m, None, False, "add", log=True)
    if name == "ets_sd":
        return _ets(y, H, m, "add", True, "add")
    if name == "ets_sd_log":
        return _ets(y, H, m, "add", True, "add", log=True)
    raise KeyError(name)


# ----------------------------------------------------------------------------- CV + intervals
def _plan(n, H, m, cfg):
    """Backtest folds. Tier 3: >= 2 full cycles in every training window (all models, incl. seasonal smoothing);
    tier 2: >= 1 cycle + 3 periods (adds seasonal naive and its growth-adjusted form); tier 1: non-seasonal models only.
    Every candidate is scored on the same folds, so the tier decides which candidates exist."""
    h_cv = int(max(1, min(H, 12, n // 3)))
    stride = max(1, h_cv // 2)

    def origins(min_train):
        out, t = [], n - h_cv
        while t >= min_train and len(out) < cfg["max_folds"]:
            out.append(t)
            t -= stride
        return sorted(out)

    tier = 1
    og = origins(cfg["min_train"])
    if m > 1:
        og2 = origins(m + 3)
        if len(og2) >= 2:
            og, tier = og2, 2
        if _HAVE_SM and m <= cfg["max_ets_season"]:
            og3 = origins(2 * m)
            if len(og3) >= cfg["min_seasonal_folds"]:
                og, tier = og3, 3
    return og, h_cv, tier


def _interval_params(E, h_cv, cfg):
    """Horizon law sigma_h = c * h^b and 80/95% multipliers from backtest errors E (folds x h_cv).

    c is the RMS of E / h^b. Multipliers are empirical quantiles of |E| / sigma_h (rank-corrected), floored so the 95% band is never
    narrower than the normal-shape ratio to the 80% band, pulled toward normal quantiles when there are few points, and widened by
    (1 + a / folds) because few folds underestimate future error and the chosen model's backtest error is optimistic."""
    F = E.shape[0]
    hs = np.arange(1, h_cv + 1, dtype="float64")
    b0 = cfg["interval_b"]
    rms_h = np.sqrt(np.mean(E ** 2, axis=0))
    if h_cv >= 3 and F >= 2 and np.all(rms_h > 0):
        b_hat = float(np.clip(np.polyfit(np.log(hs), np.log(rms_h), 1)[0], 0.0, 0.6))
        b = cfg["interval_b_weight"] * b_hat + (1 - cfg["interval_b_weight"]) * b0
    else:
        b = b0
    c = float(np.sqrt(np.mean(E ** 2 / hs ** (2 * b))))
    if c <= 0:
        return dict(c=0.0, b=b, q80=Z80, q95=Z95)
    z = np.sort(np.abs(E / (c * hs ** b)).ravel())
    N = z.size

    def emp(level):
        return float(z[min(N, math.ceil((N + 1) * level)) - 1])

    e80, e95 = emp(0.80), emp(0.95)
    e95 = max(e95, e80 * Z95 / Z80)
    if N < 16:
        w = N / 16.0
        e80, e95 = w * e80 + (1 - w) * Z80, w * e95 + (1 - w) * Z95
    infl = 1.0 + cfg["interval_a"] / F
    return dict(c=c, b=b, q80=float(e80 * infl), q95=float(e95 * infl))


def _lofo_coverage(E, h_cv, cfg):
    """Coverage of the interval recipe on folds it did not see (leave-one-fold-out)."""
    F = E.shape[0]
    if F < 2:
        return None
    hs = np.arange(1, h_cv + 1, dtype="float64")
    hit80 = hit95 = tot = 0
    for f in range(F):
        P = _interval_params(np.delete(E, f, axis=0), h_cv, cfg)
        w = P["c"] * hs ** P["b"]
        a = np.abs(E[f])
        hit80 += int(np.sum(a <= P["q80"] * w))
        hit95 += int(np.sum(a <= P["q95"] * w))
        tot += a.size
    return {"80": hit80 / tot, "95": hit95 / tot}


def _smape(a, f):
    d = np.abs(a) + np.abs(f)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(d > 0, 2 * np.abs(a - f) / d, 0.0)
    return float(100 * np.mean(v))


# ----------------------------------------------------------------------------- engine: collect (expensive) + decide (cheap)
def _collect(y_raw, m, H, cfg, all_final=False):
    """Rolling-origin backtest of every applicable component. all_final=True also fits each on the full series (lab use)."""
    n = len(y_raw)
    nonneg = bool(np.nanmin(y_raw) >= 0)
    y_fit, n_clip = (_winsorize(y_raw, m, cfg["winsor_k"]) if cfg["winsorize"] else (y_raw, 0))
    pos = bool(np.min(y_fit) > 0)
    origins, h_cv, tier = _plan(n, H, m, cfg)
    seasonal = tier == 3
    names = ["naive"]
    if tier >= 2:
        names += ["snaive", "snaive_growth"]
    if _HAVE_SM:
        names += ["ses", "damped", "theta"]
        if seasonal:
            names.append("ets_s")
            if cfg["seasonal_damped"]:
                names.append("ets_sd")
            if pos and cfg["log_candidates"]:
                names.append("ets_s_log")
                if cfg["seasonal_damped"]:
                    names.append("ets_sd_log")
    names += [c for c in cfg["extra_components"] if c not in names]
    if cfg["candidates"]:
        names = [c for c in names if c in cfg["candidates"] or c == "naive"]
    fold_true = np.array([y_raw[t:t + h_cv] for t in origins]) if origins else np.zeros((0, h_cv))
    fold_fc = {}
    for nm in names:
        rows = []
        for t in origins:
            try:
                fc = _comp_forecast(nm, y_fit[:t], h_cv, m)
                if nonneg:
                    fc = np.maximum(fc, 0.0)
                rows.append(fc)
            except Exception:
                rows = None
                break
        if rows is not None:
            fold_fc[nm] = np.array(rows).reshape(len(origins), h_cv)
    final = {}
    if all_final:
        for nm in fold_fc:
            try:
                f_ = _comp_forecast(nm, y_fit, H, m)
                final[nm] = np.maximum(f_, 0.0) if nonneg else f_
            except Exception:
                pass
    return dict(y_raw=y_raw, y_fit=y_fit, m=m, H=H, n=n, nonneg=nonneg, n_clip=n_clip, origins=origins, h_cv=h_cv,
                seasonal=seasonal, tier=tier, fold_true=fold_true, fold_fc=fold_fc, final=final)


def _build_candidates(col, cfg):
    """Components + simple-average combos, all scored on identical folds."""
    allowed = cfg["candidates"]
    comps = {c: v for c, v in col["fold_fc"].items() if (not allowed or c in allowed or c == "naive")}
    cands = dict(comps)
    members = {}
    for cname, pref in cfg["combos"].items():
        if allowed and cname not in allowed:
            continue
        mem = [c for c in pref if c in comps and not (c == "snaive" and col["m"] > 12 and not col["seasonal"])]
        if len(mem) >= 2:
            cands[cname] = np.mean([comps[c] for c in mem], axis=0)
            members[cname] = mem
    return cands, members


def _choose(cv, cfg):
    """Pick the model from cross-validated absolute error."""
    mn = min(cv.values())
    if cfg["select"] == "default" and cfg["default"] in cv:
        d = cfg["default"]
        ch = min((c for c in cv if c != d), key=lambda c: (cv[c], _COMPLEXITY.get(c, 9)), default=None)
        if ch is not None and cv[ch] < cv[d] * (1 - cfg["margin"]):
            return ch
        return d
    thr = mn * (1 + cfg["parsimony"]) if mn > 0 else 0.0
    elig = [c for c in cv if cv[c] <= thr] or [min(cv, key=cv.get)]
    return sorted(elig, key=lambda c: (_COMPLEXITY.get(c, 9), cv[c]))[0]


def _decide(col, cfg):
    y_raw, y_fit, m, H, n = col["y_raw"], col["y_fit"], col["m"], col["H"], col["n"]
    origins, h_cv, nonneg = col["origins"], col["h_cv"], col["nonneg"]
    notes = []
    if col["n_clip"]:
        notes.append(f"{col['n_clip']} extreme one-off value(s) were damped before fitting so they would not distort the trend; the backtest still scores against the real values.")
    cands, members = _build_candidates(col, cfg)
    T = col["fold_true"]
    errs = {c: T - v for c, v in cands.items()} if len(origins) else {}
    scale = float(np.mean(np.abs(np.diff(y_raw)))) if n > 1 else 0.0
    cv = {c: float(np.mean(np.abs(e))) for c, e in errs.items()}
    best = _choose(cv, cfg) if cv else "naive"

    def final_of(c):
        if c in col["final"]:
            return col["final"][c]
        f_ = _comp_forecast(c, y_fit, H, m)
        return np.maximum(f_, 0.0) if nonneg else f_

    try:
        fc = np.mean([final_of(c) for c in members[best]], axis=0) if best in members else final_of(best)
        _chk(fc, H)
    except Exception:
        best = "naive"
        fc = np.full(H, y_raw[-1])
        notes.append("The chosen model failed on the full history, so a naive forecast is shown instead.")
    if nonneg:
        fc = np.maximum(fc, 0.0)

    E = errs.get(best)
    hs_all = np.arange(1, H + 1, dtype="float64")
    if E is not None and E.size:
        P = _interval_params(E, h_cv, cfg)
        w = P["c"] * hs_all ** P["b"]
        lo80, hi80 = fc - P["q80"] * w, fc + P["q80"] * w
        lo95, hi95 = fc - P["q95"] * w, fc + P["q95"] * w
        lofo = _lofo_coverage(E, h_cv, cfg)
    else:
        d = np.diff(y_raw)
        sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        w = 1.25 * sd * np.sqrt(hs_all)
        lo80, hi80, lo95, hi95 = fc - Z80 * w, fc + Z80 * w, fc - Z95 * w, fc + Z95 * w
        lofo = None
    if nonneg:
        lo80, lo95 = np.maximum(lo80, 0.0), np.maximum(lo95, 0.0)

    bt = dict(mase=None, smape=None, coverage=None, n_folds=int(len(origins)), n_points=0, horizon_tested=int(h_cv) if len(origins) else 0)
    base = {}
    if E is not None and E.size:
        bt["mase"] = float(np.mean(np.abs(E)) / scale) if scale > 0 else None
        bt["smape"] = _smape(T, T - E)
        bt["coverage"] = lofo
        bt["n_points"] = int(E.size)
        for nm in ("naive", "snaive"):
            if nm in errs:
                base[nm] = float(np.mean(np.abs(errs[nm])))
        base["chosen"] = float(np.mean(np.abs(E)))
    return dict(best=best, members=members.get(best), fc=fc, lo80=lo80, hi80=hi80, lo95=lo95, hi95=hi95, backtest=bt, base=base, cv=cv,
                notes=notes, seasonal=col["seasonal"], tier=col["tier"], n_clip=col["n_clip"], h_cv=h_cv, origins=origins)


def _engine(y_raw, m, H, cfg):
    return _decide(_collect(y_raw, m, H, cfg), cfg)


# ----------------------------------------------------------------------------- reporting helpers
def _compare(base, best, seasonal_ok=None):
    """Compare the chosen model with naive / seasonal naive on the same backtest folds (mean absolute error)."""
    if not base or "chosen" not in base:
        return "no backtest was possible, so there is no evidence this beats a naive forecast - treat with caution", dict(vs=None, improvement_pct=None, verdict="unknown")
    mc = base["chosen"]

    def imp(b):
        return None if (b is None or b <= 0) else 100.0 * (b - mc) / b

    i_n, i_s = imp(base.get("naive")), imp(base.get("snaive"))
    extra = dict(vs_naive_pct=None if i_n is None else round(i_n, 1), vs_seasonal_naive_pct=None if i_s is None else round(i_s, 1))
    if best == "naive" or i_n is None or i_n < 5:
        txt = "no better than naive - treat with caution"
        if i_s is not None and i_s >= 5:
            txt += f" (it does beat seasonal-naive by {i_s:.0f}%)"
        return txt, dict(vs="naive", improvement_pct=None if i_n is None else round(i_n, 1), verdict="same", **extra)
    if i_s is not None and i_s >= 5:
        txt = f"beats seasonal-naive by {i_s:.0f}%" + (f" and naive by {i_n:.0f}%" if i_n is not None else "")
        return txt, dict(vs="seasonal_naive", improvement_pct=round(i_s, 1), verdict="better", **extra)
    if best == "snaive":
        return f"matches seasonal-naive (the best option found); beats naive by {i_n:.0f}%", dict(vs="seasonal_naive", improvement_pct=0.0, verdict="same", **extra)
    if i_s is not None:
        return f"beats naive by {i_n:.0f}% but no better than seasonal-naive", dict(vs="naive", improvement_pct=round(i_n, 1), verdict="better", **extra)
    return f"beats naive by {i_n:.0f}%", dict(vs="naive", improvement_pct=round(i_n, 1), verdict="better", **extra)


def _confidence(n_real, n_folds, base, bt, y, interp_frac, constant, intermittent, horizon_ratio, no_skill=False):
    """Heuristic label from backtest error size and amount of evidence. Not a probability."""
    if constant or n_folds == 0 or n_real < 8:
        return "low", None
    mc = base.get("chosen")
    level = float(np.mean(np.abs(y[-min(len(y), 12):]))) if len(y) else 0.0
    rel = None if (mc is None or level <= 0) else mc / level
    score = 0
    if rel is not None:
        score += 2 if rel <= 0.12 else 1 if rel <= 0.30 else 0
    score += 1 if n_folds >= 3 else 0
    score += 1 if n_real >= 24 else 0
    nv = base.get("naive")
    if nv and mc is not None and nv > 0 and (nv - mc) / nv >= 0.05:
        score += 1
    if intermittent or interp_frac > 0.2 or horizon_ratio > 0.5:
        score -= 2
    label = "high" if score >= 4 else "medium" if score >= 2 else "low"
    if no_skill and label == "high":
        label = "medium"   # accurate but not demonstrably better than repeating the last value
    return label, rel


def _round(a):
    a = np.asarray(a, dtype="float64")
    mx = float(np.nanmax(np.abs(a))) if a.size and np.isfinite(a).any() else 0.0
    if mx >= 10:
        return [None if not np.isfinite(v) else float(v) for v in np.round(a, 2)]
    return [None if not np.isfinite(v) else float(f"{v:.8g}") for v in a]   # keep tiny magnitudes (e.g. 1e-8) intact


def _future_labels(labels, off, n_hist, H):
    if labels is not None and off is not None:
        try:
            fut = pd.date_range(labels[-1], periods=H + 1, freq=off)[1:]
            return [t.strftime("%Y-%m-%d") for t in fut], True
        except Exception:
            pass
    return list(range(n_hist, n_hist + H)), False


def _empty(status, reason, horizon, freq):
    return {"status": status, "reason": reason, "model": None, "model_name": None, "freq": freq, "horizon": horizon,
            "history": {"x": [], "y": []},
            "forecast": {"x": [], "y": [], "lower": [], "upper": [], "lower80": [], "upper80": [], "lower95": [], "upper95": []},
            "lower": {"80": [], "95": []}, "upper": {"80": [], "95": []},
            "backtest": {"mase": None, "smape": None, "coverage": None, "n_folds": 0},
            "baseline_comparison": "no forecast was made", "baseline": {"vs": None, "improvement_pct": None, "verdict": "unknown"},
            "confidence": "low", "caveats": [reason], "note": reason}


def forecast_series(s, horizon: int = 6, freq: str | None = None, *, agg: str = "sum", _cfg: dict | None = None) -> dict:
    """Forecast one evenly spaced business series. Never raises.

    s        pandas Series (DatetimeIndex preferred; a plain list/array is accepted, treated as evenly spaced).
    horizon  number of future periods (clamped to 1..60).
    freq     pandas frequency alias such as "D", "W", "MS" ("M" is mapped to "ME"); inferred from the index if None.
    agg      how duplicate dates are combined ("sum" | "mean" | "last").

    Returns a dict (see INTEGRATION.md for the full JSON shape). Key fields: status ("ok" | "limited" | "refused"),
    model, model_name, history{x,y}, forecast{x,y,lower,upper,lower80,upper80,lower95,upper95}, backtest{mase,smape,
    coverage{80,95},n_folds,...}, baseline_comparison, confidence, caveats, note.
    """
    try:
        H = int(horizon)
    except Exception:
        H = 6
    H = int(min(max(H, 1), 60))
    try:
        if _threadpool_limits is not None:
            with _threadpool_limits(limits=1):
                return _forecast_series(s, H, freq, agg, {**CFG, **(_cfg or {})})
        return _forecast_series(s, H, freq, agg, {**CFG, **(_cfg or {})})
    except Exception as e:  # last-resort guard: this function must not raise
        r = _empty("refused", "The forecaster hit an unexpected problem with this data.", H, None if freq is None else str(freq))
        r["caveats"].append(f"Internal detail: {type(e).__name__}")
        return r


def _forecast_series(s, H, freq, agg, cfg):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        info, why = _prepare(s, freq, agg)
        if info is None:
            return _empty("refused", why, H, None if freq is None else str(freq))
        y_all, labels, off, m, notes = info["y"], info["labels"], info["off"], info["m"], list(info["notes"])
        n_all = len(y_all)
        if n_all > cfg["max_fit_points"]:
            cut = n_all - cfg["max_fit_points"]
            y = y_all[cut:]
            lab_fit = labels[cut:] if labels is not None else None
            notes.append(f"Only the most recent {cfg['max_fit_points']} periods were used to fit the model.")
        else:
            y, lab_fit = y_all, labels
        n = len(y)
        freq_str = off.freqstr if off is not None else (None if freq is None else str(freq))
        unit = info["unit"]
        fut_x, dated = _future_labels(labels, off, n_all, H)
        hist_x = [t.strftime("%Y-%m-%d") for t in labels] if labels is not None else list(range(n_all))
        constant = bool(np.ptp(y) == 0)
        intermittent = bool(np.mean(y == 0) > 0.2)

        if info["n_real"] < 3 or n < 3:
            r = _empty("refused", f"Only {info['n_real']} data point(s): at least 3 are needed to say anything about the future, and 8+ for a real forecast.", H, freq_str)
            r["history"] = {"x": hist_x, "y": _round(y_all)}
            return r

        caveats = []
        if n < 8:
            # honest naive fallback
            fc = np.full(H, y[-1])
            d = np.diff(y)
            sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
            hs = np.arange(1, H + 1, dtype="float64")
            infl = 1.0 + 2.0 / max(1, len(d))
            w = infl * sd * np.sqrt(hs)
            lo80, hi80, lo95, hi95 = fc - Z80 * w, fc + Z80 * w, fc - Z95 * w, fc + Z95 * w
            if np.min(y) >= 0:
                lo80, lo95 = np.maximum(lo80, 0), np.maximum(lo95, 0)
            caveats = [f"Only {n} periods of history: this is a plain 'same as the last value' forecast, not a fitted model.",
                       "The range is a rough guide from how much the series moved between periods; it has not been backtested."] + notes
            return _assemble("limited", "naive", fc, lo80, hi80, lo95, hi95, hist_x, y_all, fut_x, H, freq_str, unit,
                             dict(mase=None, smape=None, coverage=None, n_folds=0), "no backtest possible with this little history - treat with caution",
                             dict(vs=None, improvement_pct=None, verdict="unknown"), "low", caveats, info, m)

        if not _HAVE_SM:
            notes.append("statsmodels is not installed, so only naive-type models were available.")
        out = _engine(y, m, H, cfg)
        best = out["best"]
        caveats.extend(notes)
        caveats.extend(out["notes"])
        bt = out["backtest"]
        text, bdict = _compare(out["base"], best, "snaive" in out["base"])
        horizon_ratio = H / n
        interp_frac = info["n_interp"] / max(1, n)
        conf, rel = _confidence(info["n_real"], bt["n_folds"], out["base"], bt, y, interp_frac, constant, intermittent, horizon_ratio,
                                no_skill=(bdict["verdict"] == "same" and bdict["vs"] == "naive"))
        if m > 1 and not out["seasonal"]:
            need = 2 * m + out["h_cv"] + (cfg["min_seasonal_folds"] - 1) * max(1, out["h_cv"] // 2)
            caveats.append(f"Seasonal smoothing models need about {need} {unit}s of history to be tested fairly (you have {n}), so they were not used; "
                           f"a seasonal pattern is only captured if the seasonal-naive model was good enough to win.")
        if bt["n_folds"] == 0:
            caveats.append("Too little history to backtest; the range is a rough guide only.")
        elif bt["n_folds"] < 3:
            caveats.append(f"The backtest has only {bt['n_folds']} fold(s), so the model choice and the range are uncertain.")
        if H > out["h_cv"]:
            caveats.append(f"The backtest covers {out['h_cv']} period(s) ahead; the range further out extends that pattern and is less reliable.")
        if horizon_ratio > 0.5:
            caveats.append(f"You asked for {H} periods ahead from only {n} of history; long-range forecasts from short history are weak.")
        if constant:
            caveats.append("The history is perfectly flat, so the range has no width. Real results rarely stay exactly constant.")
        if intermittent:
            caveats.append("Over a fifth of periods are exactly zero (intermittent demand); percentage errors and ranges are unreliable here.")
        if interp_frac > 0.2:
            caveats.append(f"{100 * interp_frac:.0f}% of periods were missing and filled in, which weakens everything above.")
        if best == "naive":
            caveats.append("No candidate beat 'repeat the last value' in the backtest, so that is what is shown.")
        caveats.append("Backtest numbers were also used to pick the model, so they are slightly optimistic. The range assumes the future behaves like the recent past; a promotion, price change or shock is not anticipated.")
        status = "ok" if bt["n_folds"] >= 1 else "limited"
        return _assemble(status, best, out["fc"], out["lo80"], out["hi80"], out["lo95"], out["hi95"], hist_x, y_all, fut_x, H,
                         freq_str, unit, bt, text, bdict, conf, caveats, info, m, rel=rel, cv=out["cv"], h_cv=out["h_cv"])


def _assemble(status, model, fc, lo80, hi80, lo95, hi95, hist_x, y_hist, fut_x, H, freq_str, unit, bt, text, bdict, conf, caveats, info, m,
              rel=None, cv=None, h_cv=0):
    name = _NAMES.get(model)
    if name is None:
        name = {"combo_trend": "Average of Theta and exponential-smoothing models",
                "combo_all": "Average of exponential-smoothing, Theta and seasonal models",
                "combo_seasonal": "Average of seasonal smoothing, Theta and seasonal naive"}.get(model, model)
    f, l80, u80, l95, u95 = (_round(a) for a in (fc, lo80, hi80, lo95, hi95))
    last_n = min(H, len(y_hist))
    prev = float(np.sum(y_hist[-last_n:])) if last_n else 0.0
    nxt = float(np.sum(fc))
    ch = (nxt - prev) / abs(prev) * 100 if prev else 0.0
    plural = unit + "s"
    note = (f"Over the next {H} {plural}, the total is expected to be about {abs(ch):.0f}% {'higher' if ch >= 0 else 'lower'} than the last {last_n}. "
            f"The shaded band is a 95% range from backtested errors (an 80% band is also given); forecasts are estimates, not promises.")
    cov = bt.get("coverage")
    return {
        "status": status, "reason": None, "model": model, "model_name": name, "freq": freq_str, "horizon": H, "season_length": m,
        "history": {"x": hist_x, "y": _round(y_hist)},
        "forecast": {"x": fut_x, "y": f, "lower": l95, "upper": u95, "lower80": l80, "upper80": u80, "lower95": l95, "upper95": u95},
        "lower": {"80": l80, "95": l95}, "upper": {"80": u80, "95": u95},
        "backtest": {**bt, "relative_error": None if rel is None else round(float(rel), 3),
                     "cv_mae": None if not cv else {k: round(float(v), 4) for k, v in cv.items()}},
        "baseline_comparison": text, "baseline": bdict, "confidence": conf, "caveats": caveats, "note": note,
    }
