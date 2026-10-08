"""Profiling, starter charts, statistical insights and forecasting. No LLM needed here."""
import io, re, warnings
import numpy as np, pandas as pd

METRIC_HINT = re.compile(r"amount|sales|revenue|total|donat|expens|cost|profit|qty|quantity|units|price", re.I)
STRONG_HINT = re.compile(r"amount|revenue|sales|total|donat|expens|cost|profit", re.I)
MEAN_HINT = re.compile(r"price|rate|score|age|temp", re.I)


def clean(o):
    """Make anything JSON-safe (numpy types, NaN, Timestamps)."""
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    if isinstance(o, pd.Timestamp): return o.isoformat()
    if o is pd.NaT: return None
    return o


def load_df(raw: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
    elif name.endswith(".csv"):
        try: df = pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError: df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    else:
        raise ValueError("Upload a .csv or .xlsx file.")
    if df.empty: raise ValueError("The file has no rows.")
    df.columns = [re.sub(r"\W+", "_", str(c).strip()).strip("_") or f"col_{i}" for i, c in enumerate(df.columns)]
    for c in df.columns:  # detect date columns stored as text
        if df[c].dtype == object:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.8 and df[c].astype(str).str.contains(r"[-/]|[A-Za-z]{3}").mean() > 0.8:
                df[c] = parsed
    return df.head(200_000)


def kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s): return "date"
    if pd.api.types.is_bool_dtype(s): return "category"
    if pd.api.types.is_numeric_dtype(s): return "numeric"
    return "category" if s.nunique() <= max(30, 0.05 * len(s)) else "text"


def profile(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]; k = kind(s)
        info = {"name": c, "kind": k, "missing_pct": round(float(s.isna().mean()) * 100, 1), "unique": int(s.nunique())}
        if k == "numeric":
            q1, q3 = s.quantile([.25, .75]); iqr = q3 - q1
            info["outliers"] = int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
            info["id_like"] = bool(pd.api.types.is_integer_dtype(s) and s.is_unique)
        cols.append(info)
    dates = [c["name"] for c in cols if c["kind"] == "date"]
    metrics = [c["name"] for c in cols if c["kind"] == "numeric" and not c.get("id_like")]
    cats = [c["name"] for c in cols if c["kind"] == "category" and 2 <= c["unique"] <= 30]
    hinted = [m for m in metrics if STRONG_HINT.search(m)] or [m for m in metrics if METRIC_HINT.search(m)]
    out = {"rows": len(df), "columns": cols, "date_cols": dates, "metric_cols": metrics, "cat_cols": cats,
            "metric": (hinted or metrics or [None])[0], "date": (dates or [None])[0]}
    out["kpis"] = kpis(df, out)
    return out


def kpis(df, p):
    m, d = p["metric"], p["date"]
    out = [{"label": "Rows analysed", "value": len(df), "kind": "int"}]
    if not m: return out
    tot = df[m].sum() if agg_for(m) == "sum" else df[m].mean()
    out.append({"label": f"{'Total' if agg_for(m) == 'sum' else 'Average'} {m}", "value": float(tot), "kind": "num"})
    if d:
        s, freq = period_series(df, d, m); unit = {"MS": "month", "W": "week", "D": "day"}[freq]
        if len(s) >= 2 and s.iloc[-2]:
            out.append({"label": f"Latest full {unit}", "value": float(s.iloc[-1]), "kind": "num",
                        "delta": float((s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]) * 100), "vs": f"vs previous {unit}"})
        out.append({"label": f"Best {unit}", "value": float(s.max()), "kind": "num", "note": s.idxmax().strftime("%b %Y" if freq == "MS" else "%d %b %Y")})
    return out


def agg_for(metric): return "mean" if MEAN_HINT.search(metric) else "sum"


def period_series(df, date, metric):
    d = df[[date, metric]].dropna()
    span = (d[date].max() - d[date].min()).days
    freq = "MS" if span > 180 else "W" if span > 45 else "D"
    s = getattr(d.set_index(date)[metric].resample(freq), agg_for(metric))()
    if freq == "MS" and len(s) > 6 and d[date].max() < s.index[-1] + pd.offsets.MonthEnd(0):
        s = s.iloc[:-1]  # drop the incomplete last month so it doesn't look like a crash
    return s, freq


def starter_charts(df, p):
    out, m, d = [], p["metric"], p["date"]
    if not m: return out
    if d:
        s, freq = period_series(df, d, m)
        label = {"MS": "month", "W": "week", "D": "day"}[freq]
        out.append({"title": f"{m} by {label}", "type": "line", "x": [t.strftime("%Y-%m-%d") for t in s.index], "y": s.round(2).tolist()})
    for c in p["cat_cols"][:2]:
        g = getattr(df.groupby(c)[m], agg_for(m))().sort_values(ascending=False).head(8)
        out.append({"title": f"{m} by {c}", "type": "bar", "x": [str(i) for i in g.index], "y": g.round(2).tolist()})
    return out


def suggested_questions(p):
    m, c, d = p["metric"], (p["cat_cols"] or [None])[0], p["date"]
    q = []
    if m and c: q.append(f"Which {c} brings in the most {m}?")
    if m and d: q.append(f"How did {m} change month over month?")
    if m and c and d: q.append(f"Which {c} is growing fastest?")
    return q or ["Give me a summary of this data"]


def insights(df, p):
    """Statistical findings, each with plain-English detail and a suggested action."""
    f, m, d = [], p["metric"], p["date"]
    for c in p["columns"]:
        if c["missing_pct"] > 20:
            f.append({"kind": "quality", "severity": "warn", "title": f"{c['name']} is {c['missing_pct']}% empty",
                      "detail": f"Over a fifth of rows have no value for {c['name']}, so anything based on it may be misleading.",
                      "action": f"Fill in or remove the missing {c['name']} values before relying on results that use it."})
    if not m: return f
    if d:
        s, freq = period_series(df, d, m)
        n = len(s); unit = {"MS": "month", "W": "week", "D": "day"}[freq]
        if n >= 6:
            k = max(2, n // 3); first, last = s.iloc[:k].mean(), s.iloc[-k:].mean()
            if first:
                ch = (last - first) / abs(first) * 100
                if abs(ch) >= 5:
                    up = ch > 0
                    f.append({"kind": "trend", "severity": "good" if up else "warn",
                              "title": f"{m} is {'up' if up else 'down'} {abs(ch):.0f}% over the period",
                              "detail": f"The average {unit}ly {m} in the most recent {k} {unit}s is {abs(ch):.0f}% {'higher' if up else 'lower'} than in the first {k}.",
                              "action": "Find out what changed and repeat it." if up else f"Look at which {(p['cat_cols'] or ['product'])[0]} or period dropped and act on it first."})
            med = s.median(); mad = (s - med).abs().median()
            if mad > 0:
                z = 0.6745 * (s - med) / mad
                for t, v in z[abs(z) > 3.5].abs().sort_values(ascending=False).head(2).index.to_series().items():
                    val = s[v]
                    f.append({"kind": "anomaly", "severity": "warn", "title": f"Unusual {unit}: {v.strftime('%d %b %Y')}",
                              "detail": f"{m} was {val:,.0f} that {unit}, far from the typical {med:,.0f}.",
                              "action": "Check whether this was a real event (a campaign, a big donor) or a data-entry mistake."})
    for c in p["cat_cols"]:
        g = getattr(df.groupby(c)[m], agg_for(m))()
        if agg_for(m) == "sum" and len(g) >= 3 and g.sum() > 0:
            top = g.idxmax(); share = g.max() / g.sum()
            if share > 0.35:
                f.append({"kind": "driver", "severity": "info", "title": f"{top} drives {share*100:.0f}% of {m}",
                          "detail": f"Across {c}, a single value ({top}) accounts for {share*100:.0f}% of total {m}. You depend heavily on it.",
                          "action": f"Protect what makes {top} work, and test whether other {c} values can grow."})
    others = [x for x in p["metric_cols"] if x != m]
    best = None
    for o in others:
        r = df[m].corr(df[o])
        if pd.notna(r) and abs(r) > 0.5 and (best is None or abs(r) > abs(best[1])): best = (o, r)
    if best:
        f.append({"kind": "driver", "severity": "info", "title": f"{m} moves with {best[0]}",
                  "detail": f"{m} and {best[0]} are strongly {'positively' if best[1] > 0 else 'negatively'} related (correlation {best[1]:.2f}). This shows they move together, not that one causes the other.",
                  "action": f"Track {best[0]} alongside {m} and test changing it."})
    order = {"warn": 0, "good": 1, "info": 2}
    return sorted(f, key=lambda x: order[x["severity"]])[:8]


def forecast(df, date, value, periods=6):
    if date not in df or value not in df: raise ValueError("Unknown column.")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    s, freq = period_series(df, date, value)
    if len(s) < 8: raise ValueError("Need at least 8 time periods (days, weeks or months) to forecast.")
    m = {"MS": 12, "W": 52, "D": 7}[freq]
    seasonal = "add" if len(s) >= 2 * m else None
    fit = ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal=seasonal, seasonal_periods=m if seasonal else None).fit()
    fc = fit.forecast(periods)
    sd = float(np.std(fit.resid))
    h = np.sqrt(np.arange(1, periods + 1))
    lo, hi = fc.values - 1.96 * sd * h, fc.values + 1.96 * sd * h
    if s.min() >= 0: lo = np.maximum(lo, 0)
    prev = s.iloc[-periods:].sum() if agg_for(value) == "sum" else s.iloc[-periods:].mean()
    nxt = fc.sum() if agg_for(value) == "sum" else fc.mean()
    ch = (nxt - prev) / abs(prev) * 100 if prev else 0
    unit = {"MS": "months", "W": "weeks", "D": "days"}[freq]
    fmt = lambda idx: [t.strftime("%Y-%m-%d") for t in idx]
    return {"freq": freq, "history": {"x": fmt(s.index), "y": s.round(2).tolist()},
            "forecast": {"x": fmt(fc.index), "y": fc.round(2).tolist(), "lower": np.round(lo, 2).tolist(), "upper": np.round(hi, 2).tolist()},
            "note": f"Over the next {periods} {unit}, {value} is expected to be about {abs(ch):.0f}% {'higher' if ch >= 0 else 'lower'} than the last {periods}. The shaded band is a 95% range; forecasts are estimates, not promises."}


def demo_df(seed=7):
    """Synthetic shop sales with a trend, seasonality, one spike and a dominant product."""
    rng = np.random.default_rng(seed)
    prods = {"Hoodie": 900, "Tote bag": 250, "Notebook": 120, "Mug": 300, "Sticker pack": 60}
    weights = np.array([.38, .17, .2, .15, .1]); regions = ["North", "South", "Campus", "Online"]
    rows = []
    for day in pd.date_range("2024-01-01", "2025-12-31"):
        winter = 1 + .5 * np.cos((day.dayofyear - 20) / 365 * 2 * np.pi)
        n = rng.poisson(8 * winter * (1 + (day - pd.Timestamp("2024-01-01")).days / 900) * (1.3 if day.dayofweek >= 5 else 1))
        if day == pd.Timestamp("2025-03-14"): n *= 5
        for _ in range(n):
            p = rng.choice(list(prods), p=weights); q = int(rng.integers(1, 4))
            rows.append((day, p, rng.choice(regions, p=[.2, .15, .35, .3]), q, prods[p], q * prods[p]))
    return pd.DataFrame(rows, columns=["order_date", "product", "region", "quantity", "unit_price", "amount"])
