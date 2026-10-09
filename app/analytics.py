"""Profiling, starter charts, statistical insights and forecasting. No LLM needed here."""
import io, re, warnings, datetime
import numpy as np, pandas as pd

METRIC_HINT = re.compile(r"amount|sales|revenue|total|donat|expens|cost|profit|qty|quantity|units|price|balance|spend|value|volume|score|rate", re.I)
STRONG_HINT = re.compile(r"amount|revenue|sales|total|donat|expens|cost|profit|spend|volume", re.I)
MEAN_HINT = re.compile(r"price|rate|score|age|temp|ratio|avg|average|pct|percent", re.I)
ID_HINT = re.compile(r"(^|_)(id|uuid|guid|zip|zipcode|postal|phone|ssn|ein|code|index|num|no)(_|$)", re.I)
STRONG_DATE_HINT = re.compile(r"order|trans|invoice|sale|event|created|purchase|bill|payment|record|activity|checkout|revenue|entry", re.I)
DATE_HINT = re.compile(r"date|time|day|month|period|timestamp|dt", re.I)
AVOID_DATE_HINT = re.compile(r"birth|dob|born|expir|delet|cancel|valid_until", re.I)
NA_RE = re.compile(r"^(n/a|na|null|none|#n/a|<na>|-)$", re.I)


def clean(o):
    """Make anything JSON-safe (numpy types, NaN, Timestamps, pd.NA, datetime)."""
    if isinstance(o, dict): return {clean(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)): return [clean(v) for v in o]
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if o is pd.NaT or o is pd.NA: return None
    if isinstance(o, (pd.Timestamp, datetime.datetime, datetime.date)):
        if pd.isna(o): return None
        return o.isoformat()
    if isinstance(o, (np.datetime64,)):
        ts = pd.Timestamp(o)
        return None if pd.isna(ts) else ts.isoformat()
    if isinstance(o, np.bool_): return bool(o)
    if isinstance(o, (np.integer, int)): return int(o)
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    if pd.isna(o): return None
    return o


def read_raw_df(raw: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xls")) or raw[:4] == b"PK\x03\x04" or raw[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        try: return pd.read_excel(io.BytesIO(raw))
        except Exception as e: raise ValueError(f"Could not read Excel file: {e}")

    encodings = ["utf-8-sig", "utf-8", "latin-1", "cp1252", "iso-8859-1"]
    text = None
    for enc in encodings:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError: pass
    if text is None:
        raise ValueError("Could not decode file. Please ensure it is a valid CSV or Excel file.")

    lines = [ln for ln in text.splitlines() if ln.strip()][:20]
    if not lines: raise ValueError("The file has no rows.")

    sample = "\n".join(lines)
    sep = None
    try:
        import csv
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample, delimiters=[",", ";", "\t", "|"])
        sep = dialect.delimiter
    except Exception:
        first_line = lines[0]
        counts = {d: first_line.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        sep = best if counts[best] > 0 else ","

    try: df = pd.read_csv(io.StringIO(text), sep=sep, index_col=False)
    except Exception:
        try: df = pd.read_csv(io.StringIO(text), sep=None, engine="python", index_col=False)
        except Exception as e: raise ValueError(f"Could not parse CSV file: {e}")
    return df


def try_parse_dates(series: pd.Series, col_name: str) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(series):
        if hasattr(series.dt, "tz") and series.dt.tz is not None:
            return series.dt.tz_localize(None)
        return series

    # If already an object series containing date/datetime objects
    if pd.api.types.is_object_dtype(series):
        valid_objs = series.dropna()
        if len(valid_objs) > 0 and isinstance(valid_objs.iloc[0], (datetime.date, datetime.datetime, pd.Timestamp)):
            try: return pd.to_datetime(series, errors="coerce")
            except Exception: pass

    # Numeric series (Excel serial, compact YYYYMMDD, 4-digit years, Unix timestamps)
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) == 0: return None
        # Excel serial dates (e.g. 30000 to 60000)
        if re.search(r"(^|_)(date|dt|time|timestamp|day)(_|$)", col_name, re.I):
            if non_null.between(30000, 60000).mean() > 0.8:
                try:
                    parsed = pd.to_datetime(series, unit="D", origin="1899-12-30", errors="coerce")
                    if parsed.dropna().shape[0] / len(non_null) > 0.8: return parsed
                except Exception: pass
        # Compact YYYYMMDD integers (e.g. 20230115)
        if non_null.between(19000101, 21001231).mean() > 0.8:
            try:
                parsed = pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
                if parsed.dropna().shape[0] / len(non_null) > 0.8: return parsed
            except Exception: pass
        # 4-digit year integers (e.g. 2021, 2022)
        if re.search(r"(^|_)(year|yr|period|date)(_|$)", col_name, re.I):
            if non_null.between(1900, 2100).mean() > 0.8:
                try:
                    parsed = pd.to_datetime(series.astype(str) + "-01-01", format="%Y-%m-%d", errors="coerce")
                    if parsed.dropna().shape[0] / len(non_null) > 0.8: return parsed
                except Exception: pass
        # Unix timestamps (seconds or milliseconds)
        if re.search(r"(^|_)(timestamp|time|date)(_|$)", col_name, re.I):
            if non_null.between(1e9, 2.5e9).mean() > 0.8:
                try: return pd.to_datetime(series, unit="s", errors="coerce")
                except Exception: pass
            elif non_null.between(1e12, 2.5e12).mean() > 0.8:
                try: return pd.to_datetime(series, unit="ms", errors="coerce")
                except Exception: pass
        return None

    # Text / string / object / category series
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
        non_null = series.dropna().astype(str).str.strip()
        if len(non_null) == 0: return None
        if set(non_null.str.lower().unique()).issubset({"true", "false", "yes", "no", "t", "f", "y", "n"}): return None

        # 4-digit years like "2021", "2022"
        if non_null.str.match(r"^(19\d\d|20\d\d|2100)$").mean() > 0.8:
            if re.search(r"(^|_)(year|yr|date|period)(_|$)", col_name, re.I):
                return pd.to_datetime(series.astype(str) + "-01-01", format="%Y-%m-%d", errors="coerce")

        # Compact YYYYMMDD strings (e.g. "20230115")
        if non_null.str.match(r"^(19|20)\d{6}$").mean() > 0.8:
            try:
                parsed = pd.to_datetime(series.astype(str), format="%Y%m%d", errors="coerce")
                if parsed.dropna().shape[0] / len(non_null) > 0.8: return parsed
            except Exception: pass

        # General date patterns (separators like - / . or month words or column name hint)
        has_separators = non_null.str.contains(r"[-/.]|[A-Za-z]{3,}").mean() > 0.7
        col_suggests_date = bool(re.search(r"(^|_)(date|dt|time|timestamp|day|month|created_at|updated_at|order_date|trans_date|invoice_date|period)(_|$)", col_name, re.I))

        if has_separators or col_suggests_date:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(series, format="mixed", errors="coerce")
                if parsed.dropna().shape[0] / len(non_null) > 0.75:
                    years = parsed.dropna().dt.year
                    if (years >= 1800).all() and (years <= 2200).all():
                        if hasattr(parsed.dt, "tz") and parsed.dt.tz is not None:
                            parsed = parsed.dt.tz_localize(None)
                        return parsed
    return None


def try_parse_numeric(series: pd.Series, col_name: str) -> pd.Series | None:
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype)):
        return None
    if ID_HINT.search(col_name): return None
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0: return None
    if non_null.str.match(r"^0\d{2,}").any(): return None
    if set(non_null.str.lower().unique()).issubset({"true", "false", "yes", "no", "t", "f", "y", "n"}): return None

    def clean_num_str(s):
        return (s.astype(str).str.strip()
                .str.replace(r"^[\$€£¥₹\s]+", "", regex=True)
                .str.replace(r"[\$€£¥₹\s]+$", "", regex=True)
                .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
                .str.replace(r"%$", "", regex=True)
                .str.replace(",", "", regex=False)
                .str.replace(" ", "", regex=False)
                .str.replace(r"^-$", "0", regex=True))

    s_cleaned = clean_num_str(non_null)
    num = pd.to_numeric(s_cleaned, errors="coerce")
    valid_ratio = num.dropna().shape[0] / len(non_null)
    if valid_ratio > 0.8:
        full_cleaned = clean_num_str(series)
        parsed = pd.to_numeric(full_cleaned, errors="coerce")
        parsed[series.isna()] = np.nan
        return parsed
    return None


def try_parse_boolean(series: pd.Series) -> pd.Series | None:
    if not (pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)): return None
    non_null = series.dropna().astype(str).str.strip().str.lower()
    if len(non_null) == 0: return None
    unique_vals = set(non_null.unique())
    bool_map = {"true": True, "false": False, "yes": True, "no": False, "y": True, "n": False, "t": True, "f": False}
    if unique_vals.issubset(set(bool_map.keys())) and len(unique_vals) > 0:
        return series.astype(str).str.strip().str.lower().map(bool_map).astype("boolean")
    return None


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing: clean column names, parse types, handle missing values."""
    if df.empty: raise ValueError("The file has no rows.")
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty: raise ValueError("The file contains only empty cells.")

    clean_cols = []
    seen = {}
    for i, c in enumerate(df.columns):
        base = re.sub(r"\W+", "_", str(c).strip()).strip("_").lower() or f"col_{i}"
        if base in seen:
            seen[base] += 1
            clean_cols.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            clean_cols.append(base)
    df.columns = clean_cols

    # Drop artificial index columns exported from dataframes
    for c in list(df.columns):
        if re.match(r"^unnamed_\d+$|^index$", c) and pd.api.types.is_integer_dtype(df[c]):
            if (df[c].dropna() == np.arange(len(df[c].dropna()))).all():
                df = df.drop(columns=[c])

    # Strip whitespace & standardize missing values
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x)
            df[c] = df[c].replace(r"^\s*$", np.nan, regex=True)
            df[c] = df[c].replace(to_replace=NA_RE, value=np.nan)

    # Type detection: Dates FIRST, then formatted numerics, then booleans
    for c in list(df.columns):
        date_series = try_parse_dates(df[c], c)
        if date_series is not None:
            df[c] = date_series
            continue
        num_series = try_parse_numeric(df[c], c)
        if num_series is not None:
            df[c] = num_series
            continue
        bool_series = try_parse_boolean(df[c])
        if bool_series is not None:
            df[c] = bool_series
            continue

    return df.head(200_000)


def load_df(raw: bytes, filename: str) -> pd.DataFrame:
    """Read raw bytes into a DataFrame and preprocess it."""
    return preprocess_df(read_raw_df(raw, filename))


def df_to_preview(df: pd.DataFrame, n: int = 15) -> dict:
    """Convert a DataFrame to a JSON-safe preview dict with first n rows."""
    sample = df.head(n)
    cols = [str(c) for c in sample.columns]
    dtypes = [str(sample[c].dtype) for c in sample.columns]
    rows = []
    for _, row in sample.iterrows():
        rows.append([clean(v) if not isinstance(v, str) else v for v in row.tolist()])
    return {
        "columns": cols,
        "dtypes": dtypes,
        "rows": rows,
        "total_rows": len(df),
        "total_cols": len(df.columns)
    }


def kind(s: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(s): return "date"
    if pd.api.types.is_bool_dtype(s): return "category"
    if pd.api.types.is_numeric_dtype(s): return "numeric"
    # Fallback check for datetime objects
    valid = s.dropna()
    if len(valid) and isinstance(valid.iloc[0], (pd.Timestamp, datetime.date, datetime.datetime)):
        return "date"
    return "category" if s.nunique() <= max(30, 0.05 * len(s)) else "text"


def profile(df: pd.DataFrame) -> dict:
    cols = []
    for c in df.columns:
        s = df[c]; k = kind(s)
        missing = round(float(s.isna().mean()) * 100, 1)
        uniq = int(s.nunique())
        info = {"name": c, "kind": k, "missing_pct": missing, "unique": uniq}
        if k == "numeric":
            valid = s.dropna()
            if len(valid) > 0:
                q1, q3 = valid.quantile([.25, .75]); iqr = q3 - q1
                info["outliers"] = int(((valid < q1 - 3 * iqr) | (valid > q3 + 3 * iqr)).sum()) if iqr > 0 else 0
            else: info["outliers"] = 0
            is_id = False
            if ID_HINT.search(c):
                is_id = True
            elif not METRIC_HINT.search(c) and pd.api.types.is_integer_dtype(s) and s.is_unique and len(s) >= 10:
                diffs = s.diff().dropna()
                if (diffs == 1).all(): is_id = True
            info["id_like"] = is_id
        cols.append(info)

    dates = [c["name"] for c in cols if c["kind"] == "date"]
    metrics = [c["name"] for c in cols if c["kind"] == "numeric" and not c.get("id_like")]
    cats = [c["name"] for c in cols if c["kind"] == "category" and 2 <= c["unique"] <= 30]

    hinted_metrics = [m for m in metrics if STRONG_HINT.search(m)] or [m for m in metrics if METRIC_HINT.search(m)]
    selected_metric = (hinted_metrics or metrics or [None])[0]

    def date_rank(col_name):
        score = 0
        if STRONG_DATE_HINT.search(col_name): score += 10
        if DATE_HINT.search(col_name): score += 5
        if AVOID_DATE_HINT.search(col_name): score -= 10
        return score

    sorted_dates = sorted(dates, key=date_rank, reverse=True)
    selected_date = (sorted_dates or [None])[0]

    out = {"rows": len(df), "columns": cols, "date_cols": sorted_dates, "metric_cols": metrics, "cat_cols": cats,
           "metric": selected_metric, "date": selected_date}
    out["kpis"] = kpis(df, out)
    return out


def kpis(df, p):
    m, d = p["metric"], p["date"]
    out = [{"label": "Rows analysed", "value": len(df), "kind": "int"}]
    if not m: return out
    val_series = df[m].dropna()
    if val_series.empty: return out
    tot = val_series.sum() if agg_for(m) == "sum" else val_series.mean()
    out.append({"label": f"{'Total' if agg_for(m) == 'sum' else 'Average'} {m}", "value": float(tot), "kind": "num"})
    if d:
        s, freq = period_series(df, d, m); unit = {"MS": "month", "W": "week", "D": "day"}[freq]
        if len(s) >= 2 and s.iloc[-2]:
            out.append({"label": f"Latest full {unit}", "value": float(s.iloc[-1]), "kind": "num",
                        "delta": float((s.iloc[-1] - s.iloc[-2]) / abs(s.iloc[-2]) * 100), "vs": f"vs previous {unit}"})
        if len(s) >= 1 and not s.empty and s.notna().any():
            out.append({"label": f"Best {unit}", "value": float(s.max()), "kind": "num", "note": s.idxmax().strftime("%b %Y" if freq == "MS" else "%d %b %Y")})
    return out


def agg_for(metric): return "mean" if MEAN_HINT.search(metric) else "sum"


def period_series(df, date, metric):
    d = df[[date, metric]].dropna()
    if len(d) == 0:
        return pd.Series(dtype=float), "D"
    span = (d[date].max() - d[date].min()).days
    freq = "MS" if span > 180 else "W" if span > 45 else "D"
    s = getattr(d.set_index(date)[metric].resample(freq), agg_for(metric))()
    if agg_for(metric) == "sum":
        s = s.fillna(0.0)
    else:
        s = s.ffill().bfill().fillna(0.0)
    if freq == "MS" and len(s) > 6 and d[date].max() < s.index[-1] + pd.offsets.MonthEnd(0):
        s = s.iloc[:-1]  # drop the incomplete last month so it doesn't look like a crash
    return s, freq


def starter_charts(df, p):
    out, m, d = [], p["metric"], p["date"]
    if not m: return out
    if d:
        s, freq = period_series(df, d, m)
        if len(s) > 1:
            label = {"MS": "month", "W": "week", "D": "day"}[freq]
            out.append({"title": f"{m} by {label}", "type": "line", "freq": freq,
                        "x": [t.strftime("%Y-%m-%d") for t in s.index], "y": s.round(2).tolist()})
    for c in p["cat_cols"][:2]:
        g = getattr(df.groupby(c)[m], agg_for(m))().sort_values(ascending=False).head(8)
        if not g.empty:
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
        if pd.notna(r) and abs(r) > 0.5 and (best is None or abs(best[1])): best = (o, r)
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
    try:
        fit = ExponentialSmoothing(s, trend="add", damped_trend=True, seasonal=seasonal, seasonal_periods=m if seasonal else None).fit()
    except Exception:
        fit = ExponentialSmoothing(s, trend="add").fit()
    fc = fit.forecast(periods)
    sd = float(np.std(fit.resid)) if len(fit.resid) else 0.0
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
