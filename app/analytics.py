"""Profiling, starter charts, statistical insights and forecasting. No LLM needed here."""
import io, re, warnings, datetime, unicodedata
import numpy as np, pandas as pd
from . import drivers as D, segments as SEG

METRIC_HINT = re.compile(r"amount|sales|revenue|total|donat|expens|cost|profit|qty|quantity|units|price|balance|spend|value|volume|score|rate", re.I)
STRONG_HINT = re.compile(r"amount|revenue|sales|total|donat|expens|cost|profit|spend|volume", re.I)
MEAN_HINT = re.compile(r"price|rate|score|age|temp|ratio|avg|average|pct|percent", re.I)
ID_HINT = re.compile(r"(^|_)(id|uuid|guid|zip|zipcode|postal|pin|pincode|phone|mobile|contact|ssn|ein|code|index|num|number|no|invoice|ticket|ref|reference|account|acct|roll|aadhaar|pan|gst)(_|$)", re.I)
CALENDAR_PART = re.compile(r"^(year|yr|month|mon|day|dow|weekday|week|quarter|qtr|hour|minute|fy|period)$", re.I)   # integer calendar columns are not measures
MEAN_TOKENS = {"price", "rate", "score", "age", "temp", "temperature", "ratio", "avg", "average", "pct", "percent", "percentage", "unit"}
COST_HINT = re.compile(r"cost|expens|spend|refund|loss|churn|debt|complaint|defect|return|waste|cancel|overdue|debit|payable|outstanding|unpaid|pending|absent|dropout|no_?show|late", re.I)   # metrics where UP is bad
BAD_SEGMENT = D.BAD_SEGMENT
UNIT = {"MS": "month", "W": "week", "D": "day", "YS": "year"}; ADJ = {"day": "daily", "week": "weekly", "month": "monthly", "year": "yearly"}
MAX_ROWS = 200_000
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
    if isinstance(o, (bool, np.bool_)): return bool(o)
    if isinstance(o, (np.integer, int)): return int(o)
    if isinstance(o, (pd.Timedelta, datetime.timedelta)): return str(pd.Timedelta(o))
    if o.__class__.__name__ == "Decimal": return float(o)
    if isinstance(o, (float, np.floating)): return float(o) if np.isfinite(o) else None
    if pd.isna(o): return None
    return o


TOTAL_RE = re.compile(r"^\s*(grand\s+|sub\s*-?\s*)?totals?\s*:?\s*$", re.I)


def _promote_header(g: pd.DataFrame) -> pd.DataFrame:
    """g was read with header=None. Find the real header row (first row, within 30, that is mostly filled text) and drop trailing TOTAL rows."""
    g = g.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    ncols, hdr = g.shape[1], 0
    for i in range(min(30, len(g) - 1)):
        row = g.iloc[i]
        if row.notna().sum() >= max(2, 0.6 * ncols) and sum(isinstance(v, str) for v in row.dropna()) >= 0.8 * row.notna().sum():
            hdr = i; break
    cols = [str(v).strip() if pd.notna(v) else f"unnamed_{j}" for j, v in enumerate(g.iloc[hdr])]
    body = g.iloc[hdr + 1:].reset_index(drop=True); body.columns = cols; body = body.infer_objects()
    for _ in range(3):   # a TOTAL line at the bottom is double counting, not data
        if len(body) and any(isinstance(v, str) and TOTAL_RE.match(v) for v in body.iloc[-1].tolist()): body = body.iloc[:-1]
    return body


def _read_excel_best_sheet(raw: bytes) -> pd.DataFrame:
    xl = pd.ExcelFile(io.BytesIO(raw)); best = None
    for sheet in xl.sheet_names:
        g = xl.parse(sheet, header=None)
        score = g.dropna(how="all").dropna(axis=1, how="all").size
        if best is None or score > best[0]: best = (score, sheet, g)
    df = _promote_header(best[2]); df.attrs["sheet"] = best[1]; df.attrs["sheets"] = len(xl.sheet_names)
    return df


def read_raw_df(raw: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    is_zip, is_ole = raw[:4] == b"PK\x03\x04", raw[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    if is_zip or is_ole:                                                                 # decide by magic bytes, not by file extension (a .xlsx that is really CSV is fine)
        try: return _read_excel_best_sheet(raw)
        except Exception as e: raise ValueError(f"Could not read Excel file: {e}")

    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"): encodings = ["utf-16"]             # Excel 'Unicode Text'
    else: encodings = ["utf-8-sig", "cp1252", "latin-1"]                               # cp1252 BEFORE latin-1 (latin-1 never fails)
    text = None
    for enc in encodings:
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError: pass
    if text is None:
        raise ValueError("Could not decode file. Please ensure it is a valid CSV or Excel file.")

    head = text[:4000]
    if head.startswith("%PDF") or sum(1 for ch in head if ord(ch) < 32 and ch not in "\t\r\n") > 0.02 * max(len(head), 1):
        raise ValueError("This doesn't look like a CSV or Excel file.")
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


MONTHS = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*"
DATE_LIKE = re.compile(r"^\s*(\d{4}[-/.]\d{1,2}([-/.]\d{1,2})?|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/. ]" + MONTHS + r"[-/. ,]*\d{2,4}|" + MONTHS + r"[-/. ]+\d{1,2}[,-/. ]+\d{2,4}|" + MONTHS + r"[-/. ]+\d{2,4})", re.I)
DMY = re.compile(r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.]\d{2,4}\b")
AMBIGUOUS_DAYFIRST = True    # 03/04/2025 with nothing to disambiguate: day-first (India, EU, UK, AU...). Set False for US-only audiences.


def _to_dt(x, dayfirst=False):
    try: out = pd.to_datetime(x, format="mixed", errors="coerce", dayfirst=dayfirst)
    except ValueError: out = None                         # pandas 3 raises on mixed UTC offsets (DST!)
    if out is None or out.dtype == object:                # pandas 2.2 returns an object column instead; both: convert to UTC, drop the zone
        out = pd.to_datetime(x, format="mixed", errors="coerce", dayfirst=dayfirst, utc=True).dt.tz_localize(None)
    good = lambda o: (o.notna() & (o.dt.year > 1800)).mean()
    if good(out) < 0.5:                   # month-year labels like Jan-25 come back as year 0001 from format="mixed"
        for fmt in ("%b-%y", "%b %y", "%B-%y", "%B %y"):
            alt = pd.to_datetime(x, format=fmt, errors="coerce")
            if good(alt) > good(out): out = alt
    return out


def _detect_dayfirst(non_null: pd.Series) -> bool:
    m = non_null.str.extract(DMY)
    if m[0].notna().mean() < 0.5: return False                       # not a d/m/y text column
    a, b = pd.to_numeric(m[0], errors="coerce"), pd.to_numeric(m[1], errors="coerce")
    a_big, b_big = bool((a > 12).any()), bool((b > 12).any())
    if a_big and not b_big: return True                                # 13/01/2025 can only be day-first
    if b_big and not a_big: return False                               # 01/13/2025 can only be month-first
    if a_big and b_big: return False                                   # inconsistent column: keep pandas default
    # fully ambiguous: prefer the reading that yields the tighter span (real exports are contiguous), then the module default
    spans = []
    for dfirst in (True, False):
        p = _to_dt(non_null, dfirst).dropna()
        spans.append((p.max() - p.min()) if len(p) else pd.Timedelta.max)
    if spans[0] == spans[1]: return AMBIGUOUS_DAYFIRST
    return bool(spans[0] < spans[1])


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
            sample = non_null if len(non_null) <= 300 else non_null.sample(300, random_state=0)
            if sample.str.match(DATE_LIKE).mean() < 0.8: return None       # names / SKUs / version strings: do not even try (was 3 s per 200k-row text column)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dayfirst = _detect_dayfirst(non_null)
                parsed = _to_dt(series, dayfirst)
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

    CUR = r"(?i)(usd|eur|gbp|inr|rs\.?|[\$€£¥₹])"
    def strip_symbols(s):
        s = s.astype(str).str.strip().str.replace("\u2212", "-", regex=False).str.replace(CUR, "", regex=True).str.strip()
        s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True).str.replace(r"^(.*\d)-$", r"-\1", regex=True)          # (12) and 12-  => -12
        return s.str.replace(r"^-\s+", "-", regex=True).str.replace("%", "", regex=False).str.replace(r"[\s']", "", regex=True)

    def decimal_style(s):                        # "1.234,56" / "1234,56" => comma is the decimal mark
        eu = s.str.match(r"^-?\d{1,3}(\.\d{3})*,\d+$").sum() + s.str.match(r"^-?\d+,\d{1,2}$").sum()
        us = s.str.match(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$").sum() + s.str.match(r"^-?\d+\.\d+$").sum()
        return eu > us

    def clean_num_str(s, eu):
        s = strip_symbols(s)
        return s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False) if eu else s.str.replace(",", "", regex=False)

    probe = non_null if len(non_null) <= 300 else non_null.sample(300, random_state=0)      # cheap rejection of text columns before scanning every row
    if pd.to_numeric(clean_num_str(probe, decimal_style(strip_symbols(probe))), errors="coerce").notna().mean() <= 0.8: return None
    eu_style = decimal_style(strip_symbols(non_null))
    s_cleaned = clean_num_str(non_null, eu_style)
    num = pd.to_numeric(s_cleaned, errors="coerce")
    valid_ratio = num.dropna().shape[0] / len(non_null)
    if valid_ratio > 0.8:
        full_cleaned = clean_num_str(series, eu_style)
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
    meta = dict(df.attrs)
    df = df.copy()
    for c in df.columns:      # normalise blanks FIRST so whitespace-only rows / 'N/A' columns are recognised as empty
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].map(lambda x: x.strip() if isinstance(x, str) else x).replace(r"^\s*$", np.nan, regex=True).replace(to_replace=NA_RE, value=np.nan)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    if df.empty: raise ValueError("The file contains only empty cells.")
    df.attrs.update(meta)
    if len(df) > MAX_ROWS: df.attrs["truncated_from"] = len(df); df = df.head(MAX_ROWS)

    def slug(c, i):       # keep letters, digits AND combining marks (Devanagari matras, NFD accents)
        s = unicodedata.normalize("NFC", str(c)).strip()
        s = "".join(ch if (ch.isalnum() or unicodedata.category(ch)[0] == "M") else "_" for ch in s)
        return re.sub(r"_+", "_", s).strip("_").lower() or f"col_{i}"
    clean_cols, used = [], set()
    for i, c in enumerate(df.columns):
        base = slug(c, i); name, n = base, 0
        while name in used: n += 1; name = f"{base}_{n}"
        used.add(name); clean_cols.append(name)
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

    # A 'Total' / 'Grand total' line is a summary of the other rows, not a record: counting it doubles every total.
    removed_totals = 0
    lab_mask = pd.Series(False, index=df.index)
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            lab_mask |= df[c].astype(str).str.match(TOTAL_RE.pattern, flags=re.I)
    if lab_mask.any() and lab_mask.mean() <= 0.05:        # a few summary lines, not a category column that happens to contain 'Total'
        removed_totals += int(lab_mask.sum()); df = df[~lab_mask]

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

    num = df.select_dtypes('number').columns
    if len(num): df[num] = df[num].replace([np.inf, -np.inf], np.nan)   # 'inf' in a CSV is a division error, not a value
    df, extra = _drop_unlabelled_total(df); removed_totals += extra
    df, merged = _merge_label_variants(df)
    df.attrs.update(meta)
    if removed_totals: df.attrs["removed_total_rows"] = removed_totals
    if merged: df.attrs["merged_labels"] = merged
    return df


def _drop_unlabelled_total(df: pd.DataFrame):
    """Drop the last row when every number in it equals the sum of the rows above and it has no real record fields (an unlabelled total line)."""
    if len(df) < 6: return df, 0
    last, rest, numeric = df.iloc[-1], df.iloc[:-1], df.select_dtypes("number").columns
    checked = 0
    for c in numeric:
        v = last[c]
        if pd.isna(v): continue
        tot = rest[c].sum()
        if tot == 0 or not np.isclose(float(v), float(tot), rtol=1e-6, atol=1e-6): return df, 0
        checked += 1
    others = [c for c in df.columns if c not in numeric]
    if checked < 1 or (others and sum(pd.notna(last[c]) for c in others) > max(1, len(others) // 2)): return df, 0
    return df.iloc[:-1], 1


def _merge_label_variants(df: pd.DataFrame):
    """'Dessert' / 'dessert' / ' DESSERT' are one label: use the most common spelling so totals and rankings are not split."""
    merged = []
    for c in df.columns:
        s = df[c]
        if not (pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s)) or pd.api.types.is_bool_dtype(s): continue
        nn = s.dropna()
        if nn.empty or not nn.map(lambda x: isinstance(x, str)).all(): continue
        norm = nn.str.strip().str.lower()
        spellings = nn.groupby(norm).nunique()
        bad = spellings[spellings > 1].index
        if not len(bad): continue
        mapping = {}
        for g in bad:
            counts = nn[norm == g].value_counts(); keep = counts.index[0]
            for alt, n in counts.items():
                if alt != keep: mapping[alt] = keep; merged.append({"column": c, "from": alt, "to": keep, "rows": int(n)})
        df = df.copy(); df[c] = s.replace(mapping)
    merged.sort(key=lambda m: -m["rows"])
    return df, merged[:12]


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


STATUS_LIKE = re.compile(r"status|state|stage|paid|flag|active|promo|received|sent", re.I)
ID_RECORD_HINT = re.compile(r"(^|_)(id|invoice|order|receipt|ref|reference|transaction|txn|ticket|booking)(_|$)", re.I)
ENTITY_HINT = re.compile(r"(^|_)(name|customer|client|donor|member|student|supplier|vendor|patient|employee|account|payer|contact)(_|$)", re.I)


def entity_column(df, cols):
    """A text column that names who the money comes from or goes to (donor, customer, ...): many distinct values, repeated across rows."""
    best = None
    for c in cols:
        if c["kind"] not in ("text", "category") or not ENTITY_HINT.search(c["name"]): continue
        u = c["unique"]
        if 15 <= u <= 0.8 * len(df) and (best is None or u < best[1]): best = (c["name"], u)
    return best[0] if best else None


def profile(df: pd.DataFrame, metric: str | None = None, date: str | None = None) -> dict:
    """Describe the table. `metric` / `date` override Lumen's choice of main measure and date column (the user's correction)."""
    if "records" not in df.columns and any(kind(df[c]) == "date" for c in df.columns):
        df["records"] = 1; df.attrs["synthetic_records"] = True        # 'how many records per period' is always available as a measure
    cols = []
    for c in df.columns:
        s = df[c]; k = kind(s)
        missing = round(float(s.isna().mean()) * 100, 1)
        uniq = int(s.nunique())
        info = {"name": c, "kind": k, "missing_pct": missing, "unique": uniq, "synthetic": bool(c == "records" and df.attrs.get("synthetic_records")),
                "type": "Yes/No" if (pd.api.types.is_bool_dtype(s) or str(s.dtype) == "boolean") else {"date": "Date", "numeric": "Number", "category": "Category"}.get(k, "Text")}
        nn = s.dropna()
        if k == "numeric":
            valid = nn
            if len(valid) > 0:
                heavy = bool(valid.min() > 0 and valid.quantile(.99) / max(float(valid.median()), 1e-12) > 8)    # money-like data: judge on a log scale
                basis = np.log10(valid) if heavy else valid
                q1, q3 = basis.quantile([.25, .75]); iqr = q3 - q1
                out_v = valid[(basis < q1 - 3 * iqr) | (basis > q3 + 3 * iqr)] if iqr > 0 else valid.iloc[0:0]
                info["outliers"] = int(len(out_v))
                info["min"], info["median"], info["max"] = float(valid.min()), float(valid.median()), float(valid.max())
                if len(out_v):
                    info["outlier_examples"] = [float(x) for x in out_v.reindex(out_v.abs().sort_values(ascending=False).index).head(3)]
                    tot = float(valid.sum())
                    info["outlier_share"] = float(out_v.sum() / tot) if tot > 0 and (valid >= 0).all() else None
            else: info["outliers"] = 0
            is_id = False
            if ID_HINT.search(c):
                is_id = True
            elif not METRIC_HINT.search(c) and pd.api.types.is_integer_dtype(s) and s.is_unique and len(s) >= 10:
                diffs = s.diff().dropna()
                if (diffs == 1).all(): is_id = True
            info["id_like"] = is_id
        elif k in ("category", "text") and len(nn):
            txt = nn.astype(str)
            info["examples"] = [str(v) for v in txt.value_counts().head(3).index]
            norm = txt.str.strip().str.lower()
            spellings = txt.groupby(norm).nunique()              # distinct spellings that mean the same label ("Hoodie" / "hoodie")
            bad = spellings[spellings > 1]
            if len(bad):
                info["label_variant_groups"] = int(len(bad))
                info["label_variants"] = [sorted(txt[norm == g].unique().tolist())[:3] for g in bad.index[:3]]
        elif k == "date" and len(nn):
            info["range"] = [pd.Timestamp(nn.min()).strftime("%Y-%m-%d"), pd.Timestamp(nn.max()).strftime("%Y-%m-%d")]
        cols.append(info)

    dates = [c["name"] for c in cols if c["kind"] == "date"]
    metrics = [c["name"] for c in cols if c["kind"] == "numeric" and not c.get("id_like") and not CALENDAR_PART.match(c["name"])]
    cats = [c for c in cols if c["kind"] == "category" and 2 <= c["unique"] <= 50]
    cats.sort(key=lambda c: (bool(STATUS_LIKE.search(c["name"])), c["unique"] < 3, abs(np.log(max(c["unique"], 1) / 8))))   # product / region before status flags; 3-15 values before 40
    cats = [c["name"] for c in cats]
    df.attrs["mean_cols"] = {m for m in metrics if _is_rating_like(df[m], m) or (df[m].dropna().between(0, 1).all() and df[m].nunique() > 2)}
    for m_ in [c for c in metrics if _is_rating_like(df[c], c)]:                    # codes like 99 ('no answer') are not scores: leave them out of every average
        stray = (df[m_] > 11) | (df[m_] < 0)
        if stray.any():
            df.attrs.setdefault("cleared_scores", {})[m_] = int(stray.sum()); df.loc[stray, m_] = np.nan

    df.attrs["rating_cols"] = {m for m in metrics if _is_rating_like(df[m], m)}

    def metric_rank(name):                     # lower is better: a volume or money column beats a price, rate or score
        r = 0
        if STRONG_HINT.search(name): r -= 10
        if VOLUME_HINT.search(name): r -= 6
        if METRIC_HINT.search(name): r -= 2
        if re.search(r"paid|collected|received|actual|net", name, re.I): r -= 3      # what happened beats what was planned or owed
        if re.search(r"(^|_)(due|target|budget|budgeted|planned|expected|quota|goal)(_|$)", name, re.I): r += 3
        if name == "records": r += 5            # counting rows is the fallback measure
        elif agg_for(name, df) == "mean": r += 4 if name in df.attrs["rating_cols"] else 20     # a survey score still beats a bare count
        return r
    selected_metric = min(metrics, key=metric_rank) if metrics else None
    if metric is not None:
        if metric not in metrics: raise ValueError("That column can't be used as the main measure.")
        selected_metric = metric

    def date_rank(col_name):
        score = 0
        if STRONG_DATE_HINT.search(col_name): score += 10
        if DATE_HINT.search(col_name): score += 5
        if AVOID_DATE_HINT.search(col_name): score -= 10
        return score

    sorted_dates = sorted(dates, key=date_rank, reverse=True)
    selected_date = (sorted_dates or [None])[0]
    if date is not None:
        if date not in dates: raise ValueError("That column can't be used as the date.")
        selected_date = date

    out = {"rows": len(df), "columns": cols, "date_cols": sorted_dates, "metric_cols": metrics, "cat_cols": cats,
           "metric": selected_metric, "date": selected_date, "duplicates": int(df.duplicated().sum()),
           "entity_col": entity_column(df, cols)}
    out["kpis"] = kpis(df, out)
    return out


def kpis(df, p):
    m, d = p["metric"], p["date"]
    out = [{"label": "Rows analysed", "value": len(df), "kind": "int"}]
    if not m: return out
    val_series = df[m].dropna()
    if val_series.empty: return out
    tot = val_series.sum() if agg_for(m, df) == "sum" else val_series.mean()
    out.append({"label": f"{'Total' if agg_for(m, df) == 'sum' else 'Average'} {m}", "value": float(tot), "kind": "num"})
    if d:
        s, freq = period_series(df, d, m); unit = UNIT[freq]
        if len(s) >= 2 and s.iloc[-2]:
            last, prev, vs = float(s.iloc[-1]), float(s.iloc[-2]), f"vs previous {unit}"
            if freq == "MS" and agg_for(m, df) == "sum":        # 31 days against 28 (or 5 weekly rows against 4) is not growth
                daily = D.daily_grain(df, d); one = pd.offsets.MonthBegin(1)
                nl, unit_ = D.period_units(df, d, s.index[-1], s.index[-1] + one, daily); np_, _ = D.period_units(df, d, s.index[-2], s.index[-2] + one, daily)
                if nl != np_: last, prev, vs = last / nl, prev / np_, f"vs previous month (per {unit_})"
            out.append({"label": f"Latest full {unit}", "value": float(s.iloc[-1]), "kind": "num", "delta": float((last - prev) / abs(prev) * 100), "vs": vs})
        if len(s) >= 1 and not s.empty and s.notna().any():
            out.append({"label": f"Best {unit}", "value": float(s.max()), "kind": "num", "note": s.idxmax().strftime("%Y" if freq == "YS" else "%b %Y" if freq == "MS" else "week ending %d %b %Y" if freq == "W" else "%d %b %Y")})
    return out


def agg_for(metric, df=None):
    """'mean' for rates, prices, scores and ratings (summing them is meaningless), else 'sum'. profile() records the data-driven cases on df.attrs."""
    if df is not None and metric in df.attrs.get("mean_cols", ()): return "mean"
    return "mean" if MEAN_TOKENS & set(re.split(r"[^a-z0-9]+", str(metric).lower())) else "sum"


VOLUME_HINT = re.compile(r"visit|view|session|signup|sign_up|registration|appointment|attendance|booking|order|download|ticket|hours|yield|units|quantity|qty|count|volume|weight|kg|tonne|sold|shipped|donation|amount|revenue|sales|total|income|profit|spend|expens|cost|fee|paid|collected|balance", re.I)


def _is_rating_like(s: pd.Series, name: str) -> bool:
    """Small whole-number scale (1-5, 0-10): a survey answer or rating, to be averaged. A few stray codes such as 99 ('no answer') are tolerated."""
    v = s.dropna()
    if len(v) < 10: return False
    if VOLUME_HINT.search(name) and not re.search(r"score|rating|satisf|quality|q\d", name, re.I): return False
    inside = v[(v >= 0) & (v <= 11)]
    return bool(len(inside) >= 0.95 * len(v) and (inside == inside.round()).all() and 3 <= inside.nunique() <= 11)


def period_series(df, date, metric):
    d = df[[date, metric]].dropna()
    if len(d) == 0:
        return pd.Series(dtype=float), "D"
    lo, hi = d[date].min(), d[date].max()
    span = (hi - lo).days
    freq = "YS" if (span > 365 * 2 and (d[date].dt.month == 1).all() and (d[date].dt.day == 1).all()) else "MS" if span > 180 else "W" if span > 45 else "D"
    s = getattr(d.set_index(date)[metric].resample(freq), agg_for(metric, df))()
    if agg_for(metric, df) == "sum":
        s = s.fillna(0.0)
    else:
        s = s.ffill().bfill().fillna(0.0)
    monthly_grain = freq == "MS" and d[date].dt.normalize().nunique() <= 4 * max(d[date].dt.to_period("M").nunique(), 1)   # one row per month (e.g. the 15th): nothing is partial
    if freq == "MS" and len(s) > 3 and not monthly_grain:       # drop a first month that starts after the 3rd and a last month that stops >2 days before month end
        if lo > s.index[0] + pd.Timedelta(days=2): s = s.iloc[1:]
        if len(s) > 3 and hi < s.index[-1] + pd.offsets.MonthEnd(0) - pd.Timedelta(days=2): s = s.iloc[:-1]
    if freq == "W" and len(s) > 3:        # bins are labelled by the week-ENDING Sunday: partial if data starts after Tue / ends before Sat
        if lo.normalize() > s.index[0].normalize() - pd.Timedelta(days=5): s = s.iloc[1:]
        if len(s) > 3 and hi.normalize() < s.index[-1].normalize() - pd.Timedelta(days=1): s = s.iloc[:-1]
    return s, freq


def starter_charts(df, p):
    out, m, d = [], p["metric"], p["date"]
    if not m: return out
    if d:
        s, freq = period_series(df, d, m)
        if len(s) > 1:
            label = UNIT[freq]
            out.append({"title": f"{m} by {label}", "type": "line", "freq": freq,
                        "x": [t.strftime("%Y-%m-%d") for t in s.index], "y": s.round(2).tolist()})
    for c in p["cat_cols"][:2]:
        g = getattr(df.groupby(c)[m], agg_for(m, df))().sort_values(ascending=False).head(8)
        if not g.empty:
            out.append({"title": f"{m} by {c}", "type": "bar", "x": [D.seg_label(df, c, i) for i in g.index], "y": g.round(2).tolist()})
    ent = p.get("entity_col")
    if ent:                                                       # who matters most: top donors / customers / members
        g = getattr(df.groupby(ent)[m], agg_for(m, df))().sort_values(ascending=False).head(8)
        if not g.empty: out.append({"title": f"Top {ent} by {m}", "type": "bar", "x": [str(i) for i in g.index], "y": g.round(2).tolist()})
    h = histogram(df[m])
    if h: out.append({"title": h["title"].format(m=m), "type": "hist", "x": h["x"], "y": h["y"]})
    return out


def histogram(s: pd.Series, bins: int = 12):
    """Counts per value range. Heavy-tailed money data gets log-spaced ranges so one huge value does not flatten the picture."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    v = v[np.isfinite(v)]
    if len(v) < 20 or v.nunique() < 5: return None
    lo, hi = float(v.min()), float(v.max())
    if lo <= 0 and v.min() < 0: v = v; edges = np.linspace(v.quantile(.01), v.quantile(.99), bins + 1); logged = False
    elif lo > 0 and hi / max(float(v.median()), 1e-9) > 20: edges = np.geomspace(lo, hi, bins + 1); logged = True
    else: edges = np.linspace(lo, hi, bins + 1); logged = False
    if not np.all(np.diff(edges) > 0): return None
    counts, _ = np.histogram(v.clip(edges[0], edges[-1]), bins=edges)
    f = lambda x: f"{x:,.0f}" if abs(x) >= 100 else f"{x:,.1f}" if abs(x) >= 1 else f"{x:.2g}"
    return {"title": "How {m} values are spread out" + (" (ranges grow by multiples because a few values are far larger)" if logged else ""),
            "x": [f"{f(a)} to {f(b)}" for a, b in zip(edges[:-1], edges[1:])], "y": [int(c) for c in counts]}


def suggested_questions(p):
    m, c, d = p["metric"], (p["cat_cols"] or [None])[0], p["date"]
    q = []
    if m and c: q.append(f"Which {c} brings in the most {m}?")
    if m and d: q.append(f"How did {m} change month over month?")
    if m and c and d: q.append(f"Which {c} is growing fastest?")
    return q or ["Give me a summary of this data"]


def _n(x):
    x = float(x); return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.1f}" if abs(x) >= 1 else f"{x:.3g}"


def _record_table(df, p, m, d, n_out):
    """The largest records behind an 'unusual values' finding, as a small table: the user can open the rows instead of taking the count on trust."""
    cols = [c for c in ([d] if d else []) + ([p["entity_col"]] if p.get("entity_col") else []) + p["cat_cols"][:2] + [m] if c in df.columns]
    cols = list(dict.fromkeys(cols))[:6]
    top = df.loc[df[m].nlargest(min(10, n_out)).index, cols]
    rows = [[(v.strftime("%d %b %Y") if isinstance(v, pd.Timestamp) else v) if pd.notna(v) else "" for v in r] for r in top.itertuples(index=False)]
    return {"record_columns": cols, "records": rows}


def insights(df, p):
    """Statistical findings, each with plain-English detail and a suggested action."""
    f, m, d = [], p["metric"], p["date"]
    freq = None
    if df.attrs.get("truncated_from"):
        f.append({"kind": "quality", "severity": "warn", "title": f"Only the first {len(df):,} of {df.attrs['truncated_from']:,} rows were analysed",
                  "detail": "Totals and trends exclude the remaining rows.", "action": "Split the file by year or filter it, then upload again."})
    if df.attrs.get("sheets", 1) > 1:
        f.append({"kind": "quality", "severity": "info", "title": f"Read sheet '{df.attrs['sheet']}' of {df.attrs['sheets']}", "detail": "Other sheets in the workbook were not analysed.", "action": "Upload other sheets separately if you need them."})
    key_cols = {m, d, p.get("entity_col")}
    for c in p["columns"]:
        if 20 < c["missing_pct"] < 100 and c["name"] in key_cols:          # optional columns (comments, discount code) are legitimately blank: shown in Data health only
            f.append({"kind": "quality", "severity": "warn", "title": f"{c['name']} is {c['missing_pct']}% empty",
                      "detail": f"Over a fifth of rows have no value for {c['name']}, so anything based on it may be misleading.",
                      "action": f"Fill in or remove the missing {c['name']} values before relying on results that use it."})
    dups = p.get("duplicates", 0)
    has_id = any(ID_RECORD_HINT.search(c["name"]) and c["kind"] != "date" and c["unique"] >= 0.9 * max(len(df) - dups, 1) for c in p["columns"])   # a real record number: almost every value distinct
    if has_id and dups >= 2 and dups / max(len(df), 1) >= 0.005:     # identical rows INCLUDING an order/invoice number are double entries; without one they can be genuine repeat sales
        f.append({"kind": "quality", "severity": "warn", "title": f"{dups:,} rows are exact duplicates",
                  "detail": "Each of these rows repeats every value of another row, including its order or invoice number, so totals may be counted twice.",
                  "action": "Check whether they are double entries and remove the extras before trusting totals."})
    for c in p["columns"]:
        if c.get("label_variant_groups"):
            ex = c["label_variants"][0]
            f.append({"kind": "quality", "severity": "warn", "title": f"{c['name']} has inconsistent labels",
                      "detail": f"Spellings such as {' and '.join(repr(x) for x in ex[:2])} are counted as different values, which splits totals and rankings.",
                      "action": f"Standardise the spelling in {c['name']} so each group is counted once."})
    if not m: return f
    mc = next((c for c in p["columns"] if c["name"] == m), {})
    if mc.get("outliers"):
        share, ex = mc.get("outlier_share"), mc.get("outlier_examples", [])
        big = df.loc[df[m].nlargest(min(3, mc["outliers"])).index]
        who = []
        for _, r in big.iterrows():
            parts = [pd.Timestamp(r[d]).strftime("%d %b %Y")] if d and pd.notna(r[d]) else []
            parts += [str(r[c]) for c in ([p["entity_col"]] if p.get("entity_col") else []) + p["cat_cols"][:1] if pd.notna(r[c])]
            who.append(f"{_n(r[m])} ({', '.join(parts)})" if parts else _n(r[m]))
        f.append({"kind": "quality", "severity": "warn" if (share or 0) >= 0.2 else "info", "title": f"{mc['outliers']:,} unusual {m} values",
                  "detail": f"These records are far outside the normal range (largest: {'; '.join(who)})."
                            + (f" Together they make up {share * 100:.0f}% of total {m}." if share and share >= 0.05 else ""),
                  "action": "Check the largest records to confirm they are genuine (for example a big donor or bulk order) and not typing mistakes.",
                  "evidence": {"column": m, "count": mc["outliers"], "examples": ex, "share_of_total": share, **_record_table(df, p, m, d, mc["outliers"])}})
    if agg_for(m, df) == "sum":
        neg = df[m][df[m] < 0]
        if len(neg) >= 3:
            f.append({"kind": "quality", "severity": "info", "title": f"{len(neg):,} entries are negative ({_n(neg.sum())} in total)",
                      "detail": f"These look like refunds, reversals or corrections. They are included in every total of {m}, so totals are net figures.",
                      "action": "Check that these entries are real refunds or corrections, and decide whether you want totals before or after them."})
    ent = p.get("entity_col")
    if ent and agg_for(m, df) == "sum":
        g = df.groupby(ent)[m].sum().sort_values(ascending=False)
        if len(g) >= 15 and (g >= 0).all() and g.sum() > 0:
            k = 5; share = float(g.head(k).sum() / g.sum())
            if share >= 0.25:
                f.append({"kind": "driver", "severity": "info", "title": f"Top {k} {ent} give {share * 100:.0f}% of {m}",
                          "detail": f"{', '.join(str(x) for x in g.index[:3])} lead the list. Losing one of them would be felt.",
                          "action": f"Look after your top {ent} personally and watch for any that stop appearing.",
                          "evidence": {"entity": ent, "top": [{"name": str(i), "value": float(v)} for i, v in g.head(k).items()], "share": share}})
    if d:
        s, freq = period_series(df, d, m)
        n = len(s); unit = UNIT[freq]
        if agg_for(m, df) == "sum" and p["cat_cols"]:       # what changed between the last two equal windows, and which segment drove it
            try: chg = D.bridge(df, d, m, p["cat_cols"][:3], freq, s, D.find_quantity_column(df, m, p["metric_cols"]), cost_metric=bool(COST_HINT.search(m)))
            except Exception: chg = None
            if chg: f.append(chg)
        try: daily = D.daily_anomalies(df, d, m, p["cat_cols"][:3], mean_metric=agg_for(m, df) == "mean")
        except Exception: daily = []
        f += daily
        if n >= 6:
            k = max(2, n // 3)
            def trend_pct(series):
                first, last = series.iloc[:k].mean(), series.iloc[-k:].mean()
                return (last - first) / abs(first) * 100 if first else None
            ch = trend_pct(s); counted = None
            if ch is not None and agg_for(m, df) == "sum":
                top = df[m].nlargest(2)             # one huge record (a single big gift) should not define the headline trend
                if len(top) == 2 and df[m].sum() > 0 and top.iloc[0] >= 0.1 * df[m].sum() and top.iloc[0] >= 5 * max(top.iloc[1], 1e-9):
                    s2, _ = period_series(df.drop(index=top.index[0]), d, m)
                    ch2 = trend_pct(s2) if len(s2) == n else None
                    if ch2 is not None and abs(ch2 - ch) >= 15:
                        row = df.loc[top.index[0]]
                        counted = f"One record ({_n(top.iloc[0])}{', ' + pd.Timestamp(row[d]).strftime('%d %b %Y') if pd.notna(row[d]) else ''}) is left out of this figure; counting it the change would be {ch:+.0f}%."
                        ch = ch2
            if ch is not None and abs(ch) >= 5:
                up = ch > 0; good = up != bool(COST_HINT.search(m))
                f.append({"kind": "trend", "severity": "good" if good else "warn",
                          "title": f"{m} is {'up' if up else 'down'} {abs(ch):.0f}% over the period",
                          "detail": f"The average {ADJ[unit]} {m} in the most recent {k} {unit}s is {abs(ch):.0f}% {'higher' if up else 'lower'} than in the first {k}." + (" " + counted if counted else ""),
                          "action": "Find out what changed and repeat it." if good else f"Look at which {(p['cat_cols'] or ['product'])[0]} or period {'rose' if up else 'dropped'} and act on it first."})
            med = s.median(); mad = (s - med).abs().median()
            if mad > 0 and not daily:                 # the daily check above is finer; only fall back to period-level when it found nothing
                z = 0.6745 * (s - med) / mad
                for t, v in z[abs(z) > 3.5].abs().sort_values(ascending=False).head(2).index.to_series().items():
                    val = s[v]
                    f.append({"kind": "anomaly", "severity": "warn", "title": f"Unusual {unit}: {v.strftime('%d %b %Y')}",
                              "detail": f"{m} was {_n(val)} that {unit}, far from the typical {_n(med)}.",
                              "action": "Check whether this was a real event (a campaign, a big donor) or a data-entry mistake."})
    for c in p["cat_cols"]:
        g = getattr(df.groupby(c)[m], agg_for(m, df))()
        if agg_for(m, df) == "sum" and len(g) >= 3 and (g >= 0).all() and g.sum() > 0:
            top = D.seg_label(df, c, g.idxmax()); share = g.max() / g.sum()
            if share > max(0.35, 1.5 / len(g)):
                f.append({"kind": "driver", "severity": "info", "title": f"{top} drives {share*100:.0f}% of {m}",
                          "detail": f"Across {c}, a single value ({top}) accounts for {share*100:.0f}% of total {m}. You depend heavily on it.",
                          "action": f"Protect what makes {top} work, and test whether other {c} values can grow."})
    others = [x for x in p["metric_cols"] if x not in (m, "records")]
    best = None
    for o in others:
        r = df[m].corr(df[o])
        if pd.isna(r) or len(df) < 50 or not (0.5 < abs(r) < 0.95) or re.search(r"budget|target|planned|expected|quota|cogs|(^|_)cost|(^|_)total|pending|remaining|balance|(^|_)due|transactions?|orders?|count", o, re.I): continue   # near-1.0 = a component or copy of the measure, not a driver
        if best is None or abs(r) > abs(best[1]): best = (o, r)
    if best:
        f.append({"kind": "driver", "severity": "info", "title": f"{m} moves with {best[0]}",
                  "detail": f"{m} and {best[0]} are strongly {'positively' if best[1] > 0 else 'negatively'} related (correlation {best[1]:.2f}). This shows they move together, not that one causes the other.",
                  "action": f"Watch {best[0]} alongside {m}; if you can change {best[0]}, try it on a small scale first and see whether {m} follows."})
    f += SEG.segment_findings(df, p, m, d, freq, bool(COST_HINT.search(m)), agg_for(m, df) == "mean")      # which group is different?
    order = {"warn": 0, "good": 1, "info": 2}
    return sorted(f, key=lambda x: (x["kind"] == "quality", order[x["severity"]], x["kind"] != "change"))[:8]   # data-quality notes never crowd out findings


def forecast(df, date, value, periods=6):
    """Backtest-selected forecast (see app/forecasting.py). Raises ValueError with a plain-English reason when it cannot forecast."""
    if date not in df or value not in df: raise ValueError("Unknown column.")
    if not pd.api.types.is_datetime64_any_dtype(df[date]): raise ValueError(f"'{date}' is not a date column.")
    if not pd.api.types.is_numeric_dtype(df[value]) or pd.api.types.is_bool_dtype(df[value]): raise ValueError(f"'{value}' is not a numeric column.")
    from .forecasting import forecast_series
    s, freq = period_series(df, date, value)
    observed = int((df[[date, value]].dropna().set_index(date)[value].resample(freq).count() > 0).sum())
    if len(s) < 8 or observed < 8: raise ValueError("Need at least 8 time periods with data (days, weeks or months) to forecast.")
    mean_metric = agg_for(value, df) == "mean"
    r = forecast_series(s, horizon=periods, freq=freq, agg="mean" if mean_metric else "sum")
    if r.get("status") == "refused": raise ValueError(r.get("reason") or "This series cannot be forecast.")
    if mean_metric and r.get("note"): r["note"] = r["note"].replace("the total is expected", "the average is expected")
    return r


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
