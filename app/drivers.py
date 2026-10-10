"""'What changed and why': exact contribution analysis and conservative daily anomaly detection. Pure pandas/numpy, no LLM.

Everything here is arithmetic that can be checked by hand:
  * bridge(): the change in a total between two equal windows is split into per-segment contributions that add up
    exactly to the change (checked and reported). With a quantity column it is also split into volume vs price effects.
  * daily_anomalies(): days far outside what the recent level and day-of-week pattern predict, with the segment that
    contributed most. The threshold is deliberately high; the false-positive rate on pure noise is measured in tests.
"""
import re
import numpy as np, pandas as pd

BAD_SEGMENT = re.compile(r"overdue|unpaid|pending|cancel|expired|refund|return|fail|late|outstanding|absent|dropout|churn|no[ _-]?show|reject|defect|lost|inactive|unresolved|open", re.I)   # labels where MORE of this is bad


def _lab(v) -> str:
    if isinstance(v, (bool, np.bool_)): return "Yes" if bool(v) else "No"
    return str(v)


def seg_label(df: pd.DataFrame, dim: str, v) -> str:
    """A readable name for one value of a column: 'Hoodie', or 'promo = Yes' for a Yes/No column."""
    return f"{dim} = {_lab(v)}" if dim in df.columns and pd.api.types.is_bool_dtype(df[dim]) else _lab(v)


QTY_HINT = ("qty", "quantity", "units", "unit_count", "count", "pieces", "items")
WINDOW = {"D": 7, "W": 4, "MS": 1, "YS": 1}
UNIT = {"D": "day", "W": "week", "MS": "month", "YS": "year"}
LABEL_FMT = {"D": "%d %b %Y", "W": "week ending %d %b %Y", "MS": "%b %Y", "YS": "%Y"}


def _fmt(x) -> str:
    x = float(x); a = abs(x)
    if a >= 1e7: return f"{x/1e6:,.1f}M"
    if a >= 1e5: return f"{x/1e3:,.0f}k"
    if a >= 1000: return f"{x:,.0f}"
    return f"{x:,.1f}" if a >= 1 else f"{x:.2g}"


def _windows(s: pd.Series, freq: str):
    """Last two equal windows of the (already partial-period-trimmed) series: (prev_idx, cur_idx)."""
    w = WINDOW[freq]
    if len(s) < 2 * w: return None
    return s.index[-2 * w:-w], s.index[-w:]


def _span(idx, freq):
    """Half-open [start, end) date range covered by a block of period labels."""
    off = {"D": pd.Timedelta(days=1), "W": pd.Timedelta(days=7), "MS": pd.offsets.MonthBegin(1), "YS": pd.offsets.YearBegin(1)}[freq]
    # weekly labels are week-ENDING Sundays (pandas default): the period covers the 7 days up to and including the label
    if freq == "W": return idx[0] - pd.Timedelta(days=6), idx[-1] + pd.Timedelta(days=1)
    return idx[0], idx[-1] + off


def daily_grain(df: pd.DataFrame, date: str) -> bool:
    """True when the file has a row for (almost) every day; False for weekly or sparse reporting dates."""
    d = df[date].dropna().dt.normalize()
    return bool(len(d) and d.nunique() / max(d.dt.to_period("M").nunique(), 1) >= 20)


def period_units(df: pd.DataFrame, date: str, start, end, daily: bool) -> tuple[int, str]:
    """(how many comparable units a window holds, what to call one): calendar days for daily files, distinct reporting dates otherwise."""
    if daily: return max((end - start).days, 1), "day"
    n = df[(df[date] >= start) & (df[date] < end)][date].dt.normalize().nunique()
    return max(int(n), 1), "reporting date"


def w_is_month(win) -> bool: return len(win[0]) == 1 and len(win[1]) == 1


def bridge(df: pd.DataFrame, date: str, metric: str, dims: list, freq: str, series: pd.Series, qty: str | None = None, min_change: float = 0.03, cost_metric: bool = False):
    """Explain the change in `metric` (a summable measure) between the last two equal windows. Returns a finding dict or None.
    When the two windows hold different amounts of data (31 days against 28, or 5 weekly rows against 4) the earlier window is scaled
    to the later one's size first, so every number below (the change, each group's share of it, volume vs price) is like for like."""
    win = _windows(series, freq)
    if win is None or not dims: return None
    (p0, p1), (c0, c1) = _span(win[0], freq), _span(win[1], freq)
    d = df[[date, metric, *dims] + ([qty] if qty else [])].dropna(subset=[date, metric])
    prev, cur = d[(d[date] >= p0) & (d[date] < p1)], d[(d[date] >= c0) & (d[date] < c1)]
    pt, ct = float(prev[metric].sum()), float(cur[metric].sum())
    if pt == 0: return None
    scale, unit_name, dp, dc = 1.0, None, 0, 0
    if freq == "MS" and w_is_month(win):
        daily = daily_grain(df, date)
        dp, unit_name = period_units(df, date, p0, p1, daily); dc, _ = period_units(df, date, c0, c1, daily)
        if dp != dc and abs(dc / dp - 1) >= 0.03: scale = dc / dp
        else: unit_name = None
    pt_cmp = pt * scale
    if abs(ct - pt_cmp) / abs(pt_cmp) < min_change: return None
    delta = ct - pt_cmp
    best = None
    for dim in dims:
        a, b = prev.groupby(dim, observed=True)[metric].sum() * scale, cur.groupby(dim, observed=True)[metric].sum()
        contrib = b.sub(a, fill_value=0.0)
        if len(contrib) < 2: continue
        assert abs(contrib.sum() - delta) <= 1e-6 * max(1.0, abs(delta)), "contributions must add up to the change"
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
        focus = abs(top.iloc[0]) / max(abs(contrib).sum(), 1e-12)      # how concentrated the movement is in one segment
        if best is None or focus > best[0]: best = (focus, dim, a, b, top)
    if best is None: return None
    _, dim, a, b, top = best
    rows = [{"segment": seg_label(df, dim, k), "previous": float(a.get(k, 0.0)), "current": float(b.get(k, 0.0)), "change": float(v),
             "share_of_change": float(v / delta)} for k, v in top.head(4).items()]
    unit = UNIT[freq]; w = WINDOW[freq]
    span = f"the last {w} {unit}s" if w > 1 else f"the latest {unit}"
    prior = f"the {w} {unit}s before" if w > 1 else f"the previous {unit}"
    up = delta > 0
    lead = rows[0]
    sign = "+" if lead["change"] >= 0 else "-"
    big = abs(lead["share_of_change"]) >= 0.25
    if abs(lead["share_of_change"]) > 1: lead_txt = f"{lead['segment']} ({sign}{_fmt(abs(lead['change']))}, more than the net change because other values moved the other way)"
    elif big: lead_txt = f"{lead['segment']} ({sign}{_fmt(abs(lead['change']))}, {abs(lead['share_of_change']) * 100:.0f}% of the change)"
    else: lead_txt = "no single value dominates"
    # is the lead segment's own movement good news? (a rising cost is bad; a rising 'overdue' or 'cancelled' segment is bad)
    desirable = ((lead["change"] > 0) != cost_metric) != bool(BAD_SEGMENT.search(lead["segment"]))
    overall_good = up != cost_metric
    pct = abs(delta / pt_cmp) * 100
    if unit_name:
        head = (f"{span.capitalize()} totalled {_fmt(ct)} over {dc} {unit_name}s against {_fmt(pt)} over {dp} {unit_name}s in {prior}. "
                f"Per {unit_name} that is {_fmt(ct / dc)} against {_fmt(pt / dp)}, a change of {'+' if up else '-'}{pct:.0f}%. Compared like for like, ")
    else:
        head = f"{span.capitalize()} totalled {_fmt(ct)} against {_fmt(pt)} in {prior}, a change of {'+' if up else '-'}{_fmt(abs(delta))}. "
    out = {"kind": "change", "severity": "good" if (overall_good and (not big or desirable)) else "warn",
           "title": f"{metric} {'rose' if up else 'fell'} {pct:.0f}% in {span}" + (f" (per {unit_name})" if unit_name else ""),
           "detail": head + (f"split by {dim}, the biggest mover is {lead_txt}." if unit_name else f"Split by {dim}, the biggest mover is {lead_txt}."),
           "action": (f"Look at {lead['segment']} first: " + ("find out what worked and repeat it." if desirable else "find out what went wrong and fix it.")) if big
                     else f"Check several {dim} values, since the change is spread out.",
           "evidence": {"dimension": dim, "window": f"{span} vs {prior}", "previous_total": pt, "current_total": ct, "change": delta,
                        "like_for_like_scale": scale, "unit": unit_name,
                        "contributions": rows, "contributions_sum_to_change": True, "rows_current": int(len(cur)), "rows_previous": int(len(prev))}}
    if qty and qty in d.columns:
        pv = _price_volume(prev, cur, dim, metric, qty, scale)
        if pv:
            out["evidence"]["price_volume"] = pv
            v, pr, tot = pv["volume_effect"], pv["price_effect"], abs(delta)
            if abs(pr) < 0.02 * tot: out["detail"] += f" Almost all of it came from selling {'more' if v >= 0 else 'fewer'} units; the average price per unit barely moved."
            elif abs(v) < 0.02 * tot: out["detail"] += f" Almost all of it came from the average price per unit being {'higher' if pr >= 0 else 'lower'}; units sold barely moved."
            else: out["detail"] += (f" {'+' if v >= 0 else '-'}{_fmt(abs(v))} came from selling {'more' if v >= 0 else 'fewer'} units and "
                                    f"{'+' if pr >= 0 else '-'}{_fmt(abs(pr))} from a {'higher' if pr >= 0 else 'lower'} average price per unit.")
    return out


def _price_volume(prev, cur, dim, metric, qty, scale=1.0):
    """Volume/price split per segment; adds up exactly: sum(Q1*P1 - Q0*P0) = sum((Q1-Q0)*P0) + sum(Q1*(P1-P0))."""
    q0, q1 = prev.groupby(dim, observed=True)[qty].sum() * scale, cur.groupby(dim, observed=True)[qty].sum()
    r0, r1 = prev.groupby(dim, observed=True)[metric].sum() * scale, cur.groupby(dim, observed=True)[metric].sum()
    segs = q0.index.union(q1.index)
    q0, q1, r0, r1 = (x.reindex(segs, fill_value=0.0) for x in (q0, q1, r0, r1))
    if (q1 <= 0).all() or (q0 <= 0).all(): return None
    p0 = (r0 / q0.where(q0 > 0)).fillna(0.0); p1 = (r1 / q1.where(q1 > 0)).fillna(0.0)
    vol = ((q1 - q0) * p0).where(q0 > 0, r1)           # a brand-new segment is all volume
    price = (q1 * (p1 - p0)).where(q0 > 0, 0.0)
    total = float(r1.sum() - r0.sum())
    if abs(float(vol.sum() + price.sum()) - total) > 1e-6 * max(1.0, abs(total)): return None
    return {"volume_effect": float(vol.sum()), "price_effect": float(price.sum())}


def daily_anomalies(df: pd.DataFrame, date: str, metric: str, dims: list, mean_metric: bool = False, z: float = 6.0, max_n: int = 2):
    """Days far outside the recent level x day-of-week pattern. Returns finding dicts, strongest first."""
    d = df[[date, metric] + dims].dropna(subset=[date, metric])
    if d.empty: return []
    day = d.set_index(date)[metric].resample("D")
    y = (day.mean() if mean_metric else day.sum()).astype(float)
    if mean_metric: y = y.dropna()
    if len(y.dropna()) < 28 or (y.index.max() - y.index.min()).days < 27: return []
    nz = y[y > 0]
    if not mean_metric and (len(nz) < 20 or (y == 0).mean() > 0.5): return []      # intermittent data (most days empty): a 'spike' is not meaningful at day level
    base = y.rolling(29, center=True, min_periods=15).median()
    ratio = (y + 1e-9) / (base + 1e-9)
    ok = np.isfinite(ratio) & (base > 0)
    if ok.sum() < 28: return []
    dow = ratio[ok].groupby(ratio[ok].index.dayofweek).median(); dow = dow / dow.mean()
    exp = base * y.index.dayofweek.map(dow).values
    if not mean_metric: exp = exp.clip(lower=0.1 * float(nz.median()))              # never compare with a near-zero expectation (that produced 'a trillion times')
    # Revenue/count style totals have variance that grows with the level (more orders -> bigger swings), so residuals are
    # scaled by sqrt(expected) (Pearson); averages have roughly constant variance, so they are left unscaled.
    scale = np.ones(len(y)) if mean_metric else np.sqrt(np.clip(exp.values, 1e-9, None))
    r = pd.Series((y.values - exp.values) / scale, index=y.index).where(ok)
    med = r.median(); sig = 1.4826 * (r - med).abs().median()
    if not np.isfinite(sig) or sig <= 0: return []
    score = (r - med) / sig
    # a flagged day must be both statistically extreme and practically large (>=40% off what was expected)
    flag = score[(score.abs() > z) & ((y - exp).abs() >= 0.4 * exp.abs())].abs().sort_values(ascending=False).head(max_n)
    out = []
    for t in flag.index:
        obs, e = float(y[t]), float(exp[t]); up = obs > e
        who = _attribute(d, date, metric, dims, t)
        if who and dims: who = (seg_label(df, who[2], who[0]), who[1], who[2])
        out.append({"kind": "anomaly", "severity": "warn", "title": f"{'Spike' if up else 'Drop'} on {t.strftime('%d %b %Y')}",
                    "detail": f"{metric} was {_fmt(obs)} that day, {_how_far(obs, e)} the {_fmt(e)} expected for a {t.strftime('%A')} at that time of year."
                              + (f" Most of the difference came from {who[0]} ({'+' if who[1] >= 0 else '-'}{_fmt(abs(who[1]))})." if who else ""),
                    "action": "Check whether this was a real event (a campaign, a big order or donor) or a data-entry mistake.",
                    "evidence": {"date": t.strftime("%Y-%m-%d"), "observed": obs, "expected": e, "robust_score": float(score[t]),
                                 "segment": None if not who else {"dimension": who[2], "value": who[0], "change": who[1]}}})
    return out


def _how_far(obs: float, exp: float) -> str:
    """'about 8 times' for big spikes (a '718% above' figure is hard to read), else 'about 40% above/below'."""
    if exp > 0 and obs / exp >= 100: return "more than 100 times"
    if exp > 0 and obs / exp >= 3: return f"about {obs / exp:.0f} times"
    return f"about {abs(obs - exp) / abs(exp) * 100:.0f}% {'above' if obs > exp else 'below'}"


def _attribute(d, date, metric, dims, t):
    """Segment whose value that day differs most from its typical daily value in the surrounding +-14 days."""
    best = None
    near = d[(d[date] >= t - pd.Timedelta(days=14)) & (d[date] <= t + pd.Timedelta(days=14))]
    today = near[near[date].dt.normalize() == t.normalize()]
    others = near[near[date].dt.normalize() != t.normalize()]
    n_other = max(others[date].dt.normalize().nunique(), 1)
    for dim in dims:
        a = today.groupby(dim, observed=True)[metric].sum(); b = others.groupby(dim, observed=True)[metric].sum() / n_other
        diff = a.sub(b, fill_value=0.0)
        if diff.empty: continue
        k = diff.abs().idxmax()
        if best is None or abs(diff[k]) > abs(best[1]): best = (str(k), float(diff[k]), dim)
    return best


def find_quantity_column(df: pd.DataFrame, metric: str, metric_cols: list):
    """A numeric column that looks like units sold (never the metric itself)."""
    for c in metric_cols:
        if c != metric and any(h in c.lower().split("_") or c.lower() == h for h in QTY_HINT): return c
    return None
