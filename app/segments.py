"""'Which group is different?': the segment-level facts a small owner actually acts on.

Three checks, all plain arithmetic with the numbers shown in each finding's `evidence`:
  * segment_shifts: a group whose recent level moved far more (or in the other direction) than the whole business,
  * ratio_gaps: part-of-whole columns (paid of due, no-shows of appointments, signups of visits) whose ratio is off for one group,
  * rating_gaps: for averaged measures (survey scores), the group that scores lowest or highest.
"""
import re, time
import numpy as np, pandas as pd
from . import drivers as D

UNIT = {"D": "week", "W": "week", "MS": "month", "YS": "year"}


def _fmt(x) -> str:
    x = float(x); a = abs(x)
    return f"{x/1e6:,.1f}M" if a >= 1e7 else f"{x:,.0f}" if a >= 1000 else f"{x:,.1f}" if a >= 10 else f"{x:.2f}"


def _dims(df, p, max_dims=3):
    return [c for c in p["cat_cols"] if 2 <= df[c].nunique() <= 30][:max_dims]


def segment_shifts(df, p, m, d, cost_metric, recent=3, before=6):
    """A segment (>=5% of the total) whose last `recent` periods moved >=25% against the `before` periods, and >=20 points differently from the total."""
    if not d: return []
    freq = "W" if p.get("_freq", "MS") in ("D", "W") else p.get("_freq", "MS")
    rule = {"W": "W", "MS": "MS", "YS": "YS"}[freq]
    x = df[[d, m] + _dims(df, p)].dropna(subset=[d, m])
    if x.empty: return []
    tot = x.set_index(d)[m].resample(rule).sum()
    if len(tot) > 2 and rule == "MS":                      # a partial first/last month would look like a collapse
        lo, hi = x[d].min(), x[d].max()
        if lo > tot.index[0] + pd.Timedelta(days=2): tot = tot.iloc[1:]
        if hi < tot.index[-1] + pd.offsets.MonthEnd(0) - pd.Timedelta(days=2): tot = tot.iloc[:-1]
    if len(tot) < recent + before: return []
    idx = tot.index[-(recent + before):]
    def chg(s):
        s = s.reindex(idx).fillna(0.0); b = s.iloc[:before].mean(); r = s.iloc[before:].mean()
        return (r - b) / abs(b) * 100 if b > 0 else None, b, r
    overall, _, _ = chg(tot)
    if overall is None: return []
    out, total_sum = [], float(x[m].sum())
    for dim in _dims(df, p):
        for seg, g in x.groupby(dim, observed=True):
            if total_sum <= 0 or g[m].sum() / total_sum < 0.05 or len(g) < 15: continue
            if pd.api.types.is_bool_dtype(df[dim]): continue          # a Yes/No flag changing its share is a mix effect, not a group in trouble
            series = g.set_index(d)[m].resample(rule).sum().reindex(idx).fillna(0.0)
            c, b, r = chg(series)
            if c is None or abs(c) < 25 or abs(c - overall) < 20: continue
            sd_b = float(series.iloc[:before].std(ddof=1) or 0)
            if abs(r - b) < 2.5 * sd_b * np.sqrt(1 / before + 1 / recent): continue   # inside this group's own normal ups and downs: not a real shift
            out.append((abs(r - b) * recent, dim, seg, c, b, r))
    out.sort(reverse=True)
    findings, used = [], set()
    unit = UNIT.get(freq, "period")
    for _, dim, seg, c, b, r in out[:6]:
        if (dim, seg) in used or len(findings) >= 2: continue
        used.add((dim, seg)); name = D.seg_label(df, dim, seg)
        desirable = ((c > 0) != cost_metric) != bool(D.BAD_SEGMENT.search(name))
        findings.append({"kind": "driver", "severity": "good" if desirable else "warn",
                         "title": f"{name} {'rose' if c > 0 else 'fell'} {abs(c):.0f}% recently while the total moved {overall:+.0f}%",
                         "detail": f"{name} averaged {_fmt(r)} per {unit} over the last {recent} {unit}s, against {_fmt(b)} in the {before} {unit}s before. "
                                   f"{m} overall moved {overall:+.0f}% over the same periods, so this is specific to {name}.",
                         "action": f"Look at {name} first: " + ("find out what worked and repeat it." if desirable else "find out what changed there and fix it."),
                         "evidence": {"dimension": dim, "segment": name, "recent_avg": r, "before_avg": b, "change_pct": c, "overall_change_pct": overall,
                                      "periods": f"last {recent} vs the {before} before", "unit": unit}})
    return findings


def ratio_gaps(df, p, cost_metric=False):
    """Part-of-whole pairs (a <= b in almost every row): flag the group whose a/b ratio differs most from the overall ratio."""
    cols = [c for c in p["metric_cols"] if c != "records" and df[c].notna().mean() >= 0.9]
    cols = ([p["metric"]] if p["metric"] in cols else []) + [c for c in cols if c != p["metric"]]
    cols = cols[:8]                                    # a 300-column file would mean ~90,000 pairs; the first few measures are the ones that matter
    best, t0 = None, time.monotonic()
    for a in cols:
        for b in cols:
            if a == b: continue
            if time.monotonic() - t0 > 3.0: break                     # never let this check slow an upload down
            x = df[[a, b]].dropna(); x = x[x[b] > 0]
            if len(x) < 40 or x[a].nunique() < 3 or float(x[b].sum()) <= 0 or (x[a] <= x[b] + 1e-9).mean() < 0.95 or (x[a] < 0).any(): continue
            overall = float(x[a].sum() / x[b].sum())
            if overall <= 0 or overall >= 0.999: continue
            bad_ratio = bool(D.BAD_SEGMENT.search(a))                     # more no-shows / returns / cancellations is worse
            for dim in _dims(df, p):
                g = df.loc[x.index].groupby(dim, observed=True)[[a, b]].sum()
                n = df.loc[x.index].groupby(dim, observed=True).size()
                for seg in g.index:
                    if n[seg] < 20 or g.loc[seg, b] / x[b].sum() < 0.05: continue
                    r = float(g.loc[seg, a] / g.loc[seg, b])
                    dev = abs(r - overall)
                    if dev < max(0.10, 0.25 * overall): continue
                    score = dev * float(g.loc[seg, b] / x[b].sum())
                    if best is None or score > best[0]: best = (score, a, b, dim, seg, r, overall, int(n[seg]), bad_ratio)
    if best is None: return []
    _, a, b, dim, seg, r, overall, n, bad_ratio = best
    name = D.seg_label(df, dim, seg); worse = (r > overall) if bad_ratio else (r < overall)
    return [{"kind": "driver", "severity": "warn" if worse else "info",
             "title": f"{name}: {a} is {r * 100:.0f}% of {b}, against {overall * 100:.0f}% overall",
             "detail": f"Across {n:,} records for {name}, {a} makes up {r * 100:.1f}% of {b}; for everyone else together it is closer to {overall * 100:.1f}%.",
             "action": f"Find out what is different for {name}" + (" and what is holding it back." if worse else " and whether the others can copy it."),
             "evidence": {"dimension": dim, "segment": name, "numerator": a, "denominator": b, "ratio": r, "overall_ratio": overall, "records": n}}]


def rating_gaps(df, p, m):
    """For an averaged measure (survey score): the group scoring clearly lowest (or highest) against the overall average."""
    best = None
    overall, sd = float(df[m].mean()), float(df[m].std() or 0)
    if sd <= 0: return []
    for dim in _dims(df, p):
        for seg, g in df.groupby(dim, observed=True)[m]:
            if len(g) < 15: continue
            z = (g.mean() - overall) / (sd / np.sqrt(len(g)))
            if abs(z) >= 3 and abs(g.mean() - overall) >= 0.1 * abs(overall) and (best is None or abs(z) > abs(best[0])): best = (z, dim, seg, float(g.mean()), len(g))
    if best is None: return []
    z, dim, seg, mean, n = best; name = D.seg_label(df, dim, seg); low = z < 0
    return [{"kind": "driver", "severity": "warn" if low else "good",
             "title": f"{name} scores {'lowest' if low else 'highest'} on {m}: {mean:.2f} against {overall:.2f} overall",
             "detail": f"{n:,} responses from {name} average {mean:.2f}, {abs(mean - overall):.2f} {'below' if low else 'above'} the overall average of {overall:.2f}.",
             "action": f"Ask {name} what is driving this score" + (" and fix the biggest complaint first." if low else ", and share what works with the others."),
             "evidence": {"dimension": dim, "segment": name, "mean": mean, "overall_mean": overall, "responses": n, "z": float(z)}}]


def segment_findings(df, p, m, d, freq, cost_metric, mean_metric):
    """Up to three segment-level findings for the main measure. A failed check is skipped, never raised."""
    checks = ([lambda: rating_gaps(df, p, m)] if mean_metric else
              [lambda: segment_shifts(df, {**p, "_freq": freq or "MS"}, m, d, cost_metric), lambda: ratio_gaps(df, p, cost_metric)])
    out = []
    for fn in checks:
        try: out += fn()
        except Exception: pass
    return out[:3]
