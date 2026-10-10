"""Deterministic, evidence-quoting recommendations for Lumen.   recommend(insights, profile, df) -> list[dict]

Each item: {"priority": 1.., "title": str, "detail": str, "because": str}

Design (see REVIEW.md section D):
  1. every rule turns ONE kind of evidence into ONE candidate with a *key* (who/what it is about) and an *impact* in [0,1]
     (share of the file's total at stake); candidates with the same key are merged, so the same donor / segment / month is never
     recommended twice;
  2. score = impact x weight(kind); sorted; at most 2 data-quality items and 2 per kind-of-rule; top `max_items` returned;
  3. every sentence is built from real names and numbers (from the findings' `evidence` when present, otherwise recomputed
     from df and labelled "(computed from your data)"), never from free text, so nothing can be invented.
Uses only pandas/numpy and the existing Lumen helpers (app.analytics). No LLM.
"""
import re
import numpy as np, pandas as pd

from . import analytics as A

MONTH = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
STATUS_COL = re.compile(r"status|state|stage|paid|payment_status", re.I)
BAD_STATUS = re.compile(r"overdue|unpaid|pending|outstanding|due|failed|declin|cancel|refund|return|bounced|defaulted|late", re.I)
STOCK_COL = re.compile(r"stock|on_hand|inventory|in_hand|balance_qty|closing_qty", re.I)
FLOW_COL = re.compile(r"ship|sold|sales_qty|units|qty|quantity|dispatch|issued|consum|distribut", re.I)
ENTITY_WORD = re.compile(r"donor|customer|client|member|student|supplier|vendor|patient|employee|payer|volunteer", re.I)
NONADDITIVE = re.compile(r"rating|stars?|score|grade|marks|margin|percent|pct|rate$|_rate|ratio|satisf|age$|temperature|height|balance", re.I)
WEIGHT = {"entity": 1.0, "change": 1.0, "trend": 0.7, "season": 0.6, "stock": 1.3, "segment": 0.5, "status": 0.9,
          "refund": 0.7, "costline": 0.8, "anomaly": 0.5, "quality": 0.35}
ERRORS = []                         # rule failures of the last call (a broken rule is skipped, never raised)
PER_KIND_CAP = {"quality": 2, "segment": 1, "anomaly": 1}


# ------------------------------------------------------------------ formatting
def _run(months):
    """Sorted calendar months -> (label like 'Oct-Jan', first month) if they form one contiguous (cyclic) run, else (None, None)."""
    ms = set(int(x) for x in months)
    for st in ms:
        if all((((st - 1) + i) % 12) + 1 in ms for i in range(len(ms))) and ((st - 2) % 12) + 1 not in ms:
            en = ((st - 1 + len(ms) - 1) % 12) + 1
            return (MONTH[st] if len(ms) == 1 else f"{MONTH[st]}-{MONTH[en]}"), st
    return None, None


def _short(x):
    x = float(x); a = abs(x)
    if a >= 1e7: return f"{x / 1e6:.1f}M"
    if a >= 1e5: return f"{x / 1e3:,.0f}k"
    if a >= 1000: return f"{x:,.0f}"
    return f"{x:,.1f}" if a >= 10 and x != round(x) else f"{x:,.0f}" if a >= 10 else f"{x:,.2g}"


class _Fmt:
    def __init__(self, metric):
        cur = {"usd": "$", "inr": "Rs ", "eur": "EUR ", "gbp": "GBP "}.get((re.search(r"_(usd|inr|eur|gbp)$", metric, re.I) or [None, ""])[1].lower(), "")
        self.cur, self.metric = cur, metric
        self.label = re.sub(r"_(usd|inr|eur|gbp)$", "", metric, flags=re.I).replace("_", " ")
        self.money = bool(re.search(r"amount|revenue|sales|donat|cost|expens|spend|profit|total|price|fee|income", metric, re.I))

    def v(self, x, sign=False):
        s = _short(abs(x)); s = f"{self.cur}{s}" if (self.cur and self.money) else s
        return (("+" if x >= 0 else "-") if sign else ("-" if x < 0 else "")) + s

    def pct(self, x, digits=0): return f"{x * 100:.{digits}f}%"


def _ent_word(col): return (ENTITY_WORD.search(col or "") or [None])[0].lower() if ENTITY_WORD.search(col or "") else (col or "name").replace("_name", "").replace("_", " ")


def _d(ts): return pd.Timestamp(ts).strftime("%d %b %Y")


# ------------------------------------------------------------------ small data helpers
def _total_and_agg(df, m): return float(df[m].sum()), A.agg_for(m, df)


def _mega_row(df, m):
    """The single record that is >=10% of the whole metric AND >=5x the next largest (a 'one huge gift/order'), else None."""
    v = df[m].dropna()
    if len(v) < 10 or v.sum() <= 0: return None
    top = v.nlargest(2)
    return top.index[0] if (top.iloc[0] >= 0.10 * v.sum() and top.iloc[0] >= 5 * top.iloc[1]) else None


def _context(df, d, m, s, freq):
    """Seasonality-aware 'how are we doing' comparison. Prefers like-for-like windows over first-third vs last-third."""
    if freq == "MS" and len(s) >= 24:
        c, p = s.iloc[-12:], s.iloc[-24:-12]; lab = ("the last 12 months", "the 12 months before")
    elif freq == "MS" and len(s) >= 15:
        c, p = s.iloc[-3:], s.iloc[-15:-12]; lab = ("the last 3 months", "the same 3 months a year earlier")
    elif len(s) >= 6:
        k = max(2, len(s) // 3); c, p = s.iloc[-k:], s.iloc[:k]; lab = (f"the last {k} periods", f"the first {k} periods (not adjusted for season)")
    else: return None
    if not p.sum(): return None
    step = pd.offsets.MonthBegin(1) if freq == "MS" else (s.index[1] - s.index[0] if len(s) > 1 else pd.Timedelta(days=1))
    return {"cur": float(c.sum()), "prev": float(p.sum()), "pct": float(c.sum() / p.sum() - 1), "labels": lab, "like_for_like": freq == "MS" and len(s) >= 15,
            "cur_range": (c.index[0], c.index[-1] + step), "prev_range": (p.index[0], p.index[-1] + step)}


def _seg_change(df, d, m, dim, cur_rng, prev_rng):
    a = df[(df[d] >= prev_rng[0]) & (df[d] < prev_rng[1])].groupby(dim, observed=True)[m].sum()
    b = df[(df[d] >= cur_rng[0]) & (df[d] < cur_rng[1])].groupby(dim, observed=True)[m].sum()
    return b.sub(a, fill_value=0.0).sort_values(key=lambda x: -x.abs())


def _month_rate(x, d, m, idx, dim=None, seg=None, ref=None):
    """Monthly total divided by the length of the month in reporting units, so 5-Monday months and 28-day months are comparable.
    Weekly/sparse files (<=6 distinct dates a month) are divided by distinct dates, daily files by calendar days."""
    ref = x if ref is None else ref
    per = idx.to_period("M")
    nd = ref.groupby(ref[d].dt.to_period("M"))[d].apply(lambda v: v.dt.normalize().nunique()).reindex(per).fillna(0)
    div = nd if nd[nd > 0].median() <= 6 else pd.Series(per.days_in_month, index=per)
    y = x if seg is None else x[x[dim] == seg]
    tot = y.groupby(y[d].dt.to_period("M"))[m].sum().reindex(per).fillna(0)
    return pd.Series((tot.values / div.replace(0, np.nan).values) * 30.4, index=idx).fillna(0.0)


def _per_unit(df, d, m, p0, p1, c0, c1):
    """(previous, current) metric per reporting unit between two half-open windows; unit = distinct dates (weekly files) or days."""
    out = []
    for a, b in ((p0, p1), (c0, c1)):
        w = df[(df[d] >= a) & (df[d] < b)]
        nd = w[d].dt.normalize().nunique(); days = max((b - a).days, 1)
        out.append((float(w[m].sum()), nd if nd <= 6 else days))
    return out


def _dims(profile, df): return [c for c in profile["cat_cols"] if 3 <= df[c].nunique() <= 30 and not STATUS_COL.search(c) and df[c].nunique() < 0.5 * len(df)]


def _pl(w): return w[:-1] + "ies" if w.endswith("y") and w[-2:-1] not in "aeiou" else w + ("es" if w.endswith(("s", "x")) else "s")


# ------------------------------------------------------------------ rules (each returns a list of candidates)
def _cand(kind, key, impact, title, detail, because): return {"kind": kind, "key": key, "impact": float(max(0, min(1, impact))), "title": title, "detail": detail, "because": [because] if isinstance(because, str) else because}


def r_entity(df, p, f, d, m, ins):
    ent = p.get("entity_col")
    if not ent or ent not in df or A.agg_for(m, df) != "sum": return []
    g = df.groupby(ent)[m].sum().sort_values(ascending=False)
    tot = float(g[g > 0].sum())
    if len(g) < 5 or tot <= 0: return []
    word, out = _ent_word(ent), []
    top1, share1, share5 = g.index[0], g.iloc[0] / tot, g.head(5).sum() / tot
    rows1 = df[df[ent] == top1]
    mega = _mega_row(df, m)
    if share1 >= 0.08 and (len(g) < 2 or g.iloc[0] >= 3 * g.iloc[1]):
        one = len(rows1) == 1 or (mega is not None and df.loc[mega, ent] == top1)
        r = df.loc[mega] if (mega is not None and df.loc[mega, ent] == top1) else rows1.loc[rows1[m].idxmax()]
        where = ", ".join(f"{c} {r[c]}" for c in p["cat_cols"][:2] if pd.notna(r.get(c)))
        when = f" on {_d(r[d])}" if d and pd.notna(r.get(d)) else ""
        yr = ""
        if d and one:
            y = pd.Timestamp(r[d]).year; ytot = float(df[df[d].dt.year == y][m].sum())
            if ytot > 0: yr = f" Without it, {y} would have been {f.v(ytot - r[m])} instead of {f.v(ytot)}."
        out.append(_cand("entity", ("entity", str(top1)), share1,
            f"Secure and thank {top1}: {f.pct(share1)} of all {f.label} comes from this one {word}",
            (f"{top1} {'gave' if f.money and 'donat' in (m + ent).lower() else 'accounts for'} {f.v(g.iloc[0])}"
             f"{' in a single record' + when + (' (' + where + ')' if where else '') if one else ' in total'}.{yr} "
             f"Call or visit them this month, report what their money achieved, and ask about next year's plans. "
             f"Then set a target for {word}s giving {f.v(float(g.iloc[1]))}-sized amounts so that no single {word} is more than a tenth of the total."),
            f"{top1} = {f.v(g.iloc[0])} of {f.v(tot)} ({f.pct(share1)}); next largest {g.index[1]} = {f.v(g.iloc[1])}; {len(g)} {word}s in total (computed from your data)."))
    elif share5 >= 0.25 and len(g) >= 15:
        names = ", ".join(f"{n} ({f.v(v)})" for n, v in g.head(3).items())
        out.append(_cand("entity", ("entity", "top5"), share5 * 0.7,
            f"Look after your top 5 {word}s: they give {f.pct(share5)} of {f.label}",
            f"The biggest are {names}. Make one named person responsible for each, thank them personally, and flag any that go quiet for two periods.",
            f"Top 5 of {len(g)} {word}s = {f.v(g.head(5).sum())} of {f.v(tot)} (computed from your data)."))
    # spikes that are really this one mega record: fold them in instead of listing a separate 'check this spike'
    return out


def r_change(df, p, f, d, m, ins, s, freq):
    out = []
    for x in ins:
        if x.get("kind") != "change" or "evidence" not in x: continue
        e = x["evidence"]; c = e["contributions"]; lead = c[0]; up = e["change"] > 0
        if e["dimension"] not in _dims(p, df): continue                      # "Paid fell" or "INV-024 rose" is not something anyone can act on
        cost = bool(A.COST_HINT.search(m)); good = up != cost
        seas, seas_flag = "", False
        if freq == "MS" and len(s) >= 14:                                   # was the same two-month step there a year ago?
            ly = s.iloc[-13] / s.iloc[-14] - 1 if s.iloc[-14] else 0; now = s.iloc[-1] / s.iloc[-2] - 1 if s.iloc[-2] else 0
            if now and ly * now > 0 and abs(ly) >= 0.6 * abs(now):
                yoy = s.iloc[-1] / s.iloc[-13] - 1
                seas_flag = True
                seas = (f" Careful: the same step last year was {ly * 100:+.0f}% ({s.index[-14].strftime('%b')} to {s.index[-13].strftime('%b')}), so most of this is the normal season. "
                        f"Against {s.index[-13].strftime('%b %Y')} the change is {yoy * 100:+.0f}%.")
        cal = ""; norm_txt = ""
        try:
            win = A.D._windows(s, freq)
            if win:
                (p0, p1), (c0, c1) = A.D._span(win[0], freq), A.D._span(win[1], freq)
                (tp, up_), (tc, uc) = _per_unit(df, d, m, p0, p1, c0, c1)
                if up_ and uc and tp:
                    raw, nrm = tc / tp - 1, (tc / uc) / (tp / up_) - 1
                    if abs(uc / up_ - 1) > 0.1:
                        if raw * nrm <= 0 or abs(nrm) < 0.5 * abs(raw): continue            # the "change" is just a longer/shorter calendar: do not send anyone chasing it
                        cal = f" Allowing for the different length of the two periods ({uc} vs {up_} reporting days/dates) the change is {nrm * 100:+.0f}%."
        except Exception: pass
        seg = lead["segment"]; sh = abs(lead["share_of_change"])
        if sh >= 0.25 and seas_flag:
            verb = f"{s.index[-1].strftime('%B')} {'rose' if up else 'fell'} {abs(e['change'] / e['previous_total']) * 100:.0f}%, mostly the usual season: only dig into {seg} if you changed something there"
            body = (f"{seg} moved from {f.v(lead['previous'])} to {f.v(lead['current'])} ({f.v(lead['change'], True)}), {sh * 100:.0f}% of the total {f.v(e['change'], True)} change ({e['window']})." + seas + cal +
                    f" If you did run a campaign, price change or restock in {seg}, record the result; otherwise no action is needed.")
        elif sh >= 0.25:
            verb = ("Find out what worked in {s} and repeat it" if good else "Find out why {s} {w} and fix it").format(s=seg, w="rose" if up else "fell")
            body = (f"{seg} moved from {f.v(lead['previous'])} to {f.v(lead['current'])} ({f.v(lead['change'], True)}), {sh * 100:.0f}% of the total {f.v(e['change'], True)} change ({e['window']}). "
                    + (f"{c[1]['segment']} added {f.v(c[1]['change'], True)}. " if len(c) > 1 and abs(c[1]["share_of_change"]) >= 0.1 else "")
                    + ("Ask the person who looks after it what changed (price, stock, a campaign, a one-off buyer) and write it down so it can be repeated." if good else
                       "Ask who handles it what changed, and set a weekly check until it is back on track.") + seas + cal)
        else:
            verb = f"Look across {e['dimension']} values: the {abs(e['change'] / e['previous_total']) * 100:.0f}% {'rise' if up else 'fall'} is spread out"
            body = f"No single {e['dimension']} explains it (biggest: {seg} at {sh * 100:.0f}% of the change). Check prices and volumes overall." + seas + cal
        pv = e.get("price_volume")
        if pv and sh >= 0.25:
            tot_ = abs(e["change"])
            if abs(pv["price_effect"]) < 0.05 * tot_: body += " Almost all of it is more or fewer units sold; the price per unit did not move, so look at demand, not pricing."
            elif abs(pv["volume_effect"]) < 0.05 * tot_: body += " Almost all of it is the price per unit, not the number of units sold."
            else: body += f" Split: {f.v(pv['volume_effect'], True)} from units sold, {f.v(pv['price_effect'], True)} from price per unit."
        impact = abs(e["change"]) / max(float(s.mean()), 1e-9) * (0.4 if seas else 1.0)
        out.append(_cand("change", ("change", e["dimension"], seg), impact, verb, body,
                         f"{e['window']}: {f.v(e['current_total'])} vs {f.v(e['previous_total'])} ({f.v(e['change'], True)}); {seg} contributes {f.v(lead['change'], True)} ({sh * 100:.0f}%)."))
    return out


def r_trend(df, p, f, d, m, ins, s, freq):
    ctx = _context(df, d, m, s, freq)
    if not ctx: return []
    mega = _mega_row(df, m); ex = ""
    if mega is not None and d:
        mrow = df.loc[mega]
        if ctx["cur_range"][0] <= mrow[d] < ctx["cur_range"][1]:
            c2 = ctx["cur"] - float(mrow[m]); p2 = c2 / ctx["prev"] - 1
            ex = f" The single {f.v(mrow[m])} record on {_d(mrow[d])} is left out of the headline figure; counting it, the change would be {ctx['pct'] * 100:+.0f}%."
            ctx = dict(ctx, pct=p2, cur=c2, labels=(ctx["labels"][0] + " (without that record)", ctx["labels"][1]))
    if abs(ctx["pct"]) < 0.05: return []
    up = ctx["pct"] > 0; good = up != bool(A.COST_HINT.search(m))
    dims, why = _dims(p, df), ""
    dfx = df.drop(index=mega) if mega is not None else df
    best = None
    for dim in dims:
        ch = _seg_change(dfx, d, m, dim, ctx["cur_range"], ctx["prev_range"])
        if len(ch) < 3: continue
        total = float(ch.sum())
        if total and (best is None or abs(ch.iloc[0] / total) > best[0]): best = (abs(ch.iloc[0] / total), dim, ch)
    seg_txt = ""
    if best and best[0] >= 0.3:
        _, dim, ch = best; seg_txt = f" Apart from that record, most of the change came from {dim} {ch.index[0]} ({f.v(ch.iloc[0], True)}, {best[0] * 100:.0f}% of it)." if mega is not None else f" Most of it came from {dim} {ch.index[0]} ({f.v(ch.iloc[0], True)}, {best[0] * 100:.0f}% of the change)."
    note = "" if ctx["like_for_like"] else " This compares the start and the end of a short history, so part of it may just be the season."
    if up:
        title = f"Prepare for continued growth: {f.label} is {ctx['pct'] * 100:.0f}% {'higher' if good else 'up'} than {ctx['labels'][1]}" if good else f"Cost pressure: {f.label} is up {ctx['pct'] * 100:.0f}% on {ctx['labels'][1]}"
        act = (f"Keep {best[2].index[0]} resourced at this level and write down what is driving it so it can be repeated." if best and best[0] >= 0.3 else "Make sure stock, staff or volunteers can handle this level, and write down what is driving it.") if good else "Find the biggest line behind the increase and negotiate or cut it."
    else:
        title = f"Reverse the slide: {f.label} is {abs(ctx['pct']) * 100:.0f}% down on {ctx['labels'][1]}" if not good else f"Costs are down {abs(ctx['pct']) * 100:.0f}%: lock in the saving"
        act = "List the three biggest losses and contact those customers or donors this week." if not good else "Write down what changed so the saving is not lost."
    return [_cand("trend", ("trend",), abs(ctx["pct"]) if ctx["like_for_like"] else abs(ctx["pct"]) * 0.5, title,
                  f"{ctx['labels'][0].capitalize()} brought {f.v(ctx['cur'])} against {f.v(ctx['prev'])} in {ctx['labels'][1]}." + ex + seg_txt + note + " " + act,
                  f"{ctx['labels'][0]}: {f.v(ctx['cur'])}; {ctx['labels'][1]}: {f.v(ctx['prev'])} ({ctx['pct'] * 100:+.0f}%)" + (" [like-for-like months]" if ctx["like_for_like"] else " [not season-adjusted]") + ".")]


def r_season(df, p, f, d, m, ins, s, freq):
    if freq != "MS" or len(s) < 18 or A.agg_for(m, df) != "sum": return []
    x = df
    mega = _mega_row(df, m)
    if mega is not None: x = df.drop(index=mega)
    sx0, _ = A.period_series(x, d, m)
    sx = _month_rate(x, d, m, sx0.index)
    moy = sx.groupby(sx.index.month).mean(); idx = moy / moy.mean()
    years = len(sx) / 12
    peak = idx[idx >= 1.3].sort_values(ascending=False)
    low = idx[idx <= 0.6].sort_values()
    out = []
    lab, st = _run(peak.index) if len(peak) else (None, None)
    if len(peak) and lab:
        months = lab
        share = float(moy[peak.index].sum() / moy.sum())
        lead_m = MONTH[(st - 2) % 12 + 1]
        out.append(_cand("season", ("season", "overall"), (share - len(peak) / 12) * 1.5,
            f"Plan around your busy season ({months})",
            f"{months} bring {share * 100:.0f}% of a typical year's {f.label} (an even spread would be {len(peak) / 12 * 100:.0f}%); the strongest month, {MONTH[peak.index[0]]}, runs at {peak.iloc[0]:.1f}x an average month. "
            f"Build stock, staff or the campaign calendar from {lead_m}, and avoid scheduling big changes in these months."
            + (f" {MONTH[low.index[0]]} is the quietest ({low.iloc[0]:.1f}x): use it for maintenance, training or a promotion." if len(low) else ""),
            f"Average {f.label} by calendar month over {years:.1f} years" + (f" (excluding the single {f.v(df.loc[mega, m])} record)" if mega is not None else "") + f": peak {MONTH[peak.index[0]]} {f.v(moy[peak.index[0]])} vs overall monthly average {f.v(moy.mean())}."
            + (" Less than 2 full years: treat as indicative." if years < 2 else "")))
    # a segment whose season differs from the business as a whole (e.g. Winter goods in a mixed warehouse)
    best = None
    for dim in _dims(p, df):
        if len(sx) < 12: continue
        for seg in x[dim].dropna().unique():
            share = float(x.loc[x[dim] == seg, m].sum() / max(x[m].sum(), 1e-9))
            if share < 0.10: continue
            sm = _month_rate(x, d, m, sx0.index, dim, seg, ref=x)
            if sm.mean() <= 0: continue
            prof = sm.groupby(sm.index.month).mean(); prof = prof / prof.mean()
            rel = prof / idx.reindex(prof.index)                                  # relative to the whole business's own season
            hi = prof[(prof >= 1.3) & (rel >= 1.15)]
            hl, hst = _run(hi.index) if len(hi) else (None, None)
            same = len(peak) and len(set(hi.index) & set(peak.index)) / len(set(hi.index) | set(peak.index)) >= 0.6   # same months as the whole business: already said above
            if 2 <= len(hi) <= 5 and hl and not same:
                score = (prof.max() - 1) * share
                if best is None or score > best[0]: best = (score, dim, seg, hi, share, prof, hl, hst)
    if best:
        _, dim, seg, hi, share, prof, months, hst = best
        out.append(_cand("season", ("season", dim, str(seg)), best[0] * 0.6,
            f"{seg} has its own season ({months}): order and promote for it early",
            f"{seg} ({share * 100:.0f}% of {f.label}) runs at {hi.max():.1f}x its normal level in {months}, a different pattern from the rest. Plan purchasing and storage about six weeks before {MONTH[hst]}, "
            f"and treat {months} jumps in {seg} as expected, not as a surprise.",
            f"{dim}={seg}: calendar-month profile has peak {hi.max():.1f}x; different months from the whole business; based on {len(sx)} months."))
    return out


def r_stock(df, p, f, d, m, ins):
    """Weeks of cover = latest stock / recent average outflow, then stressed with that product's own seasonal peak in the next 6 months."""
    stock = next((c["name"] for c in p["columns"] if c["kind"] == "numeric" and STOCK_COL.search(c["name"]) and not c.get("id_like")), None)
    flow = next((c for c in p["metric_cols"] if c != stock and FLOW_COL.search(c) and not A.MEAN_TOKENS & set(re.split(r"[^a-z0-9]+", c.lower()))), None)
    dims = [c for c in _dims(p, df) if df[c].nunique() <= 15]
    if not (stock and flow and d and dims): return []
    dim = max(dims, key=lambda c: df[c].nunique())
    last = df[d].max(); per = df[d].drop_duplicates().sort_values()
    if len(per) < 8: return []
    gap = per.diff().median(); unit = "week" if gap >= pd.Timedelta(days=6) else "period"
    stock_now = df[df[d] == last].groupby(dim, observed=True)[stock].sum()
    recent = df[df[d].isin(per.iloc[-8:])].groupby([d, dim], observed=True)[flow].sum().groupby(level=1).mean()
    cover = (stock_now / recent.replace(0, np.nan)).dropna()
    if cover.empty: return []
    idx = A.period_series(df, d, flow)[0].index
    rows = []
    for k in cover.index:
        ratio, pk = 1.0, None
        if len(idx) >= 12:
            prof = _month_rate(df[df[dim] == k], d, flow, idx); prof = prof.groupby(prof.index.month).mean(); prof = prof / prof.mean() if prof.mean() > 0 else prof
            now = [idx[-1].month, idx[-2].month]; nxt = [((idx[-1].month + i - 1) % 12) + 1 for i in range(1, 7)]
            base = float(prof.reindex(now).mean())
            if base > 0 and prof.reindex(nxt).notna().any():
                pk = int(prof.reindex(nxt).idxmax()); ratio = max(1.0, float(prof[pk] / base))
        rows.append((k, float(cover[k]), ratio, pk, float(cover[k] / ratio)))
    rows.sort(key=lambda r: r[4]); k, cv, ratio, pk, adj = rows[0]; med = float(cover.median())
    if not (adj <= 3.0 or cv <= 0.8 * med): return []
    peak_txt = (f" This product normally sells about {ratio:.1f}x as much by {MONTH[pk]}, when the same stock would last only about {adj:.1f} {unit}s (based on {len(idx)} months of history)." if ratio >= 1.25 else "")
    need = max(0.0, 6 * float(recent[k]) * ratio - float(stock_now[k]))
    others = ", ".join(f"{n} ({a_:.1f} {unit}s{' at peak' if r_ >= 1.25 else ''})" for n, c, r_, _, a_ in rows[1:3])
    return [_cand("stock", ("stock", str(k)), 0.12 + (6 - min(adj, 6)) / 30, f"Reorder {k} first: only {cv:.1f} {unit}s of stock left" + (f", {adj:.1f} at the {MONTH[pk]} peak" if ratio >= 1.25 else ""),
                  f"Latest {stock} is {_short(stock_now[k])} against about {_short(recent[k])} {flow} per {unit} over the last 8 {unit}s.{peak_txt} "
                  f"Next thinnest: {others}; typical cover today is {med:.1f} {unit}s. To hold 6 {unit}s of cover (an assumption; use your real resupply time) you would need about {need:,.0f} more units of {k}.",
                  f"{stock} on {_d(last)} / average {flow} per {unit} over the last 8 {unit}s, by {dim}; seasonal ratio from {dim} {k}'s own monthly history (computed from your data).")]


def r_costline(df, p, f, d, m, ins):
    """For a cost/expense measure, the few lines that take most of the money are where a saving is worth chasing."""
    if not A.COST_HINT.search(m) or A.agg_for(m, df) != "sum": return []
    best = None
    for dim in _dims(p, df):
        g = df.groupby(dim, observed=True)[m].sum().sort_values(ascending=False)
        if len(g) >= 5 and (g > 0).all():
            sh3 = g.head(3).sum() / g.sum()
            if sh3 >= 0.5 and (best is None or sh3 > best[0]): best = (sh3, dim, g)
    if not best: return []
    sh3, dim, g = best; tot = g.sum()
    names = ", ".join(f"{k} ({v / tot * 100:.0f}%)" for k, v in g.head(3).items())
    return [_cand("costline", ("costline", dim), sh3 * 0.5, f"Start cost-cutting with the top 3 {_pl(dim)}: they take {sh3 * 100:.0f}% of {f.label}",
                  f"{names} of {f.v(tot)} in total. Ask two suppliers for a quote on these first, or buy in bulk; for illustration, a 5% better price on just these three would save about {f.v(0.05 * g.head(3).sum())} on the amount recorded here.",
                  f"{dim}: top 3 = {f.v(g.head(3).sum())} of {f.v(tot)} ({sh3 * 100:.0f}%) (computed from your data).")]


def r_dow(df, p, f, d, m, ins):
    if A.agg_for(m, df) != "sum": return []
    y = df.set_index(d)[m].resample("D").sum()
    if len(y) < 56 or (y > 0).mean() < 0.5: return []
    dow = y.groupby(y.index.dayofweek).mean(); idx = dow / dow.mean()
    wk = y.groupby(y.index.to_period("W")).transform("mean")
    names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hi = [i for i in idx.index if idx[i] >= 1.15 and float((y[y.index.dayofweek == i] > wk[y.index.dayofweek == i]).mean()) >= 0.7]
    if not hi: return []
    lo = idx.idxmin()
    return [_cand("season", ("dow",), (idx[hi].mean() - 1) * 0.5, f"Staff and stock for {', '.join(names[i] for i in hi)}: the busiest days of the week",
                  f"{', '.join(f'{names[i]} {idx[i]:.1f}x' for i in hi)} an average day's {f.label}, consistently across weeks" + (f"; {names[lo]} is the quietest ({idx[lo]:.1f}x)" if idx[lo] <= 0.85 else "") +
                  ". Put your best-selling items, extra hands and any promotions on those days" + (f" and use {names[lo]} for restocking or admin." if idx[lo] <= 0.85 else "."),
                  f"Average {f.label} by weekday over {len(y) // 7} weeks; {names[hi[0]]} is above its own week's average in {float((y[y.index.dayofweek == hi[0]] > wk[y.index.dayofweek == hi[0]]).mean()) * 100:.0f}% of weeks (computed from your data).")]


def r_status(df, p, f, d, m, ins):
    out = []
    ent = p.get("entity_col") or next((c for c in p["cat_cols"] if re.search(r"client|customer|donor|member|party|payer", c, re.I)), None)
    for c in p["cat_cols"]:
        if not STATUS_COL.search(c) or A.agg_for(m, df) != "sum": continue
        g = df.groupby(c)[m].sum(); tot = float(g.sum())
        bad = g[[bool(BAD_STATUS.search(str(k))) for k in g.index]]
        if tot <= 0 or bad.sum() / tot < 0.03: continue
        OWED = re.compile(r"overdue|unpaid|pending|outstanding|due|late|default", re.I)
        for kind, part in (("owed", bad[[bool(OWED.search(str(k))) for k in bad.index]]), ("lost", bad[[not OWED.search(str(k)) for k in bad.index]])):
            if part.empty or part.sum() / tot < 0.03: continue
            sub = df[df[c].isin(part.index)]
            who = ""
            if ent and ent in df:
                t = sub.groupby(ent)[m].sum().sort_values(ascending=False).head(3); who = " Biggest: " + ", ".join(f"{k} ({f.v(v)})" for k, v in t.items()) + "."
            names = ", ".join(map(str, part.index))
            if kind == "owed":
                title = f"Chase the money that is not settled: {f.v(part.sum())} is '{names}'"
                todo = " Phone or message the largest ones this week, offer a payment date, and set a weekly reminder for anything older than the agreed terms."
            else:
                title = f"Find out why {part.sum() / tot * 100:.0f}% of {f.label} ends as '{names}' ({f.v(part.sum())})"
                todo = " Ask the biggest ones what went wrong, and look for a pattern (a plan, a person, a month) that you can change."
            out.append(_cand("status", ("status", c, kind), float(part.sum() / tot), title,
                f"{part.sum() / tot * 100:.0f}% of all {f.label} ({f.v(part.sum())} of {f.v(tot)}) has status {names}." + who + todo,
                f"{c}: " + "; ".join(f"{k} {f.v(v)} ({v / tot * 100:.0f}%)" for k, v in g.sort_values(ascending=False).items()) + " (computed from your data)."))
    return out


def r_refund(df, p, f, d, m, ins):
    v = df[m]
    if A.agg_for(m, df) != "sum" or (v < 0).sum() < 3: return []
    neg, pos = float(v[v < 0].sum()), float(v[v > 0].sum())
    if pos <= 0 or abs(neg) / pos < 0.03: return []
    dim = next(iter(_dims(p, df)), None); top = ""
    if dim:
        t = df[v < 0].groupby(dim, observed=True)[m].sum().sort_values().head(2); top = f" Most come from {', '.join(f'{k} ({f.v(x)})' for k, x in t.items())}."
    return [_cand("refund", ("refund",), abs(neg) / pos, f"Cut refunds and returns: {f.v(abs(neg))} ({abs(neg) / pos * 100:.0f}% of gross {f.label}) went back",
                  f"{int((v < 0).sum())} records are negative, worth {f.v(neg)}, so reported totals are net of them.{top} Record a reason for each one and fix the top cause (wrong size, damaged, late delivery).",
                  f"Gross {f.v(pos)}, refunds {f.v(neg)}, net {f.v(pos + neg)} (computed from your data).")]


def r_lapsed(df, p, f, d, m, ins):
    """Entities (donors/customers) that used to give regularly and have gone quiet: the cheapest money to win back."""
    ent = p.get("entity_col")
    if not (ent and d) or A.agg_for(m, df) != "sum": return []
    end = df[d].max(); cut = end - pd.Timedelta(days=180)
    g = df.groupby(ent).agg(last=(d, "max"), n=(m, "size"))
    before = df[(df[d] >= cut - pd.Timedelta(days=365)) & (df[d] < cut)].groupby(ent)[m].sum()
    lapsed = g[(g["last"] < cut) & (g["n"] >= 2)].index.intersection(before.index)
    recent_tot = float(df[df[d] >= end - pd.Timedelta(days=365)][m].sum())
    val = float(before.reindex(lapsed).sum())
    if len(lapsed) < 3 or recent_tot <= 0 or val < 0.03 * recent_tot: return []
    top = before.reindex(lapsed).sort_values(ascending=False).head(3); word = _ent_word(ent)
    return [_cand("entity", ("entity", "lapsed"), val / recent_tot, f"Win back {len(lapsed)} {word}s who have gone quiet",
                  f"{len(lapsed)} {word}s gave {f.v(val)} in the year before {_d(cut)} but nothing in the 180 days since. Largest: " + ", ".join(f"{k} ({f.v(v)})" for k, v in top.items()) +
                  f". Send each a short personal message this month, thank them for what they did, and make one specific, easy ask.",
                  f"{len(lapsed)} {word}s with 2+ records, last seen before {_d(cut)} (computed from your data).")]


def r_anomaly(df, p, f, d, m, ins, mega_keys):
    xs = [x for x in ins if x.get("kind") == "anomaly" and isinstance(x.get("evidence"), dict) and "date" in x["evidence"]]
    ent, mega = p.get("entity_col"), _mega_row(df, m)
    keep, folded = [], []
    tot = float(df[m].sum())
    for x in xs:
        e = x["evidence"]
        if mega is not None and d and pd.Timestamp(e["date"]) == pd.Timestamp(df.loc[mega, d]).normalize() and ent:
            folded.append(x); continue
        if (tot > 0 and abs(e["observed"] - e["expected"]) < 0.01 * tot) or e["expected"] < 0.02 * e["observed"]: continue   # too small to matter, or "expected" is ~0 (intermittent data)
        keep.append(x)
    out = []
    if keep:
        lines = []
        for x in keep[:3]:
            e = x["evidence"]; seg = e.get("segment")
            day = df[df[d].dt.normalize() == pd.Timestamp(e["date"])]
            big = ""
            if ent and len(day): r = day.loc[day[m].idxmax()]; big = f"; largest record {r[ent]} {f.v(r[m])}"
            lines.append(f"{_d(e['date'])}: {f.v(e['observed'])} vs about {f.v(e['expected'])} expected" + (f", mostly {seg['value']} ({f.v(seg['change'], True)})" if seg else "") + big)
        out.append(_cand("anomaly", ("anomaly",), 0.05 * len(keep),
            f"Confirm {'this unusual day is' if len(keep) == 1 else str(len(keep)) + ' unusual days are'} real before you plan on it",
            "Open the records for " + "; ".join(lines) + ". If genuine, find the source (campaign, event, big buyer) and decide whether it can be repeated; if it is a typing or duplicate error, fix it so totals and forecasts are not inflated.",
            "Days flagged by the daily anomaly check (evidence: date, observed, expected): " + "; ".join(lines) + "."))
    if folded and ent:   # tell the mega-record story once, inside the entity recommendation
        e = folded[0]["evidence"]; out.append(_cand("entity", ("entity", str(df.loc[mega, ent])), 0.0, "", "", f"Anomaly check agrees: {_d(e['date'])} was {f.v(e['observed'])} vs about {f.v(e['expected'])} expected, because of this record."))
    return out


def r_quality(df, p, f, d, m, ins):
    out, tot = [], float(df[m].sum()) if A.agg_for(m, df) == "sum" else 0
    dims = set(p["cat_cols"])
    for c in p["columns"]:
        if c["name"] in dims and 5 <= c["missing_pct"] < 100:
            miss = df[df[c["name"]].isna()]; amt = float(miss[m].sum()) if tot else 0
            out.append(_cand("quality", ("quality", c["name"]), (amt / tot if tot else c["missing_pct"] / 100) * 0.8,
                f"Fill in the blank {c['name']} values ({len(miss):,} records)",
                f"{len(miss):,} records ({c['missing_pct']}%)" + (f" worth {f.v(amt)} ({amt / tot * 100:.1f}% of {f.label})" if tot else "") + f" have no {c['name']}, so every {c['name']} comparison leaves them out. "
                f"Fill them from your paper or bank records, or add a 'Not recorded' option so the gap is visible.",
                f"{c['name']} is empty in {c['missing_pct']}% of rows (computed from your data)."))
    for c in p["columns"]:
        if c.get("label_variant_groups"):
            ex = c["label_variants"][0]
            out.append(_cand("quality", ("quality", c["name"], "labels"), 0.05, f"Standardise spellings in {c['name']} ({' / '.join(map(repr, ex[:2]))})",
                             f"These are counted as separate values, which splits totals and can hide your real top {c['name']}. Pick one spelling, and use a drop-down list in your sheet from now on.", f"{c['label_variant_groups']} label group(s) with several spellings (profile)."))
    if p.get("duplicates", 0) >= 2 and any(x.get("title", "").endswith("exact duplicates") for x in ins):
        out.append(_cand("quality", ("quality", "dups"), p["duplicates"] / max(len(df), 1), f"Remove {p['duplicates']:,} duplicate rows",
                         "These rows repeat an earlier row including its invoice or order number, so totals count them twice.", f"{p['duplicates']} exact duplicate rows (profile)."))
    return out


def r_segment(df, p, f, d, m, ins, s, freq):
    """Dependence on one segment, with the growth/decline and the 'excluding one huge record' view so the share is not misleading."""
    if A.agg_for(m, df) != "sum": return []
    out, mega = [], _mega_row(df, m)
    ctx = _context(df, d, m, s, freq) if d else None
    for dim in _dims(p, df):
        g = df.groupby(dim, observed=True)[m].sum()
        if (g < 0).any() or g.sum() <= 0: continue
        top, sh = g.idxmax(), g.max() / g.sum()
        k = len(g)
        if sh <= max(0.5, 2.5 / k): continue
        adj = ""
        if mega is not None and df.loc[mega, dim] == top:
            g2 = df.drop(index=mega).groupby(dim, observed=True)[m].sum(); sh2 = g2[top] / g2.sum()
            adj = f" (it is {sh2 * 100:.0f}% if the single {f.v(df.loc[mega, m])} record is excluded)"
            sh = sh2                                                                      # judge dependence on the typical business, not on one gift
            if sh <= max(0.5, 2.5 / k): continue
        grow = ""
        if ctx:
            dfx = df.drop(index=mega) if mega is not None else df
            ch = _seg_change(dfx, d, m, dim, ctx["cur_range"], ctx["prev_range"]); prev = dfx[(dfx[d] >= ctx["prev_range"][0]) & (dfx[d] < ctx["prev_range"][1])].groupby(dim, observed=True)[m].sum()
            rel = (ch / prev.reindex(ch.index).replace(0, np.nan)).dropna(); rel = rel[(g.reindex(rel.index) / g.sum()) >= 0.05]
            if len(rel): grow = f" Fastest-growing meaningful {dim}: {rel.idxmax()} ({rel.max() * 100:+.0f}% vs {ctx['labels'][1]}); slowest: {rel.idxmin()} ({rel.min() * 100:+.0f}%)."
        second = g.drop(top).sort_values(ascending=False)
        out.append(_cand("segment", ("seg", dim, str(top)), sh * 0.6,
            f"Reduce your reliance on {top}: {sh * 100:.0f}% of {f.label} comes from this {dim}",
            f"{top} brings {f.v(g.max())} of {f.v(g.sum())}{adj}; the next is {second.index[0]} at {f.v(second.iloc[0])}.{grow} Pick one smaller {dim} with growth and give it a specific target and budget for next quarter.",
            f"{dim}: " + "; ".join(f"{k} {v / g.sum() * 100:.0f}%" for k, v in g.sort_values(ascending=False).head(4).items()) + " (computed from your data)."))
    return out


# ------------------------------------------------------------------ main
def recommend(insights, profile, df, max_items=6):
    m, d = profile.get("metric"), profile.get("date")
    if not m or m not in df: return []
    f = _Fmt(m)
    cands, ctxs = [], {}; ERRORS.clear()
    if A.agg_for(m, df) == "mean": return []         # an average (rating, rate, price): total-based advice does not apply; the findings' own actions are used
    if NONADDITIVE.search(m) or re.search(r"stock|balance|on_hand", m, re.I):
        return [{"priority": 1, "title": f"Choose the number that matters: '{f.label}' is a score, rate or balance, so adding it up is meaningless",
                 "detail": f"Lumen picked '{m}' as the main measure, but totals and 'share of {m}' do not mean anything for it. Pick a money or quantity column (sales, amount, units) as the main measure, "
                           f"or ask for the average {f.label} by group instead of totals.", "because": f"'{m}' matches a rating/percentage/balance name pattern; all total-based rules were switched off."}]
    small = len(df) < 30
    s, freq = (A.period_series(df, d, m) if d else (pd.Series(dtype=float), "D"))
    rules = [(r_entity, ()), (r_lapsed, ()), (r_change, (s, freq)), (r_trend, (s, freq)), (r_season, (s, freq)), (r_stock, ()), (r_status, ()),
             (r_refund, ()), (r_costline, ()), (r_dow, ()), (r_segment, (s, freq)), (r_anomaly, (None,)), (r_quality, ())]
    for fn, extra in rules:
        if (d is None or small) and fn in (r_dow, r_trend, r_season, r_change, r_stock, r_anomaly, r_segment) or d is None and fn in (r_dow, r_lapsed, r_change, r_trend, r_season, r_stock, r_anomaly): continue
        try: cands += fn(df, profile, f, d, m, insights, *extra)
        except Exception as e:                                           # a broken rule must never take the dashboard down
            ERRORS.append(f"{fn.__name__}: {e!r}")
    # merge by key (no repeats): keep the best-scoring wording, collect every piece of evidence
    merged = {}
    for c in cands:
        if c.get("_error") or (not c["title"] and c["key"] not in merged): continue
        k = c["key"]
        if k in merged:
            b = merged[k]
            if c["title"] and c["impact"] * WEIGHT[c["kind"]] > b["impact"] * WEIGHT[b["kind"]]: c["because"] = b["because"] + c["because"]; merged[k] = c
            else: b["because"] += c["because"]
        else: merged[k] = c
    ranked = sorted((c for c in merged.values() if c["title"]), key=lambda c: -c["impact"] * WEIGHT[c["kind"]])
    # one recommendation per segment value (a 'change' and a 'dependence' item about the same segment would read as a repeat)
    out, per_kind, seen = [], {}, []
    for c in ranked:
        if per_kind.get(c["kind"], 0) >= PER_KIND_CAP.get(c["kind"], 3): continue
        sig = c["key"][-1] if c["key"][0] in ("seg", "change") else None
        if sig and seen.count(sig) >= 2: continue                                       # never more than two recommendations about the same segment value
        if sig: seen.append(sig)
        per_kind[c["kind"]] = per_kind.get(c["kind"], 0) + 1
        out.append(c)
        if len(out) >= max_items: break
    return [{"priority": i + 1, "title": c["title"], "detail": c["detail"], "because": " ".join(dict.fromkeys(c["because"]))} for i, c in enumerate(out)]
