"""Regression tests for the problems found by black-box testing on realistic small-organisation files."""
import numpy as np, pandas as pd
import pytest
from app import analytics as A, drivers as D, segments as SEG


def sales(n=240, seed=1, **extra):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"order_date": pd.date_range("2025-01-01", periods=n), "item": rng.choice(["Tea", "Cake", "Soup"], n),
                       "store": rng.choice(["North", "South"], n), "amount": rng.integers(50, 400, n).astype(float)})
    for k, v in extra.items(): df[k] = v
    return df


def csv_bytes(df): return df.to_csv(index=False).encode()


# ---- total rows -------------------------------------------------------------------------------------------------
def test_labelled_total_row_in_a_csv_is_removed_not_double_counted():
    df = sales(); true_total = df["amount"].sum()
    with_total = pd.concat([df, pd.DataFrame([{"order_date": "TOTAL", "item": None, "store": None, "amount": true_total}])], ignore_index=True)
    out = A.preprocess_df(A.read_raw_df(csv_bytes(with_total), "x.csv"))
    assert out.attrs["removed_total_rows"] == 1 and out["amount"].sum() == pytest.approx(true_total) and len(out) == len(df)


def test_unlabelled_sum_row_is_removed_but_a_normal_last_row_is_kept():
    df = sales(); tot = pd.DataFrame([{"order_date": None, "item": None, "store": None, "amount": df["amount"].sum()}])
    out = A.preprocess_df(A.read_raw_df(csv_bytes(pd.concat([df, tot], ignore_index=True)), "x.csv"))
    assert out.attrs.get("removed_total_rows") == 1 and len(out) == len(df)
    keep = A.preprocess_df(A.read_raw_df(csv_bytes(df), "x.csv")); assert "removed_total_rows" not in keep.attrs and len(keep) == len(df)


def test_a_category_literally_named_total_is_not_deleted():
    df = sales(); df["item"] = df["item"].replace("Soup", "Total")        # 1/3 of rows: a category, not a summary line
    assert len(A.preprocess_df(A.read_raw_df(csv_bytes(df), "x.csv"))) == len(df)


# ---- spelling variants ------------------------------------------------------------------------------------------
def test_case_variants_are_merged_to_the_common_spelling_and_reported():
    df = sales(); df.loc[df.index[:10], "item"] = "tea"; df.loc[df.index[10:14], "item"] = " TEA "
    out = A.preprocess_df(A.read_raw_df(csv_bytes(df), "x.csv"))
    assert set(out["item"].unique()) == {"Tea", "Cake", "Soup"} and out.attrs["merged_labels"][0]["to"] == "Tea"


# ---- main measure ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cols,expected", [
    ({"visits": 1000, "signups": 30, "bounce_rate": 0.55}, "visits"),
    ({"fee_due": 500, "fee_paid": 430}, "fee_paid"),
    ({"budgeted": 100, "actual_spend": 120}, "actual_spend"),
    ({"price_paid": 40}, "records"),                                       # nothing countable: count registrations per period
])
def test_measure_choice(cols, expected):
    rng = np.random.default_rng(3); n = 120
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n), "group": rng.choice(["a", "b"], n),
                       **{k: (v * rng.uniform(.5, 1.5, n)).round(2) for k, v in cols.items()}})
    assert A.profile(A.preprocess_df(df))["metric"] == expected


def test_ratings_are_averaged_and_stray_codes_are_ignored():
    rng = np.random.default_rng(4); n = 300
    df = pd.DataFrame({"dept": rng.choice(["Arts", "Commerce", "Science"], n), "q1_teaching_quality": rng.integers(1, 6, n)})
    df.loc[df.index[:6], "q1_teaching_quality"] = 99                         # 'no answer' code
    out = A.preprocess_df(df); p = A.profile(out)
    assert A.agg_for(p["metric"], out) == "mean" and out.attrs["cleared_scores"]["q1_teaching_quality"] == 6 and out["q1_teaching_quality"].max() <= 5
    assert next(k for k in p["kpis"] if "Average" in k["label"])["value"] < 5


# ---- spikes -------------------------------------------------------------------------------------------------------
def test_sparse_daily_data_gives_no_absurd_spike_sentences():
    rng = np.random.default_rng(5); days = pd.date_range("2024-01-01", periods=400)
    keep = rng.random(400) < 0.15                                           # most days empty
    df = pd.DataFrame({"date": days[keep], "plan": rng.choice(["a", "b"], keep.sum()), "fee": rng.integers(1000, 3000, keep.sum()).astype(float)})
    assert D.daily_anomalies(df, "date", "fee", ["plan"]) == []


def test_huge_ratios_are_capped_in_words():
    assert D._how_far(1e6, 1.0) == "more than 100 times" and D._how_far(80, 10) == "about 8 times" and D._how_far(140, 100) == "about 40% above"


# ---- direction wording ----------------------------------------------------------------------------------------------
def _two_month(seg_a_prev, seg_a_cur, seg_b_prev, seg_b_cur, a="Paid", b="Overdue"):
    rows = []
    for month, (x, y) in (("2025-01", (seg_a_prev, seg_b_prev)), ("2025-02", (seg_a_cur, seg_b_cur))):
        for day in range(1, 29):
            rows += [(f"{month}-{day:02d}", a, x / 28), (f"{month}-{day:02d}", b, y / 28)]
    df = pd.DataFrame(rows, columns=["date", "status", "amount"]); df["date"] = pd.to_datetime(df["date"]); return df


def test_a_rising_overdue_segment_is_not_called_what_worked():
    df = _two_month(1000, 1000, 100, 400)                                    # overdue grows, paid flat
    s = df.set_index("date")["amount"].resample("MS").sum()
    f = D.bridge(df, "date", "amount", ["status"], "MS", s)
    assert f and "Overdue" in f["detail"] and "went wrong" in f["action"] and f["severity"] == "warn"


def test_a_rising_cost_is_bad_news_and_yes_no_columns_read_naturally():
    df = _two_month(1000, 1300, 500, 500, "A", "B"); df["flag"] = df["status"] == "A"
    s = df.set_index("date")["amount"].resample("MS").sum()
    f = D.bridge(df, "date", "amount", ["flag"], "MS", s, cost_metric=True)
    assert f["severity"] == "warn" and "flag = Yes" in f["detail"] and "True" not in f["detail"]


# ---- segment-level findings -------------------------------------------------------------------------------------------
def test_segment_that_collapses_while_the_business_is_flat_is_found():
    rng = np.random.default_rng(6); rows = []
    for t in pd.date_range("2025-01-01", "2025-12-31"):
        for store in ("Airport", "Downtown", "Mall"):
            v = rng.normal(100, 8) * (0.5 if store == "Airport" and t >= pd.Timestamp("2025-10-01") else 1.0) * (1.5 if store == "Mall" and t >= pd.Timestamp("2025-10-01") else 1.0)
            rows.append((t, store, max(v, 1)))
    df = pd.DataFrame(rows, columns=["date", "store", "net_sales"]); p = A.profile(df); f = A.insights(df, p)
    hit = [x for x in f if x["kind"] == "driver" and "Airport" in x["title"] and "fell" in x["title"]]
    assert hit and hit[0]["severity"] == "warn" and hit[0]["evidence"]["change_pct"] < -30


def test_a_group_with_a_poor_paid_to_due_ratio_is_found():
    rng = np.random.default_rng(7); n = 600
    cls = rng.choice(["Class 8", "Class 9", "Class 10"], n); due = rng.integers(800, 1200, n).astype(float)
    frac = np.where(cls == "Class 9", 0.6, 0.9) * rng.uniform(.9, 1.1, n)
    df = pd.DataFrame({"class": cls, "fee_due": due, "fee_paid": np.minimum(due * frac, due)})
    f = SEG.ratio_gaps(df, A.profile(df))
    assert f and "Class 9" in f[0]["title"] and f[0]["severity"] == "warn" and "fee_paid" in f[0]["title"]


def test_lowest_scoring_group_is_found_and_normal_noise_is_not():
    rng = np.random.default_rng(8); n = 400
    dept = rng.choice(["Arts", "Commerce", "Science"], n); score = np.clip(np.round(rng.normal(np.where(dept == "Commerce", 2.7, 3.7), .9)), 1, 5)
    df = pd.DataFrame({"dept": dept, "q1_quality": score}); p = A.profile(df)
    assert "Commerce" in SEG.rating_gaps(df, p, "q1_quality")[0]["title"]
    flat = pd.DataFrame({"dept": dept, "q1_quality": np.clip(np.round(rng.normal(3.5, .9, n)), 1, 5)})
    assert SEG.rating_gaps(flat, A.profile(flat), "q1_quality") == []


def test_ordinary_weekly_noise_in_a_small_group_is_not_called_a_shift():
    rng = np.random.default_rng(9); rows = []
    for t in pd.date_range("2025-01-01", periods=140):
        for g in ("A", "B", "C"):
            for _ in range(rng.poisson(1.2)): rows.append((t, g, float(rng.integers(50, 150))))
    df = pd.DataFrame(rows, columns=["date", "grp", "amount"])
    assert [x for x in SEG.segment_shifts(df, {**A.profile(df), "_freq": "W"}, "amount", "date", False) if "fell" in x["title"] and abs(x["evidence"]["change_pct"]) > 60] == []


def test_wide_files_do_not_slow_the_part_of_whole_search():
    import time
    rng = np.random.default_rng(10); df = pd.DataFrame(rng.random((300, 250)) * 100, columns=[f"m{i}" for i in range(250)]); df["g"] = rng.choice(list("abc"), 300)
    t = time.time(); SEG.ratio_gaps(df, A.profile(df)); assert time.time() - t < 6


# ---- calendar artefacts, monthly-grain files, one-record trends, dimension order ------------------------------------
def _flat(end, rise_from=None):
    rng = np.random.default_rng(1); df = pd.DataFrame({"date": pd.date_range("2025-07-01", end)})
    df["region"] = rng.choice(["N", "S"], len(df)); df["amount"] = 1000.0
    if rise_from: df.loc[df["date"] >= rise_from, "amount"] = 1200.0
    return df


def test_a_longer_month_is_not_reported_as_growth():
    df = _flat("2026-02-28"); p = A.profile(df)                                    # Jan 31 days -> Feb 28 days, flat per day
    assert not [i for i in A.insights(df, p) if i["kind"] == "change"]
    k = next(k for k in p["kpis"] if "delta" in k); assert k["delta"] == pytest.approx(0, abs=1e-9) and "per day" in k["vs"]


def test_real_growth_is_reported_per_day_when_months_differ():
    df = _flat("2026-02-28", rise_from="2026-02-01"); p = A.profile(df)
    f = [i for i in A.insights(df, p) if i["kind"] == "change"]
    assert f and "20%" in f[0]["title"] and "per day" in f[0]["title"]


def test_one_row_per_month_files_keep_every_month():
    m = pd.DataFrame({"date": [pd.Timestamp(2025, k, 15) for k in range(1, 10)], "amount": np.arange(9) * 100 + 1000.0})
    assert len(A.period_series(m, "date", "amount")[0]) == 9


def test_one_huge_record_does_not_define_the_headline_trend():
    rng = np.random.default_rng(2); rows = []
    for t in pd.date_range("2024-01-01", "2025-12-31"):
        for _ in range(rng.poisson(1.5)): rows.append((t, rng.choice(["Online", "Event"]), float(np.round(rng.lognormal(3.4, .6), 2)) * (1 + 0.25 * (t.year == 2025))))
    df = pd.DataFrame(rows, columns=["date", "channel", "amount"])
    big = df.index[(df["date"] >= "2025-12-10")][0]; df.loc[big, "amount"] = 60_000.0
    trend = next(i for i in A.insights(df, A.profile(df)) if i["kind"] == "trend")
    assert "One record" in trend["detail"] and "left out" in trend["detail"] and float(trend["title"].split("%")[0].split()[-1]) < 60


def test_business_dimensions_come_before_status_flags():
    rng = np.random.default_rng(3); n = 300
    df = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=n), "promo": rng.choice(["Yes", "No"], n), "status": rng.choice(["Paid", "Open"], n),
                       "product": rng.choice([f"P{i}" for i in range(9)], n), "amount": rng.integers(10, 99, n).astype(float)})
    cats = A.profile(df)["cat_cols"]; assert cats[0] == "product" and cats.index("status") > cats.index("product")


def test_weekly_rows_are_compared_per_reporting_date_and_direction_agrees_with_the_kpi():
    rng = np.random.default_rng(5); rows = []
    for t in pd.date_range("2024-10-07", "2025-06-30", freq="W-MON"):         # one row per Monday; June has 5 Mondays (the latest full month), May has 4
        for wh in ("A", "B"): rows.append((t, wh, 100.0 + (0 if t.month != 6 else -3.0) + rng.normal(0, .01)))
    df = pd.DataFrame(rows, columns=["week", "warehouse", "cost"]); p = A.profile(df)
    k = next(k for k in p["kpis"] if "delta" in k); f = [i for i in A.insights(df, p) if i["kind"] == "change"]
    assert "reporting date" in k["vs"] and k["delta"] < 0                      # per week of data, June is slightly LOWER
    assert not f or "fell" in f[0]["title"]                                    # and the finding must agree with the KPI, never say 'rose'


def test_unusual_values_finding_lists_the_largest_records():
    from app import samples
    df = A.preprocess_df(samples.raw_sample("donations")); p = A.profile(df)
    f = next(x for x in A.insights(df, p) if x["title"].endswith("unusual amount_usd values"))
    ev = f["evidence"]
    assert 1 <= len(ev["records"]) <= 10 and ev["column"] in ev["record_columns"] and "records" not in ev["record_columns"]
    top = ev["records"][0][ev["record_columns"].index(ev["column"])]
    assert top == df["amount_usd"].max()                                   # the biggest gift leads the table
