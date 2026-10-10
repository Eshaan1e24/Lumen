"""Recommendations must quote real names and numbers from the data, never repeat, and never crash."""
import pathlib, warnings
import pandas as pd, numpy as np
import pytest
from app import analytics as A, samples, recommend as R

FIX = pathlib.Path(__file__).parent / "fixtures"


def recs(df):
    p = A.profile(df); return R.recommend(A.insights(df, p), p, df), p


def test_samples_name_the_real_cause():
    d, _ = recs(A.preprocess_df(samples.raw_sample("donations")))
    assert "Brightfield Foundation" in d[0]["title"] and "25,000" in (d[0]["detail"] + d[0]["because"])
    i, _ = recs(A.preprocess_df(samples.raw_sample("inventory")))
    assert any("Winter coats" in r["title"] and "weeks of stock" in r["title"] for r in i)


@pytest.mark.parametrize("name", ["shop", "donations", "inventory"])
def test_no_repeats_and_each_has_evidence(name):
    r, _ = recs(A.preprocess_df(samples.raw_sample(name)))
    assert 2 <= len(r) <= 6 and len({x["title"] for x in r}) == len(r)
    assert all(x["because"] and x["detail"] for x in r) and [x["priority"] for x in r] == list(range(1, len(r) + 1))


def test_every_fixture_runs_without_a_rule_error_or_crash():
    for f in sorted(FIX.iterdir()):
        if f.name.startswith("edge_empty") or f.name in ("edge_header_only.csv", "edge_binary_garbage.csv", "edge_html_page.csv", "edge_newlines_only.csv"): continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = A.preprocess_df(A.read_raw_df(f.read_bytes(), f.name)); p = A.profile(df)
            out = R.recommend(A.insights(df, p), p, df)
        assert isinstance(out, list) and not R.ERRORS, (f.name, R.ERRORS)


def test_unsettled_money_is_chased_but_cancelled_is_investigated():
    rng = np.random.default_rng(1); n = 300
    df = pd.DataFrame({"issue_date": pd.date_range("2025-01-01", periods=n), "client": rng.choice([f"C{i}" for i in range(12)], n),
                       "status": rng.choice(["Paid", "Overdue", "Cancelled"], n, p=[.6, .25, .15]), "amount": rng.integers(100, 900, n).astype(float)})
    d, _ = recs(df)
    chase = next(r for r in d if r["title"].startswith("Chase"))
    assert "Overdue" in chase["title"] and "Cancelled" not in chase["title"]


def test_an_averaged_measure_gets_no_total_based_advice():
    rng = np.random.default_rng(2); n = 200
    df = pd.DataFrame({"dept": rng.choice(["A", "B", "C"], n), "q1_quality": rng.integers(1, 6, n)})
    p = A.profile(df); assert A.agg_for(p["metric"], df) == "mean"
    assert R.recommend(A.insights(df, p), p, df) == []
