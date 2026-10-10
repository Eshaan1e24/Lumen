"""Ingestion regression suite: real-world-style messy files (generated during QA) must load, pick a sensible measure and date,
and produce strictly JSON-safe output. Files that are not data must fail with a clear 400-style message, never a crash."""
import json, pathlib, warnings
import pytest
from app import analytics as A

DIR = pathlib.Path(__file__).parent / "fixtures"

# file -> (metric, date column) the profiler should choose; None means "no such column in this file"
EXPECT = {
    "accents_nfc.csv": ("montant_ttc", "date_de_création"), "accents_nfd.csv": ("montant_ttc", "date_de_création"),
    "ambig_dmy_small.csv": ("sales", "date"), "bom_utf8.csv": ("amount", "date"), "clean.xlsx": ("revenue", "order_date"),
    "cp1252_euro.csv": ("amount", "date"), "csv_named_xlsx.xlsx": ("amount", "date"), "donations.csv": ("amount", "date"),
    "european.csv": ("betrag", "datum"), "european_nothousands.csv": ("betrag", "datum"), "expenses_currency.csv": ("amount", "date"),
    "hindi_headers.csv": ("राशि", "तारीख"), "ids_plus_amount.csv": ("amount", "order_date"), "inr_lakh.csv": ("revenue", "date"),
    "latin1.csv": ("amount", "date"), "multi_sheet_cover_first.xlsx": ("sales", "date"), "multi_sheet_data_first.xlsx": ("sales", "date"),
    "offset_table.xlsx": ("sales", "date"), "title_rows_total.xlsx": ("amount", "date"), "total_row_literal.xlsx": ("amount", "date"),
    "tz_mixed_offsets.csv": ("amount", "created_at"), "tz_utc.csv": ("amount", "created_at"), "utf16_tab.txt.csv": ("amount", "date"),
    "xlsx_named_csv.csv": ("revenue", "order_date"), "yyyymmdd_int.csv": ("amount", "date"), "epoch_seconds.csv": ("amount", "timestamp"),
    "excel_serial.csv": ("amount", "date"), "thousands_text.csv": ("revenue", "date"), "edge_dup_cols.csv": ("amount", "date"),
    "edge_single_row.csv": ("amount", "date"), "edge_no_date.csv": ("qty", None), "edge_no_numeric.csv": (None, None),
    "edge_constant.csv": ("amount", "date"), "edge_ragged.csv": ("amount", "date"), "edge_trailing_blank.csv": ("sales", "date"),
    "edge_wide_300.csv": ("metric_0", "date"), "us_mdy_halfyear.csv": ("sales", "date"), "edge_reserved_cols.csv": ("order", "date"),
}
NOT_DATA = ["edge_empty.csv", "edge_header_only.csv", "edge_binary_garbage.csv", "edge_html_page.csv", "edge_newlines_only.csv"]


def load(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return A.preprocess_df(A.read_raw_df((DIR / name).read_bytes(), name))


@pytest.mark.parametrize("name", sorted(EXPECT))
def test_profiler_picks_sensible_columns_and_output_is_json_safe(name):
    df = load(name); p = A.profile(df)
    assert (p["metric"], p["date"]) == EXPECT[name], (name, p["metric"], p["date"])
    out = A.clean({"p": p, "insights": A.insights(df, p), "charts": A.starter_charts(df, p), "q": A.suggested_questions(p), "prev": A.df_to_preview(df)})
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize("name", NOT_DATA)
def test_non_data_files_fail_politely(name):
    with pytest.raises(ValueError) as e: load(name)
    assert str(e.value) and "Traceback" not in str(e.value)


def test_every_fixture_is_covered():
    assert {f.name for f in DIR.iterdir()} == set(EXPECT) | set(NOT_DATA) | {"edge_inf_values.csv"}


def test_infinite_values_are_not_treated_as_data():
    df = load("edge_inf_values.csv"); assert not df.select_dtypes("number").isin([float("inf"), float("-inf")]).any().any()


def test_total_row_is_not_double_counted():
    plain, with_total = load("title_rows_total.xlsx"), load("total_row_literal.xlsx")
    assert not plain.astype(str).apply(lambda c: c.str.fullmatch(r"(?i)\s*total\s*")).any().any()
    assert not with_total.astype(str).apply(lambda c: c.str.fullmatch(r"(?i)\s*(grand\s+)?total\s*")).any().any()


def test_data_health_fields_and_quality_findings():
    import pandas as pd, numpy as np
    rng = np.random.default_rng(3); n = 400
    df = pd.DataFrame({"invoice_no": [f"INV{i}" for i in range(n)], "date": pd.date_range("2025-01-01", periods=n, freq="D"),
                       "customer_name": rng.choice([f"Client {i}" for i in range(40)], n), "item": rng.choice(["Hoodie", "hoodie", "Mug"], n),
                       "amount": rng.integers(10, 100, n).astype(float)})
    df.loc[5, "amount"] = 50_000.0                                                     # one huge record
    df = pd.concat([df, df.iloc[10:16]], ignore_index=True)                            # six double entries (same invoice numbers)
    p = A.profile(df); ins = {i["title"]: i for i in A.insights(df, p)}
    amount = next(c for c in p["columns"] if c["name"] == "amount")
    assert amount["outliers"] >= 1 and amount["outlier_examples"][0] == 50_000.0 and amount["outlier_share"] > 0.3
    assert next(c for c in p["columns"] if c["name"] == "item")["label_variant_groups"] == 1
    assert p["duplicates"] == 6 and p["entity_col"] == "customer_name"
    assert any("exact duplicates" in t for t in ins) and any("inconsistent labels" in t for t in ins) and any("unusual amount" in t for t in ins)
    assert any(c["type"] == "hist" for c in A.starter_charts(df, p)) and any(c["title"].startswith("Top customer_name") for c in A.starter_charts(df, p))


def test_identical_rows_without_a_record_number_are_not_called_duplicates():
    from app import samples
    df = A.preprocess_df(samples.raw_sample("shop")); p = A.profile(df)
    assert p["duplicates"] > 0 and not any("duplicates" in i["title"] for i in A.insights(df, p))
