"""End-to-end through the HTTP API with a scripted fake Gemini (no network, no key)."""
import io, json
import pytest
from fastapi.testclient import TestClient
from app import main, llm, limits, samples
from tests.test_llm import FakeClient, plan, expl, api_err   # reuse the scripted client helpers

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(main, "SESSIONS", limits.SessionStore())
    monkeypatch.setattr(main, "ASK", limits.RateLimiter(per_window=3, per_day=50, global_per_day=100))
    monkeypatch.setattr(main, "NARRATE", limits.RateLimiter(per_window=50, per_day=50, global_per_day=100))
    monkeypatch.setattr(llm, "_cool", {}); monkeypatch.setattr(llm, "_NARR_CACHE", {}); monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    yield
    llm._client = None


def open_sample(name="shop"):
    r = client.post("/api/sample", json={"name": name}); assert r.status_code == 200, r.text
    return r.json()


def test_health_and_samples():
    assert client.get("/api/health").json() == {"ok": True, "ai": False}
    ids = [s["id"] for s in client.get("/api/samples").json()]
    assert ids == ["shop", "donations", "inventory"]


@pytest.mark.parametrize("name", ["shop", "donations", "inventory"])
def test_every_sample_loads_with_insights_and_charts(name):
    d = open_sample(name)
    assert d["profile"]["metric"] and d["charts"] and d["insights"] and d["narrative"]["summary"]
    assert d["sample"] == name and d["ai"] is False and d["suggested_questions"]
    json.dumps(d, allow_nan=False)                                              # strictly JSON-safe


@pytest.mark.parametrize("name", ["shop", "donations", "inventory"])
def test_all_stored_questions_run_in_demo_mode(name):
    d = open_sample(name)
    for q in samples.SAMPLES[name]["questions"]:
        r = client.post("/api/ask", json={"session_id": d["session_id"], "question": q["q"]})
        assert r.status_code == 200, (q["q"], r.text)
        a = r.json(); assert a["mode"] == "demo" and a["rows"] and a["answer"] and a["status"] == "demo"


def test_free_text_without_key_is_503_with_helpful_message():
    d = open_sample()
    r = client.post("/api/ask", json={"session_id": d["session_id"], "question": "something else entirely"})
    assert r.status_code == 503 and "suggested questions" in r.json()["detail"]


def test_ai_answer_with_checks(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    d = open_sample()
    sql = "select region, sum(amount) as revenue from data group by region order by revenue desc"
    chk = "select distinct region, sum(amount) over (partition by region) as revenue from data order by revenue desc"
    llm._client = FakeClient([plan(sql, chk, "bar", "region", "revenue"), expl("Campus brings in the most revenue.")])
    r = client.post("/api/ask", json={"session_id": d["session_id"], "question": "Which region is best?"})
    assert r.status_code == 200
    a = r.json(); assert a["mode"] == "ai" and a["status"] == "checked" and a["verified"] and len(a["checks"]) == 4


def test_ai_down_falls_back_to_stored_for_sample_questions(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    d = open_sample(); llm._client = FakeClient([api_err(429, "RESOURCE_EXHAUSTED")] * 10)
    q = samples.SAMPLES["shop"]["questions"][0]["q"]
    a = client.post("/api/ask", json={"session_id": d["session_id"], "question": q}).json()
    assert a["mode"] == "demo" and "unavailable" in a["note"]
    r = client.post("/api/ask", json={"session_id": d["session_id"], "question": "a custom question"})
    assert r.status_code == 429 and "quota" in r.json()["detail"].lower()


def test_rate_limit_applies_then_stored_questions_still_work(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    d = open_sample(); sid = d["session_id"]
    llm._client = FakeClient([api_err(503, "UNAVAILABLE")] * 60)
    for _ in range(3): client.post("/api/ask", json={"session_id": sid, "question": "custom"})
    r = client.post("/api/ask", json={"session_id": sid, "question": "custom again"})
    assert r.status_code == 429 and "minute" in r.json()["detail"]
    q = samples.SAMPLES["shop"]["questions"][0]["q"]
    assert client.post("/api/ask", json={"session_id": sid, "question": q}).json()["mode"] == "demo"


def test_upload_flow_and_forecast():
    csv = "Order Date,Product,Amount\n" + "\n".join(f"2025-{m:02d}-{d:02d},{p},{(m * 10 + d) % 90 + 10}" for m in range(1, 10) for d in (3, 11, 19, 27) for p in ("A", "B"))
    r = client.post("/api/upload", files={"file": ("sales.csv", csv.encode(), "text/csv")}); assert r.status_code == 200, r.text
    d = r.json(); assert d["profile"]["metric"] == "amount" and d["profile"]["date"] == "order_date"
    f = client.post("/api/forecast", json={"session_id": d["session_id"], "date_col": "order_date", "value_col": "amount", "periods": 3})
    assert f.status_code == 200 and f.json()["forecast"]["y"] and f.json()["confidence"] in ("high", "medium", "low")


def test_upload_limits_and_errors(monkeypatch):
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 1000)
    r = client.post("/api/upload", files={"file": ("big.csv", b"a,b\n" + b"1,2\n" * 1000, "text/csv")})
    assert r.status_code == 413
    monkeypatch.setattr(limits, "MAX_UPLOAD_BYTES", 5 * 1024 * 1024)
    assert client.post("/api/upload", files={"file": ("e.csv", b"", "text/csv")}).status_code == 400
    r = client.post("/api/upload", files={"file": ("x.csv", bytes(range(256)) * 20, "text/csv")})
    assert r.status_code == 400 and "doesn't look like" in r.json()["detail"]


def test_upload_over_the_cell_budget_is_rejected(monkeypatch):
    monkeypatch.setattr(limits, "MAX_CELLS", 50)
    r = client.post("/api/upload", files={"file": ("w.csv", b"a,b,c\n" + b"1,2,3\n" * 40, "text/csv")})
    assert r.status_code == 400 and "cells" in r.json()["detail"]


def test_forecast_is_rate_limited(monkeypatch):
    monkeypatch.setattr(main, "FORECAST", limits.RateLimiter(per_window=1, per_day=10, global_per_day=10))
    sid = open_sample()["session_id"]
    body = {"session_id": sid, "date_col": "order_date", "value_col": "amount", "periods": 3}
    assert client.post("/api/forecast", json=body).status_code == 200
    r = client.post("/api/forecast", json=body)
    assert r.status_code == 429 and "minute" in r.json()["detail"]


def test_unknown_session_and_internal_errors_do_not_leak():
    assert client.post("/api/forecast", json={"session_id": "nope", "date_col": "a", "value_col": "b"}).status_code == 404
    assert client.post("/api/ask", json={"session_id": "nope", "question": "q"}).status_code == 404
    d = open_sample()
    r = client.post("/api/forecast", json={"session_id": d["session_id"], "date_col": "order_date", "value_col": "product"})
    assert r.status_code == 400 and "numeric" in r.json()["detail"]


def test_security_headers_and_static_assets():
    r = client.get("/")
    assert r.status_code == 200 and "default-src 'self'" in r.headers["content-security-policy"] and r.headers["x-content-type-options"] == "nosniff"
    assert "cdn.plot.ly" not in r.text and "fonts.googleapis.com" not in r.text
    assert client.get("/assets/vendor/plotly-basic-2.35.2.min.js").status_code == 200
    assert client.get("/assets/../app/main.py").status_code in (404, 400)


def test_user_can_correct_the_main_measure_and_date():
    d = open_sample("inventory"); sid = d["session_id"]; assert d["profile"]["metric"] == "cost"
    r = client.post("/api/reanalyze", json={"session_id": sid, "metric": "units_shipped"})
    assert r.status_code == 200 and r.json()["profile"]["metric"] == "units_shipped" and r.json()["session_id"] == sid
    assert any("units_shipped" in i["title"] or "units_shipped" in i["detail"] for i in r.json()["insights"]) or r.json()["profile"]["kpis"][1]["label"].endswith("units_shipped")
    assert client.post("/api/reanalyze", json={"session_id": sid, "metric": "product"}).status_code == 400
    assert client.post("/api/reanalyze", json={"session_id": sid, "date": "cost"}).status_code == 400
    assert client.post("/api/reanalyze", json={"session_id": "nope", "metric": "cost"}).status_code == 404
    recs = client.post("/api/reanalyze", json={"session_id": sid, "metric": "records"}).json()
    assert recs["profile"]["metric"] == "records" and "records" not in recs["processed_preview"]["columns"]
    assert client.post("/api/forecast", json={"session_id": sid, "date_col": "week", "value_col": "records", "periods": 3}).status_code == 200


def test_cleaned_preview_never_shows_the_synthetic_records_column():
    d = open_sample("shop"); assert "records" not in d["processed_preview"]["columns"] and "records" not in d["raw_preview"]["columns"]
