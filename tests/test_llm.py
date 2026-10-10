"""llm.py against a scripted fake Gemini: fail-over, error handling, and the answer checks. No network, no key."""
import json
import pandas as pd
import pytest
from google.genai import errors
from app import llm, analytics as A

DF = pd.DataFrame({"product": ["Hoodie", "Mug", "Mug", "Tote"], "amount": [900.0, 300.0, 300.0, 250.0],
                   "order_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-03", "2025-02-04"])})


class Resp:
    def __init__(self, text): self.text = text


class FakeModels:
    def __init__(self, script): self.script, self.calls = list(script), []
    def generate_content(self, model, contents, config):
        self.calls.append((model, config)); item = self.script.pop(0)
        if isinstance(item, Exception): raise item
        return Resp(item)


class FakeClient:
    def __init__(self, script): self.models = FakeModels(script)


def api_err(code, status, msg="x"):
    cls = errors.ServerError if code >= 500 else errors.ClientError
    return cls(code, {"error": {"code": code, "message": msg, "status": status}})


def plan(sql, check="", ctype="table", x="", y=""):
    return json.dumps({"sql": sql, "check_sql": check, "chart_type": ctype, "chart_x": x, "chart_y": y})


def expl(text, caveats=""): return json.dumps({"answer": text, "caveats": caveats})


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(llm, "_cool", {}); monkeypatch.setattr(llm, "_NARR_CACHE", {}); monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    yield
    llm._client = None


def use(script): llm._client = FakeClient(script); return llm._client


SQL = "select product, sum(amount) as total from data group by product order by total desc"
CHECK = "select distinct product, sum(amount) over (partition by product) as total from data order by total desc"


def test_happy_path_all_checks_pass():
    use([plan(SQL, CHECK, "bar", "product", "total"), expl("Mug brings in 600 and Hoodie 900; Hoodie is first.")])
    r = llm.answer(DF, "Which product brings in the most revenue?")
    assert r["status"] == "checked" and r["verified"] and r["columns"] == ["product", "total"]
    assert r["rows"][0] == ["Hoodie", 900.0] and r["chart"]["type"] == "bar"
    assert [c["ok"] for c in r["checks"]] == [True, True, True, True]


def test_disagreeing_second_query_is_flagged():
    wrong = "select product, sum(amount) + 1 as total from data group by product"
    use([plan(SQL, wrong), expl("Hoodie leads with 900.")])
    r = llm.answer(DF, "q")
    assert r["status"] == "disagree" and not r["verified"]
    assert any(c["id"] == "second_query" and c["ok"] is False for c in r["checks"])


@pytest.mark.parametrize("copy", [SQL, SQL.upper() + ";", "  " + SQL.replace(" ", "\n  ") + " -- same"])
def test_repeated_second_query_is_not_a_cross_check(copy):
    use([plan(SQL, copy), expl("Hoodie leads with 900.")])
    r = llm.answer(DF, "q")
    assert r["status"] == "partly" and not r["verified"]
    assert any(c["id"] == "second_query" and c["ok"] is None and "identical" in c["label"] for c in r["checks"])


def test_made_up_number_replaced_by_plain_summary():
    use([plan(SQL, CHECK), expl("Hoodie sold 12,345 units, up 80%.")])
    r = llm.answer(DF, "q")
    assert "12,345" not in r["answer"] and "Hoodie" in r["answer"]
    assert any(c["id"] == "numbers" and c["ok"] is False for c in r["checks"]) and r["status"] != "checked"


def test_wrong_product_name_replaced_by_plain_summary():
    top2 = "select product, sum(amount) as total from data where product <> 'Tote' group by product order by total desc"
    use([plan(top2, top2), expl("Mug brings in 600 and Hoodie 900; Tote is the best seller.")])    # Tote exists in the data but is not in the result
    r = llm.answer(DF, "Which products sold the most?")
    assert "best seller" not in r["answer"] and any(c["id"] == "numbers" and c["ok"] is False and "Tote" in c["label"] for c in r["checks"])


def test_names_helper_is_quiet_for_honest_text():
    cols, rows = ["product", "total"], [["Hoodie", 900.0], ["Mug", 600.0]]
    assert llm.wrong_names("Hoodie leads, then Mug.", DF, "q", cols, rows) == []
    assert llm.wrong_names("Hoodie leads, then Tote.", DF, "q", cols, rows) == ["Tote"]
    assert llm.wrong_names("Tote was asked about", DF, "How is Tote doing?", cols, rows) == []     # named in the question: allowed
    assert llm.wrong_names("Total sales rose.", DF, "q", [["total"]], [[5.0]]) == []                  # no text cells in the result: nothing to compare


def test_derived_numbers_are_allowed():
    use([plan(SQL, CHECK), expl("Hoodie (900) is 150% of Mug (600), and the total is 1,750.")])   # 900/600 -> 150%, sum -> 1750
    assert llm.answer(DF, "q")["status"] == "checked"


def test_quota_fails_over_to_next_model_then_succeeds():
    c = use([api_err(429, "RESOURCE_EXHAUSTED"), plan(SQL, CHECK), expl("Hoodie leads with 900.")])
    r = llm.answer(DF, "q")
    assert r["status"] == "checked"
    assert c.models.calls[0][0] != c.models.calls[1][0]                      # second call used a different model


def test_retired_model_404_skipped_and_cooled_down():
    c = use([api_err(404, "NOT_FOUND", "models/x is not found for project 123404"), plan(SQL, CHECK), expl("Hoodie 900.")])
    llm.answer(DF, "q")
    assert llm.model_order()[0] in llm._cool


def test_blocked_empty_response_tries_next_model():
    use([None, plan(SQL, CHECK), expl("Hoodie 900.")])                       # .text None = blocked
    assert llm.answer(DF, "q")["status"] == "checked"


def test_bad_json_retries_in_plain_mode():
    c = use(["{not json", plan(SQL, CHECK), expl("Hoodie 900.")])
    assert llm.answer(DF, "q")["status"] == "checked"
    assert c.models.calls[0][1].response_json_schema is not None and c.models.calls[1][1].response_json_schema is None


def test_schema_rejected_400_retries_without_schema():
    c = use([api_err(400, "INVALID_ARGUMENT", "response_json_schema unsupported"), plan(SQL, CHECK), expl("Hoodie 900.")])
    assert llm.answer(DF, "q")["status"] == "checked"


def test_bad_key_is_reported_not_retried():
    c = use([api_err(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.")])
    with pytest.raises(llm.AIUnavailable) as e: llm.answer(DF, "q")
    assert e.value.kind == "key" and len(c.models.calls) == 1


def test_server_overload_everywhere_is_busy():
    use([api_err(503, "UNAVAILABLE")] * 20)
    with pytest.raises(llm.AIUnavailable) as e: llm.answer(DF, "q")
    assert e.value.kind == "busy"


def test_hostile_sql_from_model_is_refused_after_one_repair():
    bad = "select * from read_csv('/etc/passwd')"
    use([plan(bad), plan(bad)])
    with pytest.raises(ValueError, match="couldn't answer that safely"): llm.answer(DF, "q")


def test_repair_loop_recovers():
    use([plan("select nope from data"), plan(SQL, CHECK), expl("Hoodie 900.")])
    assert llm.answer(DF, "q")["rows"][0][0] == "Hoodie"


def test_no_key_means_off():
    with pytest.raises(llm.AIUnavailable) as e: llm.answer(DF, "q", ai=False)
    assert e.value.kind == "off"


def test_numbers_helpers():
    ok = llm.allowed_numbers("how many in 2025?", ["name", "amt"], [["a", 1234.56], ["b", 100.0]])
    assert llm.ungrounded("b is 100, which is 7.5% of the total", ok) == []                 # derivable: 100 / (1234.56 + 100)
    assert llm.ungrounded("b is 100 and revenue grew 42%", ok) != []                         # 42 is not in or derivable from the result
    assert llm.ungrounded("a is 1,234.6, b is 100, in 2025", ok) == []
    assert llm.ungrounded("revenue was $1.2K", llm.allowed_numbers("", ["x"], [[1234.0]])) == []   # 1.2K ~ 1234 at displayed precision
    assert llm.ungrounded("revenue was $1.3K", llm.allowed_numbers("", ["x"], [[1234.0]])) != []


def test_narrate_uses_ai_when_grounded_else_template():
    facts = [{"title": "amount is up 43%", "detail": "Up 43% over the period.", "action": "Repeat what worked."}]
    use([json.dumps({"summary": "Sales are up 43%. Keep going.", "recommendations": ["Repeat what worked.", "Watch stock."]})])
    assert llm.narrate(facts, {})["source"] == "ai"
    llm._NARR_CACHE.clear()
    use([json.dumps({"summary": "Sales are up 99%.", "recommendations": ["x"]})])           # 99 is not in the findings
    r = llm.narrate(facts, {}); assert r["source"] == "template" and "43%" in r["summary"]
    use([api_err(429, "RESOURCE_EXHAUSTED")] * 10); llm._NARR_CACHE.clear()
    assert llm.narrate(facts, {})["source"] == "template"


def test_stored_answer_runs_live_sql():
    r = llm.stored_answer(DF, {"sql": SQL, "chart": {"type": "bar", "x": "product", "y": "total"}})
    assert r["mode"] == "demo" and r["rows"][0] == ["Hoodie", 900.0] and not r["verified"]


def test_off_by_one_on_a_large_total_is_caught():
    big = pd.DataFrame({"product": ["A", "B"], "amount": [2_000_000.0, 870_000.0]})
    use([plan("select sum(amount) as total from data", "select sum(amount) + 1 as total from data"), expl("The total is 2,870,000.")])
    assert llm.answer(big, "q")["status"] == "disagree"
    use([plan("select sum(amount) as total from data", "select sum(amount * 1.0) as total from data"), expl("The total is 2,870,000.")])
    assert llm.answer(big, "q")["status"] == "checked"             # float summation noise is not a disagreement


def test_describe_time_series_and_ranking():
    cols = ["month", "revenue"]
    rows = [["2025-01-01 00:00:00", 100.0], ["2025-02-01 00:00:00", 250.0], ["2025-03-01 00:00:00", 80.0], ["2025-04-01 00:00:00", 120.0]]
    d = llm.describe(cols, rows)
    assert "from Jan 2025 to Apr 2025" in d and "highest is 250 (Feb 2025)" in d and "lowest is 80 (Mar 2025)" in d and "00:00:00" not in d
    assert llm.describe(["p", "t"], [["A", 5.0], ["B", 9.0], ["C", 1.0], ["D", 2.0]]).startswith("4 rows. Top by t: A (5)")
    assert "28 Jan 2025" in llm.describe(cols, [["2025-01-28", 1.0], ["2025-01-29", 2.0], ["2025-01-30", 3.0], ["2025-01-31", 4.0]])


def test_describe_handles_real_timestamps_from_duckdb():
    res = pd.DataFrame({"month": pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]), "revenue": [100.0, 250.0, 80.0, 120.0]})
    rows = res.astype(object).where(res.notna(), None).values.tolist()
    assert "from Jan 2025 to Apr 2025" in llm.describe(["month", "revenue"], rows)


def test_call_budget_stops_a_slow_failing_call(monkeypatch):
    monkeypatch.setattr(llm, "CALL_BUDGET_S", -1)
    c = use([plan(SQL, CHECK)])
    with pytest.raises(llm.AIUnavailable): llm.answer(DF, "q")
    assert not c.models.calls                                                  # budget already spent: no request is even sent


def test_response_whose_text_property_raises_is_treated_as_blocked():
    class Weird:
        @property
        def text(self): raise RuntimeError("no text part")
    class M:
        calls = 0
        def generate_content(self, model, contents, config):
            M.calls += 1
            return Weird() if M.calls == 1 else Resp(plan(SQL, CHECK) if M.calls == 2 else expl("Hoodie 900."))
    class C: models = M()
    llm._client = C()
    assert llm.answer(DF, "q")["status"] == "checked"
