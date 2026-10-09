"""Gemini calls + the safety layer (SQL guard, sandboxed DuckDB, answer verification)."""
import os, re, json, time, threading
import duckdb, pandas as pd
from .analytics import clean

# gemini-2.5-flash is being retired (Oct 2026); 3.5 Flash is the GA replacement. "-latest" is a fallback alias.
MODELS = [m for m in dict.fromkeys([os.getenv("GEMINI_MODEL"), "gemini-3.5-flash", "gemini-flash-latest", "gemini-3.1-flash-lite"]) if m]
_client = None


def available() -> bool: return bool(os.getenv("GEMINI_API_KEY"))


def _json(prompt: str) -> dict:
    global _client, _good
    from google import genai
    from google.genai import types
    if _client is None: _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cfg = types.GenerateContentConfig(response_mime_type="application/json")
    last = None
    for model in ([_good] + [m for m in MODELS if m != _good] if _good else MODELS):
        for attempt in range(2):
            try:
                t = _client.models.generate_content(model=model, contents=prompt, config=cfg).text.strip()
                _good = model
                return json.loads(re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip())
            except Exception as e:
                last = e
                if MISSING.search(str(e)): break                      # model gone: try the next one
                if BUSY.search(str(e)):                               # overloaded: retry once, then try the next model
                    if attempt == 0: time.sleep(1.5); continue
                    break
                raise
    raise last


MISSING = re.compile(r"404|NOT_FOUND|no longer available", re.I)
BUSY = re.compile(r"503|UNAVAILABLE|high demand|overloaded|500|INTERNAL|DEADLINE", re.I)
_good = None


# ---- Safety: only one read-only SELECT, run in an isolated DuckDB with no file/network access ----
BLOCK = re.compile(r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|import|call|set|truncate|"
                   r"read_\w+|glob|httpfs|parquet_\w+|duckdb_\w+|pg_\w+)\b", re.I)


def guard(sql: str) -> str:
    s = re.sub(r"--.*?$|/\*.*?\*/", "", sql, flags=re.S | re.M).strip().rstrip(";").strip()
    if ";" in s: raise ValueError("Only one query is allowed.")
    if not re.match(r"(select|with)\b", s, re.I): raise ValueError("Only SELECT queries are allowed.")
    if BLOCK.search(s): raise ValueError("The query uses an operation that isn't allowed.")
    return s


def run_sql(df: pd.DataFrame, sql: str, timeout: int = 10) -> pd.DataFrame:
    s = guard(sql)
    con = duckdb.connect(":memory:")
    con.register("data", df)
    con.execute("SET enable_external_access=false")
    con.execute("SET lock_configuration=true")
    timer = threading.Timer(timeout, con.interrupt); timer.start()
    try: return con.execute(f"SELECT * FROM ({s}) LIMIT 1000").df()
    finally: timer.cancel(); con.close()


def schema_text(df: pd.DataFrame) -> str:
    lines = [f"- {c} ({df[c].dtype}) e.g. {df[c].dropna().astype(str).unique()[:3].tolist()}" for c in df.columns]
    return "Table `data` with columns:\n" + "\n".join(lines)


def answer(df: pd.DataFrame, question: str) -> dict:
    prompt = f"""You write DuckDB SQL for a small-business owner. {schema_text(df)}
Question: {question}
Return JSON: {{"sql": "one SELECT on table data", "chart": {{"type": "bar|line|pie|number|table", "x": "result column", "y": "result column"}}}}
Rules: only SELECT; alias result columns with plain names; for time trends use date_trunc and ORDER BY the date; max 30 rows unless the user asks otherwise."""
    plan, err = _json(prompt), None
    for attempt in range(2):
        try: res = run_sql(df, plan["sql"]); break
        except Exception as e:
            if attempt: raise ValueError(f"Couldn't answer that question safely: {e}")
            plan = _json(prompt + f"\nYour previous SQL `{plan.get('sql')}` failed with: {e}. Fix it.")
    if res.empty: raise ValueError("The query ran but returned no rows. Try rephrasing the question.")
    head = res.head(15).to_csv(index=False)
    # Verification step: a second pass checks the result really answers the question, and writes the
    # explanation using only numbers present in the result.
    v = _json(f"""Question: {question}\nSQL: {plan['sql']}\nResult (first rows):\n{head}
Return JSON: {{"answers_question": true/false, "answer": "1-3 plain sentences using ONLY numbers in the result", "caveats": "assumptions or limits, or empty string"}}""")
    chart = plan.get("chart") or {}
    if chart.get("x") not in res.columns or chart.get("y") not in res.columns:
        chart = {"type": "table"} if chart.get("type") != "number" else chart
    rows = res.astype(object).where(res.notna(), None).values.tolist()
    return clean({"sql": plan["sql"], "chart": chart, "columns": list(res.columns), "rows": rows,
                  "verified": bool(v.get("answers_question")), "answer": v.get("answer", ""), "caveats": v.get("caveats", "")})


def narrate(facts: list, meta: dict) -> dict:
    """Plain-language summary + recommendations. Falls back to template text if no key / error."""
    fallback = {"summary": " ".join(f["detail"] for f in facts[:2]) or "No strong patterns found yet.",
                "recommendations": [f["action"] for f in facts if f.get("action")][:4]}
    if not available() or not facts: return fallback
    try:
        r = _json(f"""You advise a non-technical small-business owner or NGO coordinator. Dataset: {json.dumps(meta)}.
Verified statistical findings: {json.dumps(clean(facts))}
Return JSON: {{"summary": "3 short sentences, plain language, what matters most", "recommendations": ["3-4 concrete next steps, each one sentence"]}}
Use ONLY numbers from the findings. Do not claim causes.""")
        return {"summary": r["summary"], "recommendations": r["recommendations"][:4]}
    except Exception:
        return fallback