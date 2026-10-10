import uuid, pathlib, logging, os
from dotenv import load_dotenv
load_dotenv()  # reads GEMINI_API_KEY from .env
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from . import analytics as A, llm, limits, samples, recommend as R

app = FastAPI(title="Lumen")
app.add_middleware(GZipMiddleware, minimum_size=1024)
SESSIONS = limits.SessionStore(max_weight=limits.MAX_CELLS)   # in memory only: LRU + time-to-live, bounded by count and size; never written to disk by Lumen
ASK = limits.RateLimiter(per_window=int(os.getenv("LUMEN_ASK_PER_10MIN", 8)), per_day=int(os.getenv("LUMEN_ASK_PER_DAY", 60)),
                         global_per_day=int(os.getenv("LUMEN_ASK_GLOBAL_PER_DAY", 400)))
FORECAST = limits.RateLimiter(per_window=int(os.getenv("LUMEN_FORECAST_PER_10MIN", 30)), per_day=300, global_per_day=int(os.getenv("LUMEN_FORECAST_GLOBAL_PER_DAY", 5000)))
NARRATE = limits.RateLimiter(per_window=6, per_day=40, global_per_day=int(os.getenv("LUMEN_NARRATE_GLOBAL_PER_DAY", 200)))
STATIC = pathlib.Path(__file__).parent.parent / "static"

# Strict CSP: the page makes no third-party requests (Plotly and fonts are served from /assets/vendor), and the browser enforces it.
CSP = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
       "font-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


@app.middleware("http")
async def security_and_cache_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = CSP
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    if request.url.path.startswith("/assets/vendor/"): resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return resp


def start_session(df, raw_preview=None, sample=None, ai_narrative=True):
    sid = uuid.uuid4().hex
    processed = A.df_to_preview(df)           # captured before profiling adds the synthetic 'records' column
    sess = {"df": df, "sample": sample, "raw_preview": raw_preview or processed, "processed_preview": processed}
    payload = build_payload(sid, sess, ai_narrative)
    SESSIONS.put(sid, sess, weight=int(df.size))
    return payload


def build_payload(sid, sess, ai_narrative=True, metric=None, date=None):
    """Everything the dashboard shows, for the chosen main measure and date column (Lumen's guess unless the user overrides it)."""
    df, sample = sess["df"], sess["sample"]
    p = A.profile(df, metric=metric, date=date); facts = A.insights(df, p)
    meta = {"rows": p["rows"], "metric": p["metric"], "date_column": p["date"], "columns": [c["name"] for c in p["columns"]]}
    stored = [q["q"] for q in samples.SAMPLES[sample]["questions"]] if sample else []
    notices = []
    if df.attrs.get("truncated_from"): notices.append(f"Only the first {len(df):,} of {df.attrs['truncated_from']:,} rows were analysed.")
    if df.attrs.get("removed_total_rows"):
        n = df.attrs["removed_total_rows"]; notices.append(f"Removed {n} 'total' row{'s' if n != 1 else ''} from your file so totals are not counted twice.")
    if df.attrs.get("cleared_scores"):
        cs = df.attrs["cleared_scores"]; notices.append(f"Ignored {sum(cs.values()):,} out-of-range answers (such as 99) in {', '.join(list(cs)[:3])}: they look like 'no answer' codes, not scores.")
    if df.attrs.get("merged_labels"):
        m0, cols = df.attrs["merged_labels"][0], sorted({m["column"] for m in df.attrs["merged_labels"]})
        notices.append(f"Merged different spellings of the same label in {', '.join(cols[:3])} (for example '{m0['from']}' became '{m0['to']}').")
    try: recs = R.recommend(facts, p, df)
    except Exception:
        logging.exception("Recommendations failed"); recs = []
    narr = llm.narrate(facts, meta, ai=ai_narrative)
    if not recs:                    # nothing rule-based to say: fall back to each finding's own suggested action, without repeats
        seen, recs = set(), []
        for f in facts:
            a = f.get("action")
            if a and a not in seen: seen.add(a); recs.append({"priority": len(recs) + 1, "title": a, "detail": "", "because": f["title"]})
        recs = recs[:4]
    narr = {**narr, "recommendations": recs}
    if narr.get("source") == "template": narr["summary"] = llm.template_summary(facts, recs)
    return A.clean({"session_id": sid, "profile": p, "charts": A.starter_charts(df, p), "insights": facts,
                    "narrative": narr, "suggested_questions": stored + [q for q in A.suggested_questions(p) if q not in stored][:3 if not stored else 0],
                    "ai": llm.available(), "sample": sample, "notices": notices,
                    "raw_preview": sess["raw_preview"], "processed_preview": sess["processed_preview"]})


def get_session(sid):
    try: return SESSIONS.get(sid)
    except KeyError: raise HTTPException(404, "Session expired. Upload your file again.")


@app.get("/api/health")
def health(): return {"ok": True, "ai": llm.available()}


@app.get("/api/samples")
def sample_list(): return samples.listing()


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        raw = await limits.read_capped(file)
        limits.check_xlsx_bomb(raw)
    except limits.UploadTooLarge as e: raise HTTPException(413, str(e))
    except limits.UploadRejected as e: raise HTTPException(400, str(e))
    ai = llm.available() and NARRATE.check(limits.client_key(request))[0]
    return await run_in_threadpool(_process_upload, raw, file.filename or "", ai)


def _process_upload(raw, filename, ai):
    try:
        raw_df = A.read_raw_df(raw, filename)
        if raw_df.shape[1] > limits.MAX_COLS: raise ValueError(f"That file has {raw_df.shape[1]} columns; this server accepts up to {limits.MAX_COLS}.")
        raw_preview = A.df_to_preview(raw_df)
        df = A.preprocess_df(raw_df)
        if df.size > limits.MAX_CELLS:    # the session store keeps one oversized session, so stop it here (a small .xlsx can expand a lot)
            raise ValueError(f"That file has {df.size:,} cells after cleaning; this server accepts up to {limits.MAX_CELLS:,}.")
        return start_session(df, raw_preview, ai_narrative=ai)
    except ValueError as e: raise HTTPException(400, str(e))
    except Exception:
        logging.exception("Upload processing failed")
        raise HTTPException(400, "Could not process this file. Check that it is a CSV or Excel file with a header row.")


class SampleReq(BaseModel):
    name: str = "shop"


@app.post("/api/sample")
async def sample(request: Request, body: SampleReq):
    if body.name not in samples.SAMPLES: raise HTTPException(404, "Unknown sample dataset.")
    ai = llm.available() and NARRATE.check(limits.client_key(request))[0]
    def build():
        raw = samples.raw_sample(body.name)
        return start_session(A.preprocess_df(raw.copy()), A.df_to_preview(raw), sample=body.name, ai_narrative=ai)
    return await run_in_threadpool(build)


@app.post("/api/demo")
async def demo(request: Request): return await sample(request, SampleReq(name="shop"))


class Reanalyze(BaseModel):
    session_id: str
    metric: str | None = None
    date: str | None = None


@app.post("/api/reanalyze")
async def reanalyze(request: Request, body: Reanalyze):
    """The user corrects the main measure and/or date column: recompute the dashboard on the same session."""
    sess = get_session(body.session_id)
    ai = llm.available() and NARRATE.check(limits.client_key(request))[0]
    def run():
        try: return build_payload(body.session_id, sess, ai, metric=body.metric, date=body.date)
        except ValueError as e: raise HTTPException(400, str(e))
    return await run_in_threadpool(run)


class Ask(BaseModel):
    session_id: str
    question: str


def _demo_answer(sess, stored, reason):
    out = llm.stored_answer(sess["df"], stored)
    out["note"] = reason
    return out


@app.post("/api/ask")
def ask(request: Request, body: Ask):
    sess = get_session(body.session_id)
    q = body.question.strip()[:500]
    if not q: raise HTTPException(400, "Type a question first.")
    stored = samples.find_stored(sess["sample"], q)
    if not llm.available():
        if stored: return _demo_answer(sess, stored, "Demo mode: no AI key is configured, so this stored sample question was replayed on the data.")
        raise HTTPException(503, "Free-text questions need a Gemini API key on the server." + (" In demo mode, use the suggested questions." if sess["sample"] else ""))
    ok, retry, why = ASK.check(limits.client_key(request))
    if not ok:
        if stored: return _demo_answer(sess, stored, "Question limit reached, so this stored sample question was replayed instead of calling the AI.")
        mins = max(1, retry // 60 + 1)
        raise HTTPException(429, f"That's a lot of questions for a free demo. Please try again in about {mins} minute{'s' if mins != 1 else ''}"
                                 + (", or use a suggested question." if sess["sample"] else "."))
    try: return llm.answer(sess["df"], q)
    except llm.AIUnavailable as e:
        if stored: return _demo_answer(sess, stored, f"The AI is unavailable ({e.kind}), so this stored sample question was replayed instead.")
        raise HTTPException(429 if e.kind == "quota" else 503, str(e))
    except ValueError as e: raise HTTPException(400, str(e))
    except Exception:
        logging.exception("Answering failed")
        raise HTTPException(500, "Something went wrong while answering that. Try rephrasing the question.")


class Fc(BaseModel):
    session_id: str
    date_col: str
    value_col: str
    periods: int = 6


@app.post("/api/forecast")
def fc(request: Request, body: Fc):
    df = get_session(body.session_id)["df"]   # a 404 here must not be re-wrapped as a 400
    ok, retry, _ = FORECAST.check(limits.client_key(request))   # backtesting is CPU-heavy on a free instance
    if not ok: raise HTTPException(429, f"Too many forecasts in a short time. Please try again in about {max(1, retry // 60 + 1)} minutes.")
    try: return A.clean(A.forecast(df, body.date_col, body.value_col, max(1, min(body.periods, 24))))
    except ValueError as e: raise HTTPException(400, str(e))
    except Exception:
        logging.exception("Forecast failed")
        raise HTTPException(400, "Could not generate a forecast for those columns.")


app.mount("/assets", StaticFiles(directory=STATIC), name="assets")


@app.get("/")
def index(): return FileResponse(STATIC / "index.html")
