import uuid, pathlib, logging, re
from dotenv import load_dotenv
load_dotenv()  # reads GEMINI_API_KEY from .env
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import analytics as A, llm

app = FastAPI(title="Lumen")
SESSIONS: dict = {}   # in-memory only: data is never written to disk
MAX_BYTES = 25 * 1024 * 1024


def start_session(df):
    if len(SESSIONS) > 50: SESSIONS.pop(next(iter(SESSIONS)))
    sid = uuid.uuid4().hex; SESSIONS[sid] = df
    p = A.profile(df); facts = A.insights(df, p)
    meta = {"rows": p["rows"], "metric": p["metric"], "date_column": p["date"], "columns": [c["name"] for c in p["columns"]]}
    return A.clean({"session_id": sid, "profile": p, "charts": A.starter_charts(df, p), "insights": facts,
                    "narrative": llm.narrate(facts, meta), "suggested_questions": A.suggested_questions(p), "ai": llm.available()})


def get_df(sid):
    if sid not in SESSIONS: raise HTTPException(404, "Session expired. Upload your file again.")
    return SESSIONS[sid]


@app.get("/api/health")
def health(): return {"ok": True, "ai": llm.available()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_BYTES: raise HTTPException(413, "File is larger than 25 MB.")
    try: return start_session(A.load_df(raw, file.filename or ""))
    except ValueError as e: raise HTTPException(400, str(e))


@app.post("/api/demo")
def demo(): return start_session(A.demo_df())


class Ask(BaseModel):
    session_id: str
    question: str


@app.post("/api/ask")
def ask(body: Ask):
    df = get_df(body.session_id)
    if not llm.available(): raise HTTPException(503, "Plain-English questions need a Gemini API key (set GEMINI_API_KEY).")
    try: return llm.answer(df, body.question[:500])
    except ValueError as e: raise HTTPException(400, str(e))
    except Exception as e:
        logging.exception("Gemini call failed")  # full traceback appears in the server terminal
        t = str(e)
        if re.search(r"API_KEY|API key|401|403|PERMISSION_DENIED|UNAUTHENTICATED", t, re.I): msg = "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file and restart the server."
        elif re.search(r"429|RESOURCE_EXHAUSTED|quota", t, re.I): msg = "Gemini's free quota is used up for now. Wait a minute and try again."
        elif re.search(r"503|UNAVAILABLE|high demand", t, re.I): msg = "Gemini is very busy right now (all backup models too). Try again in a few seconds."
        elif re.search(r"404|NOT_FOUND", t, re.I): msg = "No Gemini model was available. Set GEMINI_MODEL in .env to a current model name."
        else: msg = "The AI service failed: " + t[:160]
        raise HTTPException(502, msg)


class Fc(BaseModel):
    session_id: str
    date_col: str
    value_col: str
    periods: int = 6


@app.post("/api/forecast")
def fc(body: Fc):
    try: return A.clean(A.forecast(get_df(body.session_id), body.date_col, body.value_col, max(1, min(body.periods, 24))))
    except ValueError as e: raise HTTPException(400, str(e))


app.mount("/assets", StaticFiles(directory=pathlib.Path(__file__).parent.parent / "static"), name="assets")


@app.get("/")
def index(): return FileResponse(pathlib.Path(__file__).parent.parent / "static" / "index.html")