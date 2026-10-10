"""Score the LIVE Gemini pipeline on the golden questions.   Needs a key:   GEMINI_API_KEY=... python -m evals.run_eval

Reports (1) accuracy: does the answer match the pandas truth, and (2) whether Lumen's own checks help: of the wrong
answers, how many were NOT marked 'checked', and of the right answers, how many were wrongly flagged.
Writes evals/results/latest.md. Free-tier quota: 20 questions use ~60 calls; use --limit N to run fewer."""
import argparse, json, os, pathlib, sys, time
from app import llm
from evals import cases as C, scoring


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--sleep", type=float, default=4.0, help="seconds between questions (free-tier rate limits)")
    a = ap.parse_args()
    if not llm.available(): sys.exit("Set GEMINI_API_KEY to run the live evaluation.")
    df = C.load(); items = C.cases(df)[: a.limit or None]; rows_out, calls = [], 0
    for i, case in enumerate(items, 1):
        t = time.time()
        try:
            r = llm.answer(df, case["q"]); ok = scoring.score(case, r["columns"], r["rows"]); status = r["status"]; err = ""
        except Exception as e:
            ok, status, err = False, "error", f"{type(e).__name__}: {str(e)[:80]}"
        rows_out.append({"q": case["q"], "correct": ok, "status": status, "secs": round(time.time() - t, 1), "error": err})
        print(f"{i:2}/{len(items)} {'OK ' if ok else 'BAD'} {status:9} {case['q']}  {err}", flush=True)
        time.sleep(a.sleep)
    n = len(rows_out); right = [r for r in rows_out if r["correct"]]; wrong = [r for r in rows_out if not r["correct"]]
    answered = [r for r in rows_out if r["status"] != "error"]
    flagged_wrong = [r for r in wrong if r["status"] not in ("checked",)]
    flagged_right = [r for r in right if r["status"] not in ("checked",)]
    md = [f"# Live evaluation ({time.strftime('%Y-%m-%d')}, models tried in order: {', '.join(llm.model_order())})", "",
          f"- Questions: {n}; answered without error: {len(answered)}", f"- **Correct vs pandas truth: {len(right)}/{n} ({len(right) / n:.0%})**",
          f"- Wrong answers that Lumen did NOT mark 'checked' (caught): {len(flagged_wrong)}/{len(wrong)}" if wrong else "- No wrong answers",
          f"- Correct answers that were not marked 'checked' (false alarms): {len(flagged_right)}/{len(right)}" if right else "",
          f"- Median seconds per question: {sorted(r['secs'] for r in rows_out)[n // 2]}", "", "| # | Question | Correct | Lumen status | s |", "|---|---|---|---|---|"]
    md += [f"| {i} | {r['q']} | {'yes' if r['correct'] else 'NO'} | {r['status']}{(' ' + r['error']) if r['error'] else ''} | {r['secs']} |" for i, r in enumerate(rows_out, 1)]
    out = pathlib.Path(__file__).parent / "results"; out.mkdir(exist_ok=True); (out / "latest.md").write_text("\n".join(md) + "\n"); (out / "latest.json").write_text(json.dumps(rows_out, indent=1))
    print("\n".join(md[:8]))


if __name__ == "__main__": main()
