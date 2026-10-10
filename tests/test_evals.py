"""The benchmark must be right before it can judge anything: every reference SQL, run through Lumen's own guard and sandbox,
reproduces the independently computed pandas truth, and the scorer rejects wrong answers."""
import pytest
from app import sqlguard
from evals import cases as C, scoring

DF = C.load(); CASES = C.cases(DF)


@pytest.mark.parametrize("case", CASES, ids=[c["q"][:40] for c in CASES])
def test_reference_sql_matches_pandas_truth(case):
    res = sqlguard.run_sql(DF, case["ref_sql"])
    rows = res.astype(object).where(res.notna(), None).values.tolist()
    assert scoring.score(case, list(res.columns), rows), (case["q"], case["truth"], rows[:3])


def test_scorer_rejects_wrong_answers():
    scalar = next(c for c in CASES if c["kind"] == "scalar"); label = next(c for c in CASES if c["kind"] == "label"); table = next(c for c in CASES if c["kind"] == "table")
    assert not scoring.score(scalar, ["x"], [[scalar["truth"] * 1.05]]) and not scoring.score(scalar, ["x"], [])
    assert not scoring.score(label, ["x"], [["definitely not it"]])
    bad = dict(table["truth"]); k = next(iter(bad)); bad[k] *= 1.1
    assert not scoring.score(table, ["a", "b"], [[a, b] for a, b in bad.items()])
