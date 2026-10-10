"""Score a Lumen answer (result columns + rows) against a golden case."""
import math


def _close(a, b, rel=5e-3):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-9)


def _matches(v, want: str) -> bool:
    """Equal as text, or a timestamp whose date text starts with the wanted label ('2025-12' matches 2025-12-01 00:00:00)."""
    t = str(v)
    return t == want or (len(t) >= 7 and t[:4].isdigit() and t[4] == "-" and t.startswith(want))


def score(case: dict, columns: list, rows: list) -> bool:
    if not rows: return False
    kind, truth = case["kind"], case["truth"]
    if kind == "scalar":
        nums = [v for v in rows[0] if isinstance(v, (int, float)) and not isinstance(v, bool)]
        # a share may legitimately be reported as 34.6 (percent) or 0.346 (fraction)
        return any(_close(n, truth) or _close(n / 100, truth) or _close(n * 100, truth) for n in nums) if nums and len(rows) == 1 else False
    if kind == "label":
        want = str(truth)
        return any(_matches(v, want) for v in rows[0]) and (len(rows) == 1 or _matches(rows[0][0], want))
    if kind == "table":
        got = {}
        for r in rows:
            nums = [v for v in r if isinstance(v, (int, float)) and not isinstance(v, bool)]
            labs = [str(v) for v in r if not isinstance(v, (int, float)) or isinstance(v, bool)]
            if len(nums) == 1 and labs: got[labs[0]] = nums[0]
        return set(got) == set(truth) and all(_close(got[k], truth[k]) for k in truth)
    return False
