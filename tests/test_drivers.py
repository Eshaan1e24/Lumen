"""drivers.py: planted effects must be found for the right reason; pure noise must (almost) never be flagged."""
import numpy as np, pandas as pd, pytest
from app import analytics as A, drivers as D


def _orders(seed, days=365, start="2025-01-01", products=("A", "B", "C"), base=10, price=None):
    rng = np.random.default_rng(seed); price = price or {"A": 100, "B": 50, "C": 20}
    rows = []
    for t in pd.date_range(start, periods=days):
        n = rng.poisson(base * (1.3 if t.dayofweek >= 5 else 1.0))
        for _ in range(n):
            p = rng.choice(products); q = int(rng.integers(1, 4)); rows.append((t, p, q, price[p], q * price[p]))
    return pd.DataFrame(rows, columns=["date", "product", "qty", "price", "amount"])


def _prep(df):
    s, freq = A.period_series(df, "date", "amount"); return s, freq


def test_bridge_adds_up_and_finds_planted_segment():
    df = _orders(1, days=212)                                  # 1 Jan .. 31 Jul: July is the latest COMPLETE month
    last_month = df["date"] >= pd.Timestamp("2025-07-01")
    extra = df[last_month & (df["product"] == "B")].copy(); extra["amount"] *= 2.0     # B doubles in the latest month
    df = pd.concat([df, extra], ignore_index=True)
    s, freq = _prep(df); assert freq == "MS"
    f = D.bridge(df, "date", "amount", ["product"], freq, s)
    assert f and f["kind"] == "change"
    ev = f["evidence"]; top = ev["contributions"][0]
    assert top["segment"] == "B" and top["change"] > 0
    assert ev["contributions_sum_to_change"]
    assert abs(ev["current_total"] - ev["previous_total"] * ev["like_for_like_scale"] - ev["change"]) < 1e-6      # change is like for like (July has 31 days, June 30)


def test_price_volume_split_is_exact_and_attributes_price_rise():
    rng = np.random.default_rng(3); rows = []
    for t in pd.date_range("2025-01-01", periods=120):
        for p in ("A", "B"):
            q = 5; price = 10 if p == "A" else 20
            if t >= pd.Timestamp("2025-04-01") and p == "A": price = 12      # +20% price on A in the latest month, same units
            rows.append((t, p, q, q * price))
    df = pd.DataFrame(rows, columns=["date", "product", "qty", "amount"])
    s, freq = _prep(df)
    f = D.bridge(df, "date", "amount", ["product"], freq, s, qty="qty")
    pv = f["evidence"]["price_volume"]
    assert abs(pv["volume_effect"]) < 1e-6 and pv["price_effect"] > 0
    assert abs(pv["volume_effect"] + pv["price_effect"] - f["evidence"]["change"]) < 1e-6


def test_no_bridge_when_nothing_changed():
    df = _orders(5, days=212); s, freq = _prep(df)
    assert D.bridge(df, "date", "amount", ["product"], freq, s, min_change=0.5) is None


def test_demo_spike_is_found_with_cause():
    df = A.demo_df()
    out = D.daily_anomalies(df, "order_date", "amount", ["product", "region"])
    assert any(o["evidence"]["date"] == "2025-03-14" for o in out), out
    assert len(out) <= 2


def test_planted_one_day_spike_found_in_right_segment():
    df = _orders(11, days=200); t = pd.Timestamp("2025-04-10")
    extra = pd.DataFrame({"date": [t] * 40, "product": ["C"] * 40, "qty": 1, "price": 20, "amount": 800.0})
    df = pd.concat([df, extra], ignore_index=True)
    out = D.daily_anomalies(df, "date", "amount", ["product"])
    hit = [o for o in out if o["evidence"]["date"] == "2025-04-10"]
    assert hit and hit[0]["evidence"]["segment"]["value"] == "C"


def test_false_positive_rate_on_pure_noise():
    runs, flagged = 150, 0
    for seed in range(runs):
        df = _orders(1000 + seed, days=300, base=8)             # same process every day: nothing to find
        if D.daily_anomalies(df, "date", "amount", ["product"]): flagged += 1
    assert flagged / runs <= 0.05, f"{flagged}/{runs} noise series were flagged"


@pytest.mark.parametrize("df", [pd.DataFrame({"date": pd.to_datetime([]), "amount": [], "p": []}),
                                pd.DataFrame({"date": pd.to_datetime(["2025-01-01"]), "amount": [1.0], "p": ["a"]}),
                                pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-02"] * 20), "amount": [np.nan] * 40, "p": ["a"] * 40})])
def test_never_raises_on_junk(df):
    assert D.daily_anomalies(df, "date", "amount", ["p"]) == []


def test_large_spikes_are_detected_reliably():
    """A 5x day is the kind of event worth flagging; 2x on ~8 orders a day is inside normal noise and is (honestly) not."""
    hits = 0
    for i in range(40):
        df = _orders(7000 + i, days=300, base=8); t0 = pd.Timestamp("2025-05-15")
        typical = df.groupby("date")["amount"].sum().mean()                  # one extra day's worth x4 => about 5x a normal day
        extra = pd.DataFrame({"date": [t0], "product": ["C"], "qty": 1, "price": 20, "amount": 4 * typical})
        df = pd.concat([df, extra], ignore_index=True)
        hits += any(o["evidence"]["date"] == "2025-05-15" for o in D.daily_anomalies(df, "date", "amount", ["product"]))
    assert hits >= 34, f"only {hits}/40 five-fold spikes found"
