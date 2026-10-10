"""How trustworthy is the daily spike detector? Synthetic order data (about 8 orders a day, weekday pattern, random prices).
Prints the false-alarm rate on pure noise and the detection rate for planted spikes of different sizes.   python -m evals.bench_anomaly"""
import warnings
import numpy as np, pandas as pd
from app import drivers as D
warnings.simplefilter("ignore")


def orders(seed, days=300, base=8):
    rng = np.random.default_rng(seed); price = {"A": 100, "B": 50, "C": 20}; rows = []
    for t in pd.date_range("2025-01-01", periods=days):
        for _ in range(rng.poisson(base * (1.3 if t.dayofweek >= 5 else 1.0))):
            p = rng.choice(list(price)); q = int(rng.integers(1, 4)); rows.append((t, p, q * price[p]))
    return pd.DataFrame(rows, columns=["date", "product", "amount"])


def main(runs=150):
    fp = sum(bool(D.daily_anomalies(orders(1000 + i), "date", "amount", ["product"])) for i in range(runs))
    print(f"False alarms on pure-noise series: {fp}/{runs} ({fp / runs:.1%}) (each series is 300 days)")
    for mult in (2, 3, 5, 10):
        hits = 0
        for i in range(60):
            df = orders(5000 + i); typical = df.groupby("date")["amount"].sum().mean()
            df = pd.concat([df, pd.DataFrame({"date": [pd.Timestamp("2025-05-15")], "product": ["C"], "amount": [(mult - 1) * typical]})], ignore_index=True)
            hits += any(o["evidence"]["date"] == "2025-05-15" for o in D.daily_anomalies(df, "date", "amount", ["product"]))
        print(f"Planted spike of {mult}x a normal day found: {hits}/60 ({hits / 60:.0%})")


if __name__ == "__main__": main()
