"""Built-in sample datasets (deterministic, synthetic) and the stored questions that power keyless demo mode.

Each stored question is plain SQL over the cleaned column names. In demo mode the SQL is replayed through the same
guard + sandbox as an AI-written query, so the numbers are computed live; only the question-to-SQL step is canned.
"""
import numpy as np, pandas as pd
from . import analytics as A

_CACHE: dict = {}

SAMPLES = {
    "shop": {"title": "Campus shop sales", "blurb": "Two years of orders for a student-run shop: hoodies, mugs, notebooks.",
             "questions": [
                 {"q": "Which product brings in the most revenue?", "sql": "select product, sum(amount) as revenue from data group by product order by revenue desc",
                  "chart": {"type": "bar", "x": "product", "y": "revenue"}},
                 {"q": "How did revenue change month over month?", "sql": "select date_trunc('month', order_date) as month, sum(amount) as revenue from data group by month order by month",
                  "chart": {"type": "line", "x": "month", "y": "revenue"}},
                 {"q": "Which region sells the most hoodies?", "sql": "select region, sum(quantity) as hoodies from data where product = 'Hoodie' group by region order by hoodies desc",
                  "chart": {"type": "bar", "x": "region", "y": "hoodies"}},
                 {"q": "What was the biggest single day?", "sql": "select order_date, sum(amount) as revenue from data group by order_date order by revenue desc limit 5",
                  "chart": {"type": "table"}}]},
    "donations": {"title": "NGO donations", "blurb": "Three years of donations to a small charity, with campaigns, channels and one very large gift.",
                  "questions": [
                      {"q": "Which campaign raised the most?", "sql": "select campaign, sum(amount_usd) as raised from data group by campaign order by raised desc",
                       "chart": {"type": "bar", "x": "campaign", "y": "raised"}},
                      {"q": "How did donations change month by month?", "sql": "select date_trunc('month', date) as month, sum(amount_usd) as raised from data group by month order by month",
                       "chart": {"type": "line", "x": "month", "y": "raised"}},
                      {"q": "Which channel brings in the most money?", "sql": "select channel, sum(amount_usd) as raised from data where channel is not null group by channel order by raised desc",
                       "chart": {"type": "bar", "x": "channel", "y": "raised"}},
                      {"q": "Who are the top 5 donors?", "sql": "select donor_name, sum(amount_usd) as given from data group by donor_name order by given desc limit 5",
                       "chart": {"type": "bar", "x": "donor_name", "y": "given"}}]},
    "inventory": {"title": "Warehouse inventory", "blurb": "Weekly shipments from three warehouses of a relief organisation: food, winter goods, health kits.",
                  "questions": [
                      {"q": "Which product ships the most units?", "sql": "select product, sum(units_shipped) as units from data group by product order by units desc",
                       "chart": {"type": "bar", "x": "product", "y": "units"}},
                      {"q": "How do shipments change over time?", "sql": "select date_trunc('month', week) as month, sum(units_shipped) as units from data group by month order by month",
                       "chart": {"type": "line", "x": "month", "y": "units"}},
                      {"q": "Which warehouse ships the most?", "sql": "select warehouse, sum(units_shipped) as units from data group by warehouse order by units desc",
                       "chart": {"type": "bar", "x": "warehouse", "y": "units"}}]},
}


def _donations_raw(rng):
    camps = {"Annual Appeal": .30, "Giving Tuesday": .12, "Spring Gala": .14, "Monthly Giving": .26, "School Supplies Drive": .10, "Emergency Relief": .08}
    chans = {"Online": .52, "Event": .18, "Mail": .12, "Corporate": .06, "Phone": .12}
    first, last = ["Maya", "Liam", "Noor", "Ethan", "Sofia", "Arjun", "Chloe", "Omar", "Priya", "Lucas", "Amara", "Diego", "Hana", "Jonas", "Leila"], ["Patel", "Nguyen", "Garcia", "Okafor", "Smith", "Kim", "Rossi", "Haddad"]
    names = [f"{a} {b}" for a in first for b in last]
    cp, hp = np.array(list(camps.values())) / sum(camps.values()), np.array(list(chans.values())) / sum(chans.values())
    rows = []
    for d in pd.date_range("2023-01-01", "2025-09-30"):
        season = {11: 1.6, 12: 2.6, 1: .9, 3: 1.1, 4: 1.2}.get(d.month, 1.0)
        n = rng.poisson(1.5 * season * (1 + (d - pd.Timestamp("2023-01-01")).days / 1000) * (1.2 if d.dayofweek in (1, 2) else 1))
        if d == pd.Timestamp("2024-11-26"): n += 18                                   # Giving Tuesday
        for _ in range(n):
            camp = "Giving Tuesday" if d == pd.Timestamp("2024-11-26") else rng.choice(list(camps), p=cp)
            amt = float(np.round(rng.lognormal(3.6, .8), 2)) if camp != "Monthly Giving" else float(rng.choice([10, 15, 25, 50]))
            rows.append([d.strftime("%Y-%m-%d"), rng.choice(names), camp, rng.choice(list(chans), p=hp), amt, "Yes" if camp == "Monthly Giving" else "No"])
    df = pd.DataFrame(rows, columns=["Date", "Donor Name", "Campaign", "Channel", "Amount (USD)", "Recurring"])
    big = df.index[df["Date"] >= "2024-12-10"][0]
    df.loc[big, ["Amount (USD)", "Donor Name", "Channel", "Campaign"]] = [25000.0, "Brightfield Foundation", "Corporate", "Annual Appeal"]   # one big year-end gift
    df.loc[df.sample(frac=.06, random_state=1).index, "Channel"] = np.nan             # some missing channels, as in real exports
    return df


def _inventory_raw(rng):
    prods = {"Rice 25kg": ("Pantry", 31), "Canned beans": ("Pantry", 1.2), "Olive oil 5L": ("Pantry", 22), "Blankets": ("Winter", 9),
             "Winter coats": ("Winter", 28), "Hygiene kits": ("Health", 6.5), "First-aid kits": ("Health", 14), "School bags": ("Education", 8)}
    rows = []
    for d in pd.date_range("2024-01-01", "2025-06-30", freq="W-MON"):
        for p, (cat, cost) in prods.items():
            for w in ("Central", "North depot", "Harbour"):
                u = int(max(0, rng.normal(40 * (1.8 if (cat == "Winter" and d.month in (10, 11, 12, 1)) else 1.0), 12)))
                rows.append([d.strftime("%Y-%m-%d"), p, cat, w, u, round(u * cost, 2), cost, int(max(0, rng.normal(200, 60)))])
    return pd.DataFrame(rows, columns=["week", "product", "category", "warehouse", "units_shipped", "cost", "unit_price", "stock_on_hand"])


def raw_sample(name: str) -> pd.DataFrame:
    """The file as a user would have uploaded it (before cleaning); cached because the data is deterministic."""
    if name not in SAMPLES: raise KeyError(name)
    if name not in _CACHE:
        _CACHE[name] = {"shop": lambda: A.demo_df(), "donations": lambda: _donations_raw(np.random.default_rng(23)),
                        "inventory": lambda: _inventory_raw(np.random.default_rng(23))}[name]()
    return _CACHE[name].copy()


def listing():
    return [{"id": k, "title": v["title"], "blurb": v["blurb"], "questions": [q["q"] for q in v["questions"]]} for k, v in SAMPLES.items()]


def find_stored(sample: str | None, question: str):
    """The stored question matching what the user typed or clicked (case/punctuation-insensitive), or None."""
    if not sample or sample not in SAMPLES: return None
    norm = lambda s: " ".join("".join(ch.lower() if ch.isalnum() else " " for ch in s).split())
    return next((q for q in SAMPLES[sample]["questions"] if norm(q["q"]) == norm(question)), None)
