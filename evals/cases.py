"""20 plain-English questions about the sample shop data, each with
  * `truth`: the correct answer computed independently in pandas (not SQL), and
  * `ref_sql`: a hand-written reference query, tested to reproduce `truth` through Lumen's own guard and sandbox.
Used by evals/run_eval.py to score the live Gemini pipeline, and by tests/test_evals.py to check the benchmark itself."""
import pandas as pd
from app import analytics as A


def load() -> pd.DataFrame:
    return A.preprocess_df(A.demo_df())      # the same cleaned table a user sees in Lumen


def cases(df: pd.DataFrame) -> list[dict]:
    d = df.copy(); d["month"] = d["order_date"].dt.to_period("M").dt.to_timestamp(); d["year"] = d["order_date"].dt.year
    return [
        dict(q="What is the total revenue?", kind="scalar", truth=float(d.amount.sum()), ref_sql="select sum(amount) from data"),
        dict(q="How many orders are there in total?", kind="scalar", truth=float(len(d)), ref_sql="select count(*) from data"),
        dict(q="Which product brings in the most revenue?", kind="label", truth=d.groupby("product").amount.sum().idxmax(), ref_sql="select product from data group by product order by sum(amount) desc limit 1"),
        dict(q="Which region has the lowest total sales?", kind="label", truth=d.groupby("region").amount.sum().idxmin(), ref_sql="select region from data group by region order by sum(amount) asc limit 1"),
        dict(q="What was the average order value?", kind="scalar", truth=float(d.amount.mean()), ref_sql="select avg(amount) from data"),
        dict(q="How many hoodies were sold?", kind="scalar", truth=float(d.loc[d["product"] == "Hoodie", "quantity"].sum()), ref_sql="select sum(quantity) from data where product = 'Hoodie'"),
        dict(q="What was total revenue in 2025?", kind="scalar", truth=float(d.loc[d.year == 2025, "amount"].sum()), ref_sql="select sum(amount) from data where year(order_date) = 2025"),
        dict(q="Which month had the highest revenue?", kind="label", truth=d.groupby("month").amount.sum().idxmax().strftime("%Y-%m"), ref_sql="select date_trunc('month', order_date) from data group by 1 order by sum(amount) desc limit 1"),
        dict(q="What share of revenue comes from the Campus region?", kind="scalar", truth=float(d.loc[d.region == "Campus", "amount"].sum() / d.amount.sum()), ref_sql="select sum(case when region = 'Campus' then amount else 0 end) / sum(amount) from data"),
        dict(q="Which product sold the most units?", kind="label", truth=d.groupby("product").quantity.sum().idxmax(), ref_sql="select product from data group by product order by sum(quantity) desc limit 1"),
        dict(q="How much revenue did Mugs bring in online?", kind="scalar", truth=float(d.loc[(d["product"] == "Mug") & (d.region == "Online"), "amount"].sum()), ref_sql="select sum(amount) from data where product = 'Mug' and region = 'Online'"),
        dict(q="How many distinct products do we sell?", kind="scalar", truth=float(d["product"].nunique()), ref_sql="select count(distinct product) from data"),
        dict(q="What was the biggest single order?", kind="scalar", truth=float(d.amount.max()), ref_sql="select max(amount) from data"),
        dict(q="Revenue by region", kind="table", truth=d.groupby("region").amount.sum().to_dict(), ref_sql="select region, sum(amount) from data group by region"),
        dict(q="Total revenue per product in 2024", kind="table", truth=d[d.year == 2024].groupby("product").amount.sum().to_dict(), ref_sql="select product, sum(amount) from data where year(order_date) = 2024 group by product"),
        dict(q="How many orders were on weekends?", kind="scalar", truth=float((d.order_date.dt.dayofweek >= 5).sum()), ref_sql="select count(*) from data where dayofweek(order_date) in (0, 6)"),
        dict(q="What day had the highest total revenue?", kind="label", truth=d.groupby("order_date").amount.sum().idxmax().strftime("%Y-%m-%d"), ref_sql="select order_date from data group by order_date order by sum(amount) desc limit 1"),
        dict(q="What was the average quantity per order for notebooks?", kind="scalar", truth=float(d.loc[d["product"] == "Notebook", "quantity"].mean()), ref_sql="select avg(quantity) from data where product = 'Notebook'"),
        dict(q="Which region sold the most hoodies by revenue?", kind="label", truth=d[d["product"] == "Hoodie"].groupby("region").amount.sum().idxmax(), ref_sql="select region from data where product = 'Hoodie' group by region order by sum(amount) desc limit 1"),
        dict(q="Revenue growth 2025 vs 2024 in percent", kind="scalar", truth=float((d[d.year == 2025].amount.sum() / d[d.year == 2024].amount.sum() - 1) * 100),
             ref_sql="select (sum(case when year(order_date) = 2025 then amount end) / sum(case when year(order_date) = 2024 then amount end) - 1) * 100 from data"),
    ]
