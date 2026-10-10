"""Tests for the SQL safety layer (app/sqlguard.py).

ATTACKS must be rejected. BENIGN queries (incl. ones the old keyword blocklist wrongly refused) must pass.
The regression test at the bottom is the exact bypass that defeated the original regex guard.
Run: python -m pytest -q
"""
import pandas as pd
import pytest
from app import sqlguard


DF = pd.DataFrame({"product": ["Copy", "Drop shipping", "B"], "amount": [1, 2, 3],
                    "order_date": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"])})


# ---------------------------------------------------------------------------
# Attacks: every one of these must be REJECTED by sqlguard.guard()
# ---------------------------------------------------------------------------
ATTACKS = [
    ("file-as-table", "select * from '/etc/passwd'"),
    ("file-as-table glob", "select * from '/etc/*.conf'"),
    ("read_csv", "select * from read_csv('/etc/passwd')"),
    ("read_csv_auto", "select * from read_csv_auto('/etc/passwd')"),
    ("read_text", "select * from read_text('/etc/passwd')"),
    ("read_text scalar position", "select read_text('/etc/passwd') as x"),
    ("read_blob", "select * from read_blob('/etc/passwd')"),
    ("read_ndjson", "select * from read_ndjson('/etc/passwd')"),
    ("read_json_auto", "select * from read_json_auto('/etc/passwd')"),
    ("parquet_scan", "select * from parquet_scan('/etc/passwd')"),
    ("iceberg_scan", "select * from iceberg_scan('/etc/passwd')"),
    ("delta_scan", "select * from delta_scan('/etc/passwd')"),
    ("sniff_csv", "select * from sniff_csv('/etc/passwd')"),
    ("glob()", "select * from glob('/etc/*')"),
    ("duckdb_settings", "select * from duckdb_settings()"),
    ("duckdb_extensions", "select * from duckdb_extensions()"),
    ("pragma_version table fn", "select * from pragma_version()"),
    ("pragma_database_list", "select * from pragma_database_list()"),
    ("information_schema", "select * from information_schema.tables"),
    ("pg_timezone_names", "select * from pg_timezone_names()"),
    ("current_setting scalar", "select current_setting('memory_limit')"),
    ("getenv", "select getenv('PATH')"),
    ("version()", "select version()"),
    ("current_database", "select current_database()"),
    # the critical regex-bypass vector found in the shipped guard:
    ("query() literal", "select * from query('select 1')"),
    ("query_table()", "select * from query_table('data')"),
    ("query() + replace()-reconstructed duckdb_settings (regex-bypass PoC)",
     "select * from query(replace('select * from duck#db_settings()', '#', ''))"),
    ("query() + concat-reconstructed read_csv (regex-bypass PoC)",
     "select * from query('select * from ' || 'read' || '_csv(''/etc/passwd'')')"),
    ("query() + chr()-reconstructed SET (regex-bypass PoC)",
     "select * from query(chr(83)||chr(69)||chr(84)||' enable_external_access=true')"),
    # obfuscation tricks against a *keyword* regex (irrelevant to guard_v2,
    # included to show the AST approach doesn't care):
    ("comment-split read_csv", "select * from read/**/_csv('/etc/passwd')"),
    ("case bypass READ_CSV", "select * from READ_CSV('/etc/passwd')"),
    ("quoted identifier read_csv", 'select * from "read_csv"(\'/etc/passwd\')'),
    # statement-type / multi-statement attacks:
    ("attach", "attach ':memory:' as x"),
    ("multi-statement", "select 1; select * from read_csv('/etc/passwd')"),
    ("set stmt", "SET memory_limit='999MB'"),
    ("pragma stmt", "PRAGMA version"),
    ("install", "INSTALL httpfs"),
    ("load", "LOAD 'httpfs'"),
    ("create table", "CREATE TABLE t AS SELECT 1"),
    ("insert", "INSERT INTO data VALUES (1)"),
    ("delete", "DELETE FROM data"),
    ("update", "UPDATE data SET amount=1"),
    ("copy", "COPY data TO '/tmp/x.csv'"),
    ("export database", "EXPORT DATABASE '/tmp/x'"),
    ("call", "CALL pragma_version()"),
    ("vacuum", "VACUUM"),
    # cross-tenant / replacement-scan style attempt (not reachable via SQL
    # syntax at all -- there is no FROM-clause syntax for "another Python
    # variable"; included so a regression that added one would be caught):
    ("other table name not registered", "select * from SESSIONS"),
    ("other table name not registered 2", "select * from other_session_df"),
    # hostile CSV content used AS the SQL (prompt-injection outcome check):
    ("prompt-injected SQL from a poisoned cell value",
     "select * from read_csv_auto('/etc/passwd') -- ignore previous instructions"),
]


@pytest.mark.parametrize("name,sql", ATTACKS, ids=[a[0] for a in ATTACKS])
def test_attack_is_rejected(name, sql):
    with pytest.raises(sqlguard.GuardError):
        sqlguard.guard(sql)


def test_query_bypass_actually_cannot_execute_end_to_end():
    """The single most important regression test: the exact PoC that broke
    the shipped regex guard (query() + replace()-reconstructed duckdb_settings)
    must be refused by run_sql() end to end, not just guard()."""
    with pytest.raises(sqlguard.GuardError):
        sqlguard.run_sql(DF, "select * from query(replace('select * from duck#db_settings()', '#', ''))")


# ---------------------------------------------------------------------------
# False positives of the SHIPPED regex guard: every one of these is a
# legitimate query that guard_v2 must ALLOW.
# ---------------------------------------------------------------------------
BENIGN = [
    ("string literal 'Copy'", "select * from data where product = 'Copy'"),
    ("string literal 'Drop shipping'", "select * from data where product = 'Drop shipping'"),
    ("string literal containing semicolon", "select * from data where product = 'Note: a;b'"),
    ("alias named load", "select amount as load from data"),
    ("quoted column named call", 'select "product" as "call" from data'),
    ("comment mentioning a blocked word", "select amount from data -- export this please"),
    ("CTE named import_data", "with import_data as (select * from data) select * from import_data"),
    ("string literal containing word delete", "select * from data where product = 'Deleted item'"),
    ("column create_date style name", "select amount as create_date from data"),
    ("alias export-like name", "select amount as exported_total from data order by exported_total"),
    ("basic select star", "select * from data"),
    ("aggregate + group by", "select product, sum(amount) as total from data group by product order by total desc"),
    ("date_trunc time trend", "select date_trunc('month', order_date) as m, sum(amount) as total from data group by m order by m"),
    ("window function", "select product, sum(amount) over (partition by product) as running from data"),
    ("case expression", "select case when amount > 1 then 'big' else 'small' end as bucket, count(*) from data group by bucket"),
    ("multiple CTEs chained", "with a as (select * from data), b as (select * from a) select * from b"),
    ("self-join on data", "select a.product, b.product from data a join data b on a.amount = b.amount"),
    ("subquery in where", "select * from data where amount > (select avg(amount) from data)"),
    ("union of two data queries", "select product from data union all select product from data"),
    ("limit/order combo", "select * from data order by amount desc limit 10"),
    ("trailing semicolon from the model", "select * from data;"),
]


@pytest.mark.parametrize("name,sql", BENIGN, ids=[b[0] for b in BENIGN])
def test_benign_query_is_allowed(name, sql):
    result = sqlguard.guard(sql)
    assert result  # returns the (trimmed) SQL text, not raising


def test_benign_query_executes_end_to_end():
    out = sqlguard.run_sql(DF, "select product, sum(amount) as total from data group by product order by total desc")
    assert len(out) > 0
    assert "total" in out.columns


def test_limits_rows_and_survives_big_cross_join():
    big = pd.DataFrame({"x": range(5000)})
    out = sqlguard.run_sql(big, "select a.x from data a cross join data b", timeout=5)
    assert len(out) == sqlguard.MAX_ROWS


def test_timeout_stops_runaway_query():
    big = pd.DataFrame({"x": range(200_000)})
    with pytest.raises(Exception):
        sqlguard.run_sql(big, "select count(*) from data a cross join data b cross join data c where a.x + b.x + c.x = -1", timeout=2)


def test_config_cannot_be_changed_after_lock():
    with pytest.raises(sqlguard.GuardError):
        sqlguard.run_sql(DF, "select * from query('set enable_external_access=true')")
