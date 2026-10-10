"""SQL safety layer: validate model-written SQL structurally, then run it in a locked-down DuckDB.

Why not a keyword blocklist: a regex only sees the text of the query, not what it will do. DuckDB's
`query()` table function runs a string as SQL, so `query(replace('select * from duck#db_settings()', '#', ''))`
rebuilds a blocked word at run time and slips past any blocklist. Instead we ask DuckDB's own parser what the
query *is* (`json_serialize_sql`) and allow exactly one SELECT whose only row sources are the `data` table and
its own CTEs. Table functions are rejected wherever they appear, so there is nothing to hide behind string tricks.

Layers, innermost last: structural guard -> engine sandbox (no file/network access, locked config) ->
resource limits (threads, memory, row cap) -> wall-clock interrupt.
"""
import json, re, threading
import duckdb, pandas as pd

ALLOWED_SCHEMAS = {"", "main", "memory", "system"}      # how DuckDB reports the default schema of `FROM data`
ALLOWED_CATALOGS = {"", "main", "memory", "system", "temp"}
DENIED_FROM_KINDS = {"TABLE_FUNCTION", "SHOW_REF", "PIVOT_REF"}   # row sources that are not a plain table/CTE/subquery
DENIED_FUNCS = {"current_setting", "current_database", "current_schema", "current_catalog", "version",
                "current_query", "getenv", "getvariable"}                      # functions that only disclose server state
DENIED_FUNC_RE = re.compile(r"^(read_|write_|parquet_|duckdb_|pg_|pragma_|iceberg_|delta_|sniff_csv$|glob$|query$|query_table$)", re.I)
MAX_SQL_LEN = 8000
MAX_ROWS = 1000

_local = threading.local()


class GuardError(ValueError):
    """The query is not allowed. The message is safe to show to the user."""


def _parser():
    con = getattr(_local, "con", None)
    if con is None:   # one parse-only connection per thread; it never executes the candidate SQL
        con = duckdb.connect(":memory:")
        con.execute("SET enable_external_access=false")
        con.execute("SET lock_configuration=true")
        _local.con = con
    return con


def _cte_names(node, out: set):
    if isinstance(node, dict):
        cm = node.get("cte_map")
        if isinstance(cm, dict):
            out.update(e["key"].lower() for e in cm.get("map", []) if isinstance(e.get("key"), str))
        for v in node.values(): _cte_names(v, out)
    elif isinstance(node, list):
        for v in node: _cte_names(v, out)


def _walk(node, tables: set):
    if isinstance(node, dict):
        kind = node.get("type")
        if kind == "BASE_TABLE":
            if (node.get("schema_name") or "").lower() not in ALLOWED_SCHEMAS or (node.get("catalog_name") or "").lower() not in ALLOWED_CATALOGS:
                raise GuardError("The query reads from a schema that isn't allowed.")
            if (node.get("table_name") or "").lower() not in tables:
                raise GuardError("The query reads from a table that isn't allowed. Only your uploaded data can be queried.")
        elif isinstance(kind, str) and kind in DENIED_FROM_KINDS:
            raise GuardError("The query uses a table function that isn't allowed.")
        fn = node.get("function_name")
        if isinstance(fn, str) and (fn.lower() in DENIED_FUNCS or DENIED_FUNC_RE.match(fn)):
            raise GuardError(f"The function `{fn}` isn't allowed.")
        for v in node.values(): _walk(v, tables)
    elif isinstance(node, list):
        for v in node: _walk(v, tables)


def guard(sql: str, table: str = "data") -> str:
    """Return the cleaned SQL if it is one read-only SELECT over `table` (and its own CTEs); raise GuardError otherwise."""
    if not isinstance(sql, str) or not sql.strip(): raise GuardError("Empty query.")
    if len(sql) > MAX_SQL_LEN: raise GuardError("The query is too long.")
    try: doc = json.loads(_parser().execute("select json_serialize_sql(?)", [sql]).fetchone()[0])
    except Exception as e: raise GuardError(f"Could not parse the query: {str(e)[:120]}") from e
    if doc.get("error"):   # the serializer itself refuses everything that is not a SELECT (INSERT, COPY, SET, ATTACH, ...)
        raise GuardError("Only SELECT queries are allowed.")
    stmts = doc.get("statements") or []
    if len(stmts) != 1: raise GuardError("Only one query is allowed.")
    node, tables = stmts[0].get("node"), {table.lower()}
    _cte_names(node, tables)
    _walk(node, tables)
    return sql.strip().rstrip(";").strip()


_IDENT_KEYS = {"column_names", "table_name", "alias", "function_name", "schema_name", "catalog_name"}   # case-insensitive in DuckDB


def _strip_positions(node, ident=False):
    if isinstance(node, dict): return {k: _strip_positions(v, k in _IDENT_KEYS) for k, v in node.items() if k != "query_location"}
    if isinstance(node, list): return [_strip_positions(v, ident) for v in node]
    return node.lower() if ident and isinstance(node, str) else node


def same_query(a: str, b: str) -> bool:
    """True if two queries parse to the same tree (they differ only in whitespace, case, comments or a trailing semicolon).
    Used so a 'second query' that merely repeats the first cannot count as an independent cross-check."""
    try:
        trees = [_strip_positions(json.loads(_parser().execute("select json_serialize_sql(?)", [q.strip().rstrip(";")]).fetchone()[0])) for q in (a, b)]
    except Exception: return a.strip().lower() == b.strip().lower()
    return trees[0] == trees[1]


def run_sql(df: pd.DataFrame, sql: str, timeout: int = 10, table: str = "data") -> pd.DataFrame:
    """Guard, then execute against `df` in an isolated in-memory DuckDB. Returns at most MAX_ROWS rows."""
    s = guard(sql, table)
    con = duckdb.connect(":memory:")
    try:
        con.register(table, df)
        con.execute("SET enable_external_access=false")
        con.execute("SET python_enable_replacements=false")   # explicit, rather than relying on a side effect of the line above
        con.execute("SET threads=2")
        con.execute("SET memory_limit='512MB'")
        con.execute("SET lock_configuration=true")            # last: nothing below can re-enable anything
        timer = threading.Timer(timeout, con.interrupt); timer.start()
        try: return con.execute(f"SELECT * FROM ({s}) LIMIT {MAX_ROWS}").df()
        finally: timer.cancel()
    finally: con.close()
