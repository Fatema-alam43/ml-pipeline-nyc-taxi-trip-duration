"""Run the SQL files in sql/ with DuckDB (an in-process analytical database - no server needed)."""

import logging
import re
from pathlib import Path

import duckdb
import pandas as pd

from src.config import PROJECT_ROOT

log = logging.getLogger(__name__)
SQL_DIR = PROJECT_ROOT / "sql"
_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def render(sql: str, params: dict | None = None) -> str:
    """Fill {{ name }} placeholders. Only numbers and file paths from config.yaml are inserted."""
    params = params or {}

    def substitute(match):
        name = match.group(1)
        if name not in params:
            raise KeyError(f"SQL placeholder '{{{{ {name} }}}}' has no value")
        value = params[name]
        if isinstance(value, Path):
            return value.as_posix().replace("'", "''")
        if isinstance(value, int | float):
            return repr(value)
        raise TypeError(f"Parameter '{name}' must be a number or a path, got {type(value).__name__}")

    return _PLACEHOLDER.sub(substitute, sql)


def read_sql(name: str, params: dict | None = None) -> str:
    return render((SQL_DIR / name).read_text(encoding="utf-8"), params)


def connect() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB connection with the project's SQL macros loaded."""
    con = duckdb.connect()
    con.execute(read_sql("00_macros.sql"))
    return con


def execute(con, name: str, params: dict | None = None) -> None:
    """Run a SQL file that creates tables or views (may contain several statements)."""
    log.info("Running sql/%s", name)
    con.execute(read_sql(name, params))


def query(con, name: str, params: dict | None = None) -> pd.DataFrame:
    """Run a SQL file containing one SELECT and return the result as a DataFrame."""
    return con.sql(read_sql(name, params)).df()
