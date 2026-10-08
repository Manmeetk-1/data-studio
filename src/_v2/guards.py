"""Safety rails around generated SQL.

Runs BEFORE the query touches the database. The model is allowed to be wrong; it is not
allowed to be destructive.
"""

from __future__ import annotations

import re

BLOCKED = {
    "drop", "delete", "update", "insert", "alter", "create",
    "truncate", "attach", "detach", "copy", "export", "pragma",
    "install", "load", "call",
}

DEFAULT_ROW_LIMIT = 1000

_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE)
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_HAS_LIMIT = re.compile(r"\blimit\s+\d+", re.IGNORECASE)


class UnsafeQuery(Exception):
    """Raised when generated SQL fails a guard."""


def strip_fences(sql: str) -> str:
    return _FENCE.sub("", sql.strip()).strip()


def _scrub(sql: str) -> str:
    """Comments and string literals removed, so a column called `updated_at` or a literal
    containing the word 'delete' cannot trip the keyword check."""
    out = _BLOCK_COMMENT.sub(" ", sql)
    out = _LINE_COMMENT.sub(" ", out)
    return _STRING_LITERAL.sub("''", out)


def assert_select_only(sql: str) -> None:
    if not sql or not sql.strip():
        raise UnsafeQuery("the model returned an empty query")

    body = _scrub(sql).strip().rstrip(";")
    if ";" in body:
        raise UnsafeQuery("multiple SQL statements are not allowed")

    m = re.match(r"\s*(\w+)", body)
    if not m:
        raise UnsafeQuery("could not find a SQL keyword to validate")
    head = m.group(1).lower()
    if head not in {"select", "with"}:
        raise UnsafeQuery(f"only SELECT queries are allowed, got '{head.upper()}'")

    hits = sorted(set(re.findall(r"\b\w+\b", body.lower())) & BLOCKED)
    if hits:
        raise UnsafeQuery(f"blocked keyword(s): {', '.join(hits)}")


def enforce_limit(sql: str, limit: int = DEFAULT_ROW_LIMIT) -> str:
    body = sql.rstrip().rstrip(";")
    scrubbed = _scrub(body).lower()
    if _HAS_LIMIT.search(scrubbed):
        return body
    bare_aggregate = (
        re.search(r"\b(count|sum|avg|min|max)\s*\(", scrubbed) is not None
        and "group by" not in scrubbed
    )
    if bare_aggregate:
        return body
    return f"{body}\nLIMIT {limit}"


def sanity_check(row_count: int, source_rows: int | None = None) -> str | None:
    if row_count == 0:
        return "the query ran but returned no rows -- the filters may be too narrow"
    if source_rows is not None and row_count > source_rows:
        return f"returned {row_count:,} rows from {source_rows:,} -- a join has fanned out"
    if row_count >= DEFAULT_ROW_LIMIT:
        return f"results truncated at {DEFAULT_ROW_LIMIT} rows"
    return None
