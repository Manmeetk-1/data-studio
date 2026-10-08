"""Build SQL from UI selections. No model, no API key.

The generated SQL is shown to the user, so this doubles as a teaching tool -- pick
options, see the query that results.
"""

from __future__ import annotations

AGGREGATES = ["COUNT(*)", "SUM", "AVG", "MIN", "MAX", "COUNT DISTINCT"]
OPERATORS = ["=", "!=", ">", ">=", "<", "<=", "LIKE", "IS NULL", "IS NOT NULL"]


def build(
    table: str,
    select_cols: list[str] | None = None,
    group_by: str | None = None,
    aggregate: str = "COUNT(*)",
    agg_column: str | None = None,
    where: list[tuple[str, str, str]] | None = None,
    order_desc: bool = True,
    limit: int = 100,
) -> str:
    """Compose a SELECT. Values are quoted defensively; identifiers are double-quoted."""
    # --- measure
    if aggregate == "COUNT(*)":
        measure = "COUNT(*) AS value"
    elif aggregate == "COUNT DISTINCT" and agg_column:
        measure = f'COUNT(DISTINCT "{agg_column}") AS value'
    elif agg_column:
        measure = f'{aggregate}("{agg_column}") AS value'
    else:
        measure = "COUNT(*) AS value"

    # --- projection
    if group_by:
        select = f'"{group_by}" AS category, {measure}'
    elif select_cols:
        select = ", ".join(f'"{c}"' for c in select_cols)
    else:
        select = "*"

    sql = [f"SELECT {select}", f'FROM "{table}"']

    # --- filters
    clauses = []
    for col, op, val in (where or []):
        if op in ("IS NULL", "IS NOT NULL"):
            clauses.append(f'"{col}" {op}')
        elif op == "LIKE":
            clauses.append(f"\"{col}\" LIKE '%{val}%'")
        else:
            try:
                float(val)
                clauses.append(f'"{col}" {op} {val}')
            except (TypeError, ValueError):
                safe = str(val).replace("'", "''")
                clauses.append(f"\"{col}\" {op} '{safe}'")
    if clauses:
        sql.append("WHERE " + " AND ".join(clauses))

    if group_by:
        sql.append('GROUP BY "' + group_by + '"')
        sql.append(f"ORDER BY value {'DESC' if order_desc else 'ASC'}")
    elif select_cols and order_desc:
        sql.append(f'ORDER BY "{select_cols[0]}" DESC')

    sql.append(f"LIMIT {int(limit)}")
    return "\n".join(sql)
