"""Charts and relationships.

Aggregation happens in DuckDB; only the aggregated result reaches pandas. So a chart
over 10M rows plots the same handful of points as one over 10k.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px

from .profile import _is_numeric

TOP_N = 20


def numeric_columns(store) -> list[str]:
    return [c for c, t in store.columns() if _is_numeric(t)]


def categorical_columns(store, max_distinct: int = 50) -> list[str]:
    out = []
    for c, t in store.columns():
        if _is_numeric(t):
            continue
        n = store.run(f'SELECT COUNT(DISTINCT "{c}") n FROM "{store.table}"').iloc[0]["n"]
        if 1 < n <= max_distinct:
            out.append(c)
    return out


def date_columns(store) -> list[str]:
    return [c for c, t in store.columns() if any(k in t.upper() for k in ("DATE", "TIMESTAMP"))]


def distribution(store, column: str, bins: int = 40):
    """Histogram computed by DuckDB -- the raw column never reaches pandas.

    Buckets are derived with floor() rather than width_bucket(), which is not available
    in every DuckDB build.
    """
    rng = store.run(
        f'SELECT MIN("{column}") AS lo, MAX("{column}") AS hi FROM "{store.table}"'
    ).iloc[0]
    lo, hi = rng.iloc[0], rng.iloc[1]
    if lo is None or hi is None or lo == hi:
        return px.bar(pd.DataFrame({"bin": [lo], "count": [store.row_count()]}),
                      x="bin", y="count", title=f"{column} (single value)")

    width = (hi - lo) / bins
    q = f'''
        SELECT
          {lo} + width_idx * {width} AS bin_start,
          COUNT(*)                   AS count
        FROM (
          SELECT LEAST(FLOOR(("{column}" - {lo}) / {width}), {bins - 1}) AS width_idx
          FROM "{store.table}"
          WHERE "{column}" IS NOT NULL
        )
        GROUP BY width_idx
        ORDER BY width_idx
    '''
    df = store.run(q)
    return px.bar(df, x="bin_start", y="count", title=f"Distribution of {column}")


def top_categories(store, column: str, measure: str | None = None, n: int = TOP_N):
    if measure:
        q = (f'SELECT "{column}" AS category, SUM("{measure}") AS value FROM "{store.table}" '
             f'WHERE "{column}" IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT {n}')
        title = f"Top {n} {column} by total {measure}"
    else:
        q = (f'SELECT "{column}" AS category, COUNT(*) AS value FROM "{store.table}" '
             f'WHERE "{column}" IS NOT NULL GROUP BY 1 ORDER BY 2 DESC LIMIT {n}')
        title = f"Top {n} {column} by row count"
    return px.bar(store.run(q), x="category", y="value", title=title)


def trend(store, date_col: str, measure: str | None = None, grain: str = "month"):
    agg = f'SUM("{measure}")' if measure else "COUNT(*)"
    label = f"total {measure}" if measure else "rows"
    q = (f'SELECT date_trunc(\'{grain}\', "{date_col}") AS period, {agg} AS value '
         f'FROM "{store.table}" WHERE "{date_col}" IS NOT NULL GROUP BY 1 ORDER BY 1')
    return px.line(store.run(q), x="period", y="value", markers=True,
                   title=f"{label} by {grain}")


def correlations(store, columns: list[str]) -> pd.DataFrame | None:
    """Pearson correlation for every numeric pair, computed in SQL."""
    cols = columns[:12]
    if len(cols) < 2:
        return None
    pairs = ", ".join(
        f'corr("{a}", "{b}") AS "{a}|{b}"' for i, a in enumerate(cols) for b in cols[i + 1:]
    )
    row = store.run(f'SELECT {pairs} FROM "{store.table}"').iloc[0]
    out = [
        {"column_a": k.split("|")[0], "column_b": k.split("|")[1], "correlation": round(v, 3)}
        for k, v in row.items() if v is not None
    ]
    df = pd.DataFrame(out)
    if df.empty:
        return None
    df["strength"] = df["correlation"].abs()
    return df.sort_values("strength", ascending=False).drop(columns="strength").reset_index(drop=True)
