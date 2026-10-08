"""Column profiling.

Every statistic is a SQL aggregate run by DuckDB, so this costs the same on a 2 GB file
as on a small one. Nothing is pulled into pandas except the final summary table.
"""

from __future__ import annotations

import pandas as pd

NUMERIC = {"BIGINT", "INTEGER", "DOUBLE", "DECIMAL", "FLOAT", "HUGEINT", "SMALLINT", "TINYINT"}


def _is_numeric(dtype: str) -> bool:
    return any(dtype.upper().startswith(n) for n in NUMERIC)


def profile(store) -> pd.DataFrame:
    """One row per column: type, nulls, distinct count, and numeric range."""
    total = store.row_count()
    rows = []

    for name, dtype in store.columns():
        q = f'''
            SELECT
                COUNT(*) FILTER (WHERE "{name}" IS NULL)              AS nulls,
                COUNT(DISTINCT "{name}")                              AS distinct_count
            FROM "{store.table}"
        '''
        nulls, distinct = store.run(q).iloc[0]

        lo = hi = mean = None
        if _is_numeric(dtype):
            stat = store.run(
                f'SELECT MIN("{name}") AS lo, MAX("{name}") AS hi, AVG("{name}") AS mean '
                f'FROM "{store.table}"'
            ).iloc[0]
            # .iloc, not [0] -- in pandas 3.x a bare integer is treated as a label
            lo, hi, mean = stat.iloc[0], stat.iloc[1], stat.iloc[2]

        rows.append({
            "column": name,
            "type": dtype,
            "nulls": int(nulls),
            "null_%": round(100 * nulls / total, 1) if total else 0.0,
            "distinct": int(distinct),
            "min": lo,
            "max": hi,
            "mean": round(mean, 2) if isinstance(mean, (int, float)) and mean is not None else None,
        })

    return pd.DataFrame(rows)


def overview(store) -> dict:
    cols = store.columns()
    return {
        "rows": store.row_count(),
        "columns": len(cols),
        "numeric_columns": sum(1 for _, t in cols if _is_numeric(t)),
        "source": store.source,
    }
