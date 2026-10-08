"""Data-quality checks, all expressed as SQL so they scale with the file.

Each check returns findings with a severity, so the output is a prioritised list rather
than a wall of statistics.
"""

from __future__ import annotations

import pandas as pd

from .profile import _is_numeric

HIGH_NULL = 30.0        # percent
NEAR_CONSTANT = 0.99    # share of rows holding one value


def checks(store, prof: pd.DataFrame) -> pd.DataFrame:
    total = store.row_count()
    found: list[dict] = []

    def add(severity: str, column: str, issue: str, detail: str) -> None:
        found.append({"severity": severity, "column": column, "issue": issue, "detail": detail})

    # --- whole-table -----------------------------------------------------
    cols = [c for c, _ in store.columns()]
    col_list = ", ".join(f'"{c}"' for c in cols)
    dupes = store.run(
        f'SELECT COUNT(*) AS n FROM (SELECT {col_list}, COUNT(*) c '
        f'FROM "{store.table}" GROUP BY ALL HAVING c > 1)'
    ).iloc[0]["n"]
    if dupes:
        add("high", "(table)", "duplicate rows",
            f"{int(dupes):,} value combinations appear more than once")

    if total == 0:
        add("high", "(table)", "empty", "the file has no rows")
        return pd.DataFrame(found)

    # --- per column ------------------------------------------------------
    for _, r in prof.iterrows():
        col, dtype, nullpct, distinct = r["column"], r["type"], r["null_%"], r["distinct"]

        if nullpct == 100:
            add("high", col, "entirely empty", "every value is null")
            continue
        if nullpct >= HIGH_NULL:
            add("medium", col, "high missing rate", f"{nullpct}% of values are null")

        if distinct == 1:
            add("medium", col, "constant", "only one distinct value -- carries no information")
        elif distinct / total >= NEAR_CONSTANT and distinct > 1 and not _is_numeric(dtype):
            add("low", col, "near-unique", "almost every row differs -- likely an identifier")

        # blank strings masquerading as values
        if dtype.upper().startswith("VARCHAR"):
            blanks = store.run(
                f'SELECT COUNT(*) n FROM "{store.table}" WHERE trim("{col}") = \'\''
            ).iloc[0]["n"]
            if blanks:
                add("medium", col, "blank strings",
                    f"{int(blanks):,} rows are empty text, not null -- these hide from null checks")

            # numbers stored as text
            numeric_like = store.run(
                f'SELECT COUNT(*) n FROM "{store.table}" '
                f'WHERE "{col}" IS NOT NULL AND TRY_CAST("{col}" AS DOUBLE) IS NOT NULL'
            ).iloc[0]["n"]
            nonnull = total - int(r["nulls"])
            if nonnull and numeric_like / nonnull > 0.95:
                add("medium", col, "numeric stored as text",
                    "over 95% of values parse as numbers -- the column type is probably wrong")

        # outliers, by the 1.5*IQR rule
        if _is_numeric(dtype) and distinct > 10:
            q = store.run(
                f'SELECT quantile_cont("{col}", 0.25) q1, quantile_cont("{col}", 0.75) q3 '
                f'FROM "{store.table}"'
            ).iloc[0]
            q1, q3 = q["q1"], q["q3"]
            if q1 is not None and q3 is not None and q3 > q1:
                iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                n = store.run(
                    f'SELECT COUNT(*) n FROM "{store.table}" '
                    f'WHERE "{col}" < {lo} OR "{col}" > {hi}'
                ).iloc[0]["n"]
                if n:
                    pct = round(100 * n / total, 1)
                    sev = "medium" if pct > 5 else "low"
                    add(sev, col, "outliers",
                        f"{int(n):,} values ({pct}%) outside 1.5*IQR [{lo:,.2f}, {hi:,.2f}]")

            neg = store.run(
                f'SELECT COUNT(*) n FROM "{store.table}" WHERE "{col}" < 0'
            ).iloc[0]["n"]
            if neg and any(k in col.lower() for k in ("price", "amount", "qty", "quantity", "total", "cost")):
                add("medium", col, "negative values",
                    f"{int(neg):,} negative values in a column that looks like a quantity or amount")

    out = pd.DataFrame(found)
    if out.empty:
        return out
    order = {"high": 0, "medium": 1, "low": 2}
    return out.sort_values("severity", key=lambda s: s.map(order)).reset_index(drop=True)
