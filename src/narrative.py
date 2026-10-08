"""Plain-English briefing generated from statistics. No API key, no model.

Templates with real numbers in them beat a vague LLM paragraph more often than people
expect, and this runs offline, instantly, and identically every time.
"""

from __future__ import annotations

import pandas as pd


def _fmt(n) -> str:
    return f"{n:,}" if isinstance(n, (int, float)) and n == n else "n/a"


def describe(store, prof: pd.DataFrame, issues: pd.DataFrame | None) -> str:
    rows = store.row_count()
    ncols = len(prof)
    lines: list[str] = []

    lines.append(f"**{_fmt(rows)} rows across {ncols} columns**, read from `{store.source}`.")

    # --- what the columns look like
    numeric = prof[prof["min"].notna()]
    text = prof[prof["min"].isna()]
    lines.append(
        f"{len(numeric)} numeric and {len(text)} non-numeric columns."
    )

    # --- completeness
    worst = prof.sort_values("null_%", ascending=False).head(3)
    bad = worst[worst["null_%"] > 0]
    if bad.empty:
        lines.append("No missing values anywhere — unusually clean.")
    else:
        parts = [f"`{r['column']}` ({r['null_%']}%)" for _, r in bad.iterrows()]
        lines.append("Most incomplete columns: " + ", ".join(parts) + ".")

    # --- cardinality signals
    ids = prof[(prof["distinct"] / max(rows, 1) > 0.95)]
    if not ids.empty:
        lines.append(
            "Likely identifier column(s): " + ", ".join(f"`{c}`" for c in ids["column"]) + "."
        )
    consts = prof[prof["distinct"] == 1]
    if not consts.empty:
        lines.append(
            "Carrying no information (single value): "
            + ", ".join(f"`{c}`" for c in consts["column"]) + "."
        )

    # --- the numbers
    if not numeric.empty:
        widest = numeric.assign(spread=numeric["max"] - numeric["min"]).sort_values(
            "spread", ascending=False
        ).head(1)
        for _, r in widest.iterrows():
            lines.append(
                f"Widest numeric range: `{r['column']}` from {_fmt(r['min'])} to "
                f"{_fmt(r['max'])} (mean {_fmt(r['mean'])})."
            )

    # --- what to fix first
    if issues is not None and not issues.empty:
        high = issues[issues["severity"] == "high"]
        med = issues[issues["severity"] == "medium"]
        lines.append("")
        lines.append(
            f"**{len(issues)} quality findings** — {len(high)} high, {len(med)} medium."
        )
        top = pd.concat([high, med]).head(3)
        if not top.empty:
            lines.append("Fix first:")
            for i, (_, r) in enumerate(top.iterrows(), 1):
                lines.append(f"{i}. **{r['column']}** — {r['issue']}: {r['detail']}")
    else:
        lines.append("")
        lines.append("**No quality issues detected.**")

    return "\n\n".join(lines)


def recommend(issues: pd.DataFrame | None) -> str:
    """Map each finding type to a concrete next action. Pure lookup, no model."""
    if issues is None or issues.empty:
        return "Nothing to action."

    actions = {
        "duplicate rows": "De-duplicate on the natural key before any aggregation — "
                          "totals are currently overstated.",
        "entirely empty": "Drop the column, or find out why it was never populated.",
        "high missing rate": "Decide a policy: exclude these rows, impute, or treat null "
                             "as a category. Document whichever you choose.",
        "constant": "Drop it. A single value cannot explain variation in anything.",
        "near-unique": "Treat as an identifier — exclude it from models and aggregations.",
        "blank strings": "Convert empty strings to NULL so they stop hiding from null checks.",
        "numeric stored as text": "Cast the column. Sorting and aggregation are wrong until "
                                  "you do, because text sorts '10' before '9'.",
        "outliers": "Inspect before removing. Some are data errors; some are your best "
                    "customers.",
        "negative values": "Check whether these are returns or refunds. If so, exclude them "
                           "from gross totals and report them separately.",
        "empty": "The file has no rows — check the export that produced it.",
    }

    out: list[str] = []
    seen: set[str] = set()
    for _, r in issues.iterrows():
        key = r["issue"]
        if key in seen:
            continue
        seen.add(key)
        advice = actions.get(key, "Review this column.")
        out.append(f"- **{key}** (`{r['column']}`) — {advice}")
    return "\n".join(out)
