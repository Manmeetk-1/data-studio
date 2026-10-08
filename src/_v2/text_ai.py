"""LLM work that is not SQL.

Two jobs:
  1. summarise()  -- plain-English briefing from the profile, for a stakeholder
  2. themes()     -- recurring themes in a free-text column, which SQL cannot do

Both send aggregates or a small sample, never the whole column.
"""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def _client():
    from groq import Groq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set -- add it to .env")
    return Groq(api_key=key)


def _chat(system: str, user: str, model: str, temperature: float = 0.2) -> str:
    reply = _client().chat.completions.create(
        model=model, temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (reply.choices[0].message.content or "").strip()


def summarise(store, prof: pd.DataFrame, issues: pd.DataFrame | None,
              model: str = DEFAULT_MODEL) -> str:
    """A briefing an analyst could paste into an email."""
    top = (issues.head(8).to_string(index=False)
           if issues is not None and not issues.empty else "none found")
    user = (
        f"Dataset: {store.row_count():,} rows.\n"
        f"Schema: {store.schema_text()}\n\n"
        f"Column profile:\n{prof.to_string(index=False)}\n\n"
        f"Quality findings:\n{top}\n\n"
        "In under 150 words: what does this dataset appear to contain, and which two "
        "data-quality problems should be fixed first, and why? Be concrete. "
        "Do not invent columns or numbers."
    )
    return _chat("You are a careful data analyst. You never invent figures.", user, model)


def themes(store, column: str, sample: int = 120, model: str = DEFAULT_MODEL) -> str:
    """Recurring themes in a free-text column -- the thing GROUP BY cannot tell you."""
    rows = store.run(
        f'SELECT "{column}" AS v FROM "{store.table}" '
        f'WHERE "{column}" IS NOT NULL AND trim("{column}") <> \'\' '
        f'USING SAMPLE {sample} ROWS'
    )
    values = [str(v) for v in rows["v"].tolist()][:sample]
    if not values:
        return "No non-empty text values in that column."

    joined = "\n".join(f"- {v[:200]}" for v in values)
    user = (
        f"Here is a random sample of {len(values)} values from the text column "
        f"'{column}':\n\n{joined}\n\n"
        "Identify the 3-5 recurring themes. For each: a short label, roughly how common "
        "it is in this sample, and one representative example. If the values are not "
        "meaningful free text, say so plainly instead of inventing themes."
    )
    return _chat("You summarise qualitative data honestly, and say when there is nothing to find.",
                 user, model)


def explain_findings(issues: pd.DataFrame, model: str = DEFAULT_MODEL) -> str:
    """Turn the quality table into prioritised, plain-English actions."""
    if issues is None or issues.empty:
        return "No quality issues were found to explain."
    user = (
        f"These are automated data-quality findings:\n\n{issues.to_string(index=False)}\n\n"
        "Rewrite them as a prioritised action list for a data analyst. For each, say what "
        "the likely cause is and what to do. Be brief. Do not invent findings."
    )
    return _chat("You are a pragmatic data engineer.", user, model)
