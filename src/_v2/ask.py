"""Natural language -> SQL -> validate -> run -> retry once on error.

The model receives the SCHEMA ONLY. No rows are ever sent to the API, which keeps the
prompt tiny regardless of file size and means the data never leaves the machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv

from . import guards
from .guards import UnsafeQuery

load_dotenv()

MAX_ATTEMPTS = 2
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM = """You write DuckDB SQL.
- Return ONLY SQL. No prose, no markdown fences.
- Use only the table and columns given. Never invent a column.
- Quote column names with double quotes.
- If the question is too vague to answer with one query, return exactly:
  CLARIFY: <the one question you would ask>
"""


@dataclass
class Answer:
    question: str
    sql: str | None = None
    rows: pd.DataFrame | None = None
    attempts: int = 0
    error: str | None = None
    warning: str | None = None
    refused: bool = False
    clarify: str | None = None
    tried: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.rows is not None and not self.refused and self.clarify is None


def _client():
    from groq import Groq

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set -- add it to .env")
    return Groq(api_key=key)


def ask(question: str, store, model: str = DEFAULT_MODEL) -> Answer:
    res = Answer(question=question)
    client = _client()
    last_error: str | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        res.attempts = attempt

        user = f"{store.schema_text()}\n\nQUESTION: {question}"
        if last_error:
            user += f"\n\nYour previous query failed:\n{last_error}\nReturn a corrected query."

        reply = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": user}],
        )
        sql = guards.strip_fences(reply.choices[0].message.content or "")
        res.tried.append(sql)

        if sql.upper().startswith("CLARIFY:"):
            res.clarify = sql.split(":", 1)[1].strip()
            return res

        try:
            guards.assert_select_only(sql)
            sql = guards.enforce_limit(sql)
        except UnsafeQuery as e:          # refuse, never retry
            res.refused, res.error, res.sql = True, str(e), sql
            return res

        try:
            rows = store.run(sql)
        except Exception as e:            # broken SQL -> feed the error back
            last_error = str(e).strip().splitlines()[0][:300]
            continue

        res.sql, res.rows = sql, rows
        res.warning = guards.sanity_check(len(rows), store.row_count())
        return res

    res.error = f"gave up after {MAX_ATTEMPTS} attempts. Last error: {last_error}"
    return res


def summarise(store, prof, issues, model: str = DEFAULT_MODEL) -> str:
    """Plain-English description of the dataset, from the profile only -- not the rows."""
    client = _client()
    top = issues.head(8).to_string(index=False) if issues is not None and not issues.empty else "none found"
    prompt = (
        f"Dataset: {store.row_count():,} rows.\n"
        f"Schema: {store.schema_text()}\n\n"
        f"Column profile:\n{prof.to_string(index=False)}\n\n"
        f"Quality findings:\n{top}\n\n"
        "In under 150 words, tell an analyst what this dataset appears to contain, "
        "and which two data-quality problems they should fix first and why. "
        "Be concrete. Do not invent columns."
    )
    reply = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role": "system", "content": "You are a careful data analyst."},
                  {"role": "user", "content": prompt}],
    )
    return (reply.choices[0].message.content or "").strip()
