# Data Studio

Profile, audit, visualise and model **any dataset** — CSV, Parquet, JSON or Excel — in the
browser. No API keys, no configuration.

Built because every analysis starts the same way: *what is in this file, and can I trust it?*

---

## What it does

| Tab | What you get |
|---|---|
| **Preview** | First 50 rows and the inferred schema |
| **Profile** | Per column: type, null count and %, distinct values, min / max / mean |
| **Quality** | Seven classes of data problem, ranked by severity |
| **Visualise** | Distributions, category breakdowns, time trends, correlation table |
| **Model** | Train a Random Forest against a dummy baseline, with feature importance |
| **Insights** | A written briefing and a prioritised action list |

## How it handles large files

Files are **ingested once** into DuckDB's compressed columnar format rather than queried
as a view. The difference is not small:

| Step | As a view (re-parses each query) | Ingested once |
|---|---|---|
| Profile | 24.5s | **0.35s** |
| Quality checks | 41.2s | **1.46s** |

Measured on a 501,200-row, 31 MB CSV. Every statistic is a SQL aggregate, so nothing is
pulled into memory except the final summary table, and the database lives on disk rather
than in RAM.

## What the quality checker finds

Duplicate rows · entirely empty columns · high missing rates · constant columns ·
near-unique identifier columns · blank strings masquerading as values · numbers stored as
text · outliers by the 1.5×IQR rule · negative values in quantity-like columns.

On a test file it correctly flagged all of: 1,200 duplicate rows, 15,098 negative
quantities, a 40%-empty column, a constant column, an identifier column, and 1,026
price outliers.

## The modelling tab reports when the model is useless

Every model is trained alongside a **dummy baseline** — majority class for classification,
mean for regression. If the Random Forest cannot beat it, the app says so plainly rather
than printing an accuracy figure and letting you assume it means something.

That is the whole point. An accuracy of 0.17 looks like a result until you learn the
baseline was 0.169.

---

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
streamlit run app.py
```

Then upload a file, or paste a path to one.

Generate a messy demo file to try it on:

```bash
python make_demo_data.py --rows 500000
```

## Stack

DuckDB · pandas · Streamlit · plotly · scikit-learn

## Layout

```
src/store.py      ingest + schema introspection
src/profile.py    per-column statistics, all SQL aggregates
src/quality.py    the rule engine
src/viz.py        charts, aggregated in SQL before plotting
src/model.py      train/test, baseline comparison, feature importance
src/narrative.py  the written briefing, template-driven
src/_v2/          deferred: natural-language querying via an LLM
```

## Roadmap

- Natural-language questions → SQL, with read-only guards (code in `src/_v2/`)
- Export a cleaned copy of the dataset with fixes applied
- Multi-file joins
