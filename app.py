"""Data Studio -- profile, audit, visualise and model any dataset.

Six tabs, no API keys, any file size.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import plotly.express as px
import streamlit as st

from src import model as M
from src import narrative
from src import profile as prof_mod
from src import quality, viz
from src.store import Store

st.set_page_config(page_title="Data Studio", layout="wide")
SEV = {"high": "🔴", "medium": "🟠", "low": "🟡"}


@st.cache_resource(show_spinner=False)
def load(path: str) -> Store:
    s = Store()
    s.register_file(path)
    return s


@st.cache_data(show_spinner=False)
def cached_profile(path: str):
    return prof_mod.profile(load(path))


@st.cache_data(show_spinner=False)
def cached_quality(path: str):
    s = load(path)
    return quality.checks(s, prof_mod.profile(s))


st.title("Data Studio")
st.caption(
    "Profiles, audits, charts and models any dataset. Files are ingested once into "
    "DuckDB's columnar format, so a 500,000-row CSV profiles in under a second."
)

up = st.file_uploader("Upload CSV, Parquet, JSON or Excel",
                      type=["csv", "parquet", "json", "xlsx", "xls"])
typed = st.text_input("…or paste a path to a file on disk", placeholder=r"C:\data\file.csv")

path = None
if up is not None:
    tmp = Path(tempfile.gettempdir()) / up.name
    tmp.write_bytes(up.getbuffer())
    path = str(tmp)
elif typed.strip():
    path = typed.strip()

if not path:
    st.info("Give it a file to begin.")
    st.stop()

try:
    store = load(path)
except Exception as e:
    st.error(f"Could not read that file: {e}")
    st.stop()

ov = prof_mod.overview(store)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{ov['rows']:,}")
c2.metric("Columns", ov["columns"])
c3.metric("Numeric columns", ov["numeric_columns"])
c4.metric("Loaded as", store.mode)

t_prev, t_prof, t_qual, t_viz, t_model, t_ins = st.tabs(
    ["Preview", "Profile", "Quality", "Visualise", "Model", "Insights"]
)

# ------------------------------------------------------------------ preview
with t_prev:
    st.dataframe(store.head(50), use_container_width=True)
    st.code(store.schema_text())

# ------------------------------------------------------------------ profile
with t_prof:
    prof = cached_profile(path)
    st.dataframe(prof, use_container_width=True)
    st.download_button("Download profile (CSV)", prof.to_csv(index=False),
                       "profile.csv", "text/csv")

# ------------------------------------------------------------------ quality
with t_qual:
    issues = cached_quality(path)
    if issues.empty:
        st.success("No quality issues found.")
    else:
        counts = issues["severity"].value_counts()
        cs = st.columns(3)
        for i, s in enumerate(["high", "medium", "low"]):
            cs[i].metric(f"{SEV[s]} {s}", int(counts.get(s, 0)))
        st.divider()
        for _, r in issues.iterrows():
            st.write(f"{SEV[r['severity']]} **{r['column']}** — {r['issue']}: {r['detail']}")
        st.download_button("Download findings (CSV)", issues.to_csv(index=False),
                           "quality_findings.csv", "text/csv")

# ---------------------------------------------------------------- visualise
with t_viz:
    nums = viz.numeric_columns(store)
    cats = viz.categorical_columns(store)
    dates = viz.date_columns(store)

    if not (nums or cats or dates):
        st.info("No chartable columns detected.")

    if nums:
        st.subheader("Distribution")
        col = st.selectbox("Numeric column", nums, key="dist")
        st.plotly_chart(viz.distribution(store, col), use_container_width=True)

    if cats:
        st.subheader("Breakdown by category")
        a, b = st.columns(2)
        cat = a.selectbox("Group by", cats, key="cat")
        meas = b.selectbox("Measure", ["(row count)"] + nums, key="meas")
        st.plotly_chart(
            viz.top_categories(store, cat, None if meas == "(row count)" else meas),
            use_container_width=True)

    if dates:
        st.subheader("Trend over time")
        a, b, c = st.columns(3)
        dcol = a.selectbox("Date column", dates, key="dt")
        tm = b.selectbox("Measure", ["(row count)"] + nums, key="tmeas")
        grain = c.selectbox("Grain", ["day", "week", "month", "quarter", "year"], index=2)
        st.plotly_chart(
            viz.trend(store, dcol, None if tm == "(row count)" else tm, grain),
            use_container_width=True)

    if len(nums) >= 2:
        st.subheader("Correlations")
        corr = viz.correlations(store, nums)
        if corr is not None:
            st.dataframe(corr, use_container_width=True)
            st.caption("Pearson, computed in SQL. Correlation is not causation.")

# -------------------------------------------------------------------- model
with t_model:
    st.caption(
        "Trains a Random Forest **and** a dummy baseline. If the model cannot beat the "
        "baseline, that is the finding — not a failure to hide."
    )
    sample = store.run(f'SELECT * FROM "{store.table}" USING SAMPLE 20000 ROWS')
    targets = M.candidate_targets(sample)

    if not targets:
        st.info("No suitable target column found.")
    else:
        target = st.selectbox("Column to predict", targets)
        if st.button("Train model"):
            with st.spinner("training…"):
                try:
                    rep = M.train(sample, target)
                except Exception as e:
                    st.error(str(e))
                    st.stop()

            a, b = st.columns(2)
            a.metric(f"Model {rep.metric_name}", f"{rep.model_score:.3f}")
            b.metric(f"Baseline {rep.metric_name}", f"{rep.baseline_score:.3f}")

            if rep.beats_baseline:
                st.success(f"Beats the baseline — the other columns do carry signal about {target}.")
            else:
                st.warning(
                    f"Does **not** beat the baseline. On this data, {target} is not "
                    "predictable from the other columns — reporting that honestly is the result."
                )
            st.caption(f"{rep.task} · trained on {rep.n_rows:,} rows · {rep.note}")
            st.plotly_chart(
                px.bar(rep.importances.head(10), x="importance", y="feature",
                       orientation="h", title="Feature importance"),
                use_container_width=True)

# ----------------------------------------------------------------- insights
with t_ins:
    st.subheader("Automatic briefing")
    st.markdown(narrative.describe(store, cached_profile(path), cached_quality(path)))
    st.divider()
    st.subheader("Recommended actions")
    st.markdown(narrative.recommend(cached_quality(path)))
