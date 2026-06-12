import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.filters import TIME_RANGES
st.set_page_config(page_title="Workspace", layout="wide")
data = load_data()

dag_runs = data["dag_runs"]
failed = data["failed_dag_runs"]
rca = data["dataops_results"]

st.title("🏠 Workspace Home")

##################################################
# TOP RIGHT FILTER BAR
##################################################

col1, col2, col3 = st.columns([2, 2, 1])

with col3:

    st.markdown("### ⚙️ Filters")

    st.session_state.time_range = st.selectbox(
        "Time Range",
        list(TIME_RANGES.keys()),
        index=list(TIME_RANGES.keys()).index(
            st.session_state.get("time_range", "Last week")
        ),
        label_visibility="collapsed"
    )

    st.session_state.selected_dag = st.selectbox(
        "DAG",
        ["All"] + sorted(dag_runs["dag_id"].unique()),
        index=0,
        label_visibility="collapsed"
    )

##################################################
# APPLY FILTERS (LOCAL TO PAGE BUT GLOBAL STATE)
##################################################

# TIME FILTER
from utils.filters import apply_time_filter

dag_runs = apply_time_filter(dag_runs)
failed = apply_time_filter(failed)
rca = apply_time_filter(rca, date_col="dag_run_date")

# DAG FILTER
selected_dag = st.session_state.selected_dag

if selected_dag != "All":
    dag_runs = dag_runs[dag_runs["dag_id"] == selected_dag]
    failed = failed[failed["dag_id"] == selected_dag]
    rca = rca[rca["dag_id"] == selected_dag]

##################################################
# METRICS
##################################################

c1, c2, c3, c4 = st.columns(4)

c1.metric("Distinct DAGs", dag_runs["dag_id"].nunique())
c2.metric("Total DAG Runs", len(dag_runs))
c3.metric("Failed Task Runs", len(failed))
c4.metric("RCA Reports", len(rca))

st.divider()

##################################################
# FAILURE TIMELINE
##################################################

st.subheader("📉 Failure Trend")

if not dag_runs.empty:

    trend = (
        dag_runs.groupby("run_date")
        .agg(failed_tasks=("num_failed_tasks", "sum"))
        .reset_index()
    )

    fig = px.line(
        trend,
        x="run_date",
        y="failed_tasks",
        title="Failure Trend Over Time"
    )

    st.plotly_chart(fig, use_container_width=True)

##################################################
# SUMMARY TABLES
##################################################

c1, c2 = st.columns(2)

with c1:
    st.subheader("Recent Failures")
    st.dataframe(
        failed.sort_values("run_date", ascending=False).head(10),
        use_container_width=True
    )

with c2:
    st.subheader("Recent DAG Runs")
    st.dataframe(
        dag_runs.sort_values("run_date", ascending=False).head(10),
        use_container_width=True
    )