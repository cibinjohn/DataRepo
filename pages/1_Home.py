import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.filters import apply_date_filter
from utils.helpers import render_metric_box

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
    st.markdown("### Filters")

    today = pd.Timestamp.today().normalize().date()
    default_from, default_to = st.session_state.get(
        "date_range", (today - pd.Timedelta(days=7), today)
    )

    d1, d2 = st.columns(2)
    from_date = d1.date_input("From", value=default_from)
    to_date = d2.date_input("To", value=default_to)

    if from_date > to_date:
        st.warning("'From' date is after 'To' date — swapping them.")
        from_date, to_date = to_date, from_date

    st.session_state.date_range = (from_date, to_date)

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

dag_runs = apply_date_filter(dag_runs)
failed = apply_date_filter(failed)
rca = apply_date_filter(rca, date_col="dag_run_date")

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



# 3. Render each metric box using the function
render_metric_box(c1, "Distinct DAGs", dag_runs["dag_id"].nunique())
render_metric_box(c2, "Total DAG Runs", len(dag_runs))
render_metric_box(c3, "Failed Task Runs", len(failed))
render_metric_box(c4, "RCA Reports", len(rca))

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