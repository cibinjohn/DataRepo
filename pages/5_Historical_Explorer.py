import streamlit as st
import plotly.express as px

from utils.data_loader import load_data
from utils.filters import apply_time_filter

data = load_data()

dag_runs = apply_time_filter(data["dag_runs"])
task_runs = apply_time_filter(data["task_runs"])

st.title("📈 Historical Explorer")

selected_dag = st.session_state.get("selected_dag", "All")

if selected_dag != "All":
    dag_runs = dag_runs[dag_runs["dag_id"] == selected_dag]
    task_runs = task_runs[task_runs["dag_id"] == selected_dag]

dag = st.selectbox("DAG", sorted(dag_runs["dag_id"].unique()))

subset = dag_runs[dag_runs["dag_id"] == dag]

if not subset.empty:
    fig = px.line(
        subset,
        x="run_date",
        y="num_failed_tasks",
        title="Failed Tasks Trend"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Retry Success Analysis")

retry_stats = (
    task_runs.groupby("task_id")["has_task_succeeded_on_retry"]
    .mean()
    .reset_index()
)

st.dataframe(retry_stats.sort_values("has_task_succeeded_on_retry", ascending=False))

st.subheader("Most Frequently Failing Tasks")

failures = (
    task_runs[task_runs["error_code"] == 1]
    .groupby("task_id")
    .size()
    .reset_index(name="failures")
    .sort_values("failures", ascending=False)
)

st.bar_chart(failures.head(15), x="task_id", y="failures")