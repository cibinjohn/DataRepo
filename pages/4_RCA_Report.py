import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.filters import apply_date_filter

data = load_data()
rca = apply_date_filter(data["dataops_results"], date_col="dag_run_date")

st.title("📄 RCA Reports")

# Global DAG filter (from the sidebar)
selected_dag = st.session_state.get("selected_dag", "All")
if selected_dag != "All":
    rca = rca[rca["dag_id"] == selected_dag]

if rca.empty:
    st.info("No RCA reports found for the current filters.")
    st.stop()

rca = rca.copy()
rca["dag_run_date"] = pd.to_datetime(rca["dag_run_date"])

# DAG picker — skip it if the global filter already pins one
dag_ids = sorted(rca["dag_id"].unique())
if len(dag_ids) == 1:
    dag_id = dag_ids[0]
    st.caption(f"DAG  `{dag_id}`")
else:
    dag_id = st.selectbox("DAG", dag_ids)

reports = (
    rca[rca["dag_id"] == dag_id]
    .sort_values(["dag_run_date", "agent_version"], ascending=[False, False])
)

choice = st.selectbox(
    "Failed run",
    reports.index,
    format_func=lambda i: f"{reports.loc[i, 'dag_run_date']:%Y-%m-%d}  ·  {reports.loc[i, 'agent_version']}",
)

row = reports.loc[choice]
st.caption(
    f"`{row['dag_id']}`  ·  run {row['dag_run_date']:%Y-%m-%d}  ·  agent {row['agent_version']}"
)
st.divider()
st.markdown(row["rca_report"])