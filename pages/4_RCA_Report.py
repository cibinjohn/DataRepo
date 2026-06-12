import streamlit as st
from utils.data_loader import load_data
from utils.filters import apply_time_filter

data = load_data()

rca = apply_time_filter(data["dataops_results"], date_col="dag_run_date")

st.title("📄 RCA Reports")

selected_dag = st.session_state.get("selected_dag", "All")

if selected_dag != "All":
    rca = rca[rca["dag_id"] == selected_dag]

if rca.empty:
    st.info("No RCA reports found")
    st.stop()

dag_id = st.selectbox("DAG", sorted(rca["dag_id"].unique()))

report = rca[rca["dag_id"] == dag_id]

st.dataframe(report[["dag_run_date", "agent_version"]])

idx = st.selectbox("Select Report", report.index)

st.markdown(report.loc[idx, "rca_report"])