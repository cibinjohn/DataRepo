import streamlit as st
from utils.data_loader import load_data
from utils.filters import apply_time_filter

data = load_data()

failed = apply_time_filter(data["failed_dag_runs"])

st.title("📥 Failure Inbox")

# Optional DAG filter from global state
selected_dag = st.session_state.get("selected_dag", "All")

if selected_dag != "All":
    failed = failed[failed["dag_id"] == selected_dag]

dags = (
    failed.groupby(["dag_id", "run_date"])
    .agg(failed_tasks=("task_id", "nunique"))
    .reset_index()
)

search = st.text_input("Search DAG")

if search:
    dags = dags[dags["dag_id"].str.contains(search, case=False)]

for _, row in dags.iterrows():

    with st.expander(f"{row.dag_id} | {row.run_date}"):

        tasks = failed[
            (failed["dag_id"] == row.dag_id) &
            (failed["run_date"] == row.run_date)
        ]

        st.dataframe(tasks[["task_id", "failure_signals"]])

        if st.button(
            "Open Investigation",
            key=f"{row.dag_id}_{row.run_date}"
        ):
            st.session_state["selected_dag"] = row.dag_id
            st.session_state["selected_run"] = row.run_date

            st.switch_page("pages/3_Investigation_Workspace.py")