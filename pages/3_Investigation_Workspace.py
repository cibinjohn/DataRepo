import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data
from utils.filters import apply_time_filter

data = load_data()

failed = apply_time_filter(data["failed_dag_runs"])
task_runs = apply_time_filter(data["task_runs"])
rca = data["dataops_results"]

st.title("🔬 Investigation Workspace")

dag_id = st.session_state.get("selected_dag")
run_date = st.session_state.get("selected_run")

if not dag_id or not run_date:
    st.warning("Select a DAG from Failure Inbox")
    st.stop()

# Optional global DAG filter override
global_dag = st.session_state.get("selected_dag", "All")

if global_dag != "All":
    failed = failed[failed["dag_id"] == global_dag]
    task_runs = task_runs[task_runs["dag_id"] == global_dag]
    rca = rca[rca["dag_id"] == global_dag]

left, center, right = st.columns([1, 3, 2])

##################################################
# LEFT
##################################################
with left:

    st.subheader("Navigator")

    dag_tasks = failed[
        (failed["dag_id"] == dag_id) &
        (failed["run_date"] == run_date)
    ]

    if dag_tasks.empty:
        st.warning("No tasks found")
        st.stop()

    selected_task = st.selectbox(
        "Task",
        dag_tasks["task_id"].unique()
    )

    task_attempts = task_runs[
        (task_runs["dag_id"] == dag_id) &
        (task_runs["run_date"] == run_date) &
        (task_runs["task_id"] == selected_task)
    ]

    if task_attempts.empty:
        st.warning("No attempts found")
        st.stop()

    selected_attempt = st.selectbox(
        "Attempt",
        sorted(task_attempts["attempt"].unique())
    )

    st.subheader("Execution Timeline")

    timeline = task_runs[
        (task_runs["dag_id"] == dag_id) &
        (task_runs["run_date"] == run_date)
    ]

    if not timeline.empty:
        fig = px.timeline(
            timeline,
            x_start="start_time",
            x_end="end_time",
            y="task_id",
            color="error_code"
        )
        st.plotly_chart(fig, use_container_width=True)

##################################################
# CENTER
##################################################

selected_run_df = task_attempts[
    task_attempts["attempt"] == selected_attempt
]

if selected_run_df.empty:
    st.stop()

record = selected_run_df.iloc[0]

with center:

    tabs = st.tabs(["Logs", "Stack Trace", "Signals", "Diff"])

    with tabs[0]:
        search = st.text_input("Search Logs")

        log = str(record.get("log", ""))

        if search:
            filtered = "\n".join(
                [x for x in log.split("\n") if search.lower() in x.lower()]
            )
            st.code(filtered)
        else:
            st.code(log)

    with tabs[1]:
        stack_df = failed[
            (failed["dag_id"] == dag_id) &
            (failed["run_date"] == run_date) &
            (failed["task_id"] == selected_task)
        ]

        if not stack_df.empty:
            st.code(stack_df["stack_trace"].iloc[0])

    with tabs[2]:
        signal_df = failed[
            (failed["dag_id"] == dag_id) &
            (failed["run_date"] == run_date) &
            (failed["task_id"] == selected_task)
        ]

        if not signal_df.empty:
            st.json({"signals": signal_df["failure_signals"].iloc[0]})

    with tabs[3]:
        st.info("Compare with last successful run")

        historical = task_runs[
            (task_runs["dag_id"] == dag_id) &
            (task_runs["task_id"] == selected_task) &
            (task_runs["error_code"] == 0)
        ]

        if not historical.empty:
            st.code(historical.iloc[-1]["log"][:5000])