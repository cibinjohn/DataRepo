import streamlit as st
import pandas as pd
import plotly.express as px
from utils.data_loader import load_data

data = load_data()
failed = data["failed_dag_runs"]
task_runs = data["task_runs"]
rca = data["dataops_results"]

st.title("🔍 Investigation Workspace")

# Drill-down target set by the Failure Inbox button — independent of the global DAG filter
dag_id = st.session_state.get("investigation_dag")
run_date = st.session_state.get("investigation_run")

if not dag_id or not run_date:
    st.info("Select a failed run from the Failure Inbox to begin an investigation.")
    st.stop()

st.caption(f"Investigating  `{dag_id}`  ·  run {run_date}")

# All task runs for this specific DAG run (NOT date-filtered — we want the exact run we clicked)
run_tasks = task_runs[
    (task_runs["dag_id"] == dag_id) & (task_runs["run_date"] == run_date)
].sort_values(["task_id", "attempt"])

if run_tasks.empty:
    st.warning("No task runs found for this DAG run.")
    st.stop()

def circles_for(task_id):
    g = run_tasks[run_tasks["task_id"] == task_id]
    return "".join("🟢" if ec == 0 else "🔴" for ec in g["error_code"])

def final_failed(task_id):
    g = run_tasks[run_tasks["task_id"] == task_id]
    return g["error_code"].iloc[-1] != 0

left, center, right = st.columns([1, 3, 2])

##################################################
# LEFT — Navigator
##################################################
with left:
    st.subheader("Navigator")

    # failed tasks first, each labelled with its retry circles
    task_options = sorted(
        run_tasks["task_id"].unique(),
        key=lambda t: (not final_failed(t), t),
    )
    selected_task = st.selectbox(
        "Task",
        task_options,
        format_func=lambda t: f"{t}  {circles_for(t)}",
    )

    task_attempts = run_tasks[run_tasks["task_id"] == selected_task]
    attempt_outcome = dict(zip(task_attempts["attempt"], task_attempts["error_code"]))
    selected_attempt = st.selectbox(
        "Attempt",
        sorted(task_attempts["attempt"].unique()),
        format_func=lambda a: f"Attempt {a}  {'🟢' if attempt_outcome[a] == 0 else '🔴'}",
    )

    st.subheader("Execution Timeline")
    timeline = run_tasks.copy()
    timeline["start_time"] = pd.to_datetime(timeline["start_time"])
    timeline["end_time"] = pd.to_datetime(timeline["end_time"])
    timeline["Outcome"] = timeline["error_code"].map({0: "Success", 1: "Failed"})

    fig = px.timeline(
        timeline,
        x_start="start_time",
        x_end="end_time",
        y="task_id",
        color="Outcome",
        color_discrete_map={"Success": "#2EA043", "Failed": "#E5484D"},
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=0), height=260)
    st.plotly_chart(fig, use_container_width=True)

##################################################
# CENTER — Detail
##################################################
record = task_attempts[task_attempts["attempt"] == selected_attempt].iloc[0]

# stack trace / signals live in failed_dag_runs, keyed per failed task run
detail = failed[
    (failed["dag_id"] == dag_id)
    & (failed["run_date"] == run_date)
    & (failed["task_id"] == selected_task)
]

with center:
    tabs = st.tabs(["Logs", "Stack Trace", "Signals", "Diff"])

    with tabs[0]:
        search = st.text_input("Search Logs")
        log = str(record.get("log", "") or "")
        if search:
            log = "\n".join(x for x in log.split("\n") if search.lower() in x.lower())
        st.code(log or "No log captured for this attempt.")

    with tabs[1]:
        if not detail.empty and pd.notna(detail["stack_trace"].iloc[0]):
            st.code(detail["stack_trace"].iloc[0])
        else:
            st.success("No stack trace — this task did not fail.")

    with tabs[2]:
        if not detail.empty and pd.notna(detail["failure_signals"].iloc[0]):
            st.json({"signals": detail["failure_signals"].iloc[0]})
        else:
            st.success("No failure signals — this task did not fail.")

    with tabs[3]:
        st.info("Comparing against the last successful run of this task")
        historical = task_runs[
            (task_runs["dag_id"] == dag_id)
            & (task_runs["task_id"] == selected_task)
            & (task_runs["error_code"] == 0)
        ].sort_values("run_date")
        if not historical.empty:
            last_ok = historical.iloc[-1]
            st.caption(f"Last success: {last_ok['run_date']}")
            st.code(str(last_ok["log"])[:5000])
        else:
            st.warning("No successful run found for this task in the available history.")