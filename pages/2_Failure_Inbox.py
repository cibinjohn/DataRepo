import pandas as pd
import streamlit as st
from utils.data_loader import load_data
from utils.filters import apply_date_filter

data = load_data()
failed = apply_date_filter(data["failed_dag_runs"])
task_runs = data["task_runs"]

rca = data["dataops_results"].copy()
rca["dag_run_date"] = pd.to_datetime(rca["dag_run_date"])

st.title("📥 Failure Inbox")

selected_dag = st.session_state.get("selected_dag", "All")
if selected_dag != "All":
    failed = failed[failed["dag_id"] == selected_dag]

# one row per failed DAG run
runs = (
    failed[["dag_id", "run_date"]]
    .drop_duplicates()
    .sort_values(["dag_id", "run_date"])
)

def task_status(dag_id, run_date):
    df = task_runs[
        (task_runs["dag_id"] == dag_id) & (task_runs["run_date"] == run_date)
    ]
    if df.empty:
        return df

    df = df.sort_values(["task_id", "attempt"])

    rows = []
    for task_id, g in df.groupby("task_id"):
        seq = "".join("🟢" if ec == 0 else "🔴" for ec in g["error_code"])
        final_failed = g["error_code"].iloc[-1] != 0   # last attempt still failed
        rows.append({"Task": task_id, "Status": seq, "_failed": final_failed})

    # tasks that ended in failure first, then alphabetical
    return pd.DataFrame(rows).sort_values(
        ["_failed", "Task"], ascending=[False, True]
    )
search = st.text_input("Search DAG")
if search:
    runs = runs[runs["dag_id"].str.contains(search, case=False)]

for _, row in runs.iterrows():
    status_df = task_status(row.dag_id, row.run_date)
    n_failed = int(status_df["_failed"].sum()) if not status_df.empty else 0

    with st.expander(f"{row.dag_id} | {row.run_date}  —  {n_failed} failed"):
        st.dataframe(
            status_df[["Task", "Status"]],
            use_container_width=True,
            hide_index=True,
        )

        has_rca = not rca[
            (rca["dag_id"] == row.dag_id)
            & (rca["dag_run_date"].dt.normalize() == pd.Timestamp(row.run_date).normalize())
            ].empty

        c1, c2, _ = st.columns([2, 2, 6])
        with c1:
            if st.button("Open Investigation", key=f"inv_{row.dag_id}_{row.run_date}"):
                st.session_state["investigation_dag"] = row.dag_id
                st.session_state["investigation_run"] = row.run_date
                st.switch_page("pages/3_Investigation_Workspace.py")
        with c2:
            if st.button(
                    "View RCA Report",
                    key=f"rca_{row.dag_id}_{row.run_date}",
                    disabled=not has_rca,
                    help=None if has_rca else "No RCA report generated for this run yet",
            ):
                st.session_state["rca_dag"] = row.dag_id
                st.session_state["rca_run"] = row.run_date
                st.switch_page("pages/4_RCA_Report.py")

    # with st.expander(f"{row.dag_id} | {row.run_date}  —  {n_failed} failed"):
    #     st.dataframe(
    #             status_df[["Task", "Status"]],
    #             use_container_width=True,
    #             hide_index=True,
    #         )
    #     if st.button("Open Investigation", key=f"{row.dag_id}_{row.run_date}"):
    #         st.session_state["investigation_dag"] = row.dag_id
    #         st.session_state["investigation_run"] = row.run_date
    #         st.switch_page("pages/3_Investigation_Workspace.py")