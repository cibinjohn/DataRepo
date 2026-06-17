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

st.title("🏠 Home")

##################################################
# TOP RIGHT FILTER BAR
##################################################

col1, col2, col3 = st.columns([2, 6, 2])


with col1:
    # Safely get the selected DAG from session state, default to "All" if not set yet
    current_dag = st.session_state.get("selected_dag", "All")

    # Use your custom metric box helper or a clean markdown block
    st.markdown(
        f"""
        <div style="
            font-family: sans-serif;
            background-color: #f1f3f5; 
            border-left: 5px solid #007bff; /* Add an accent color strip */
            border-radius: 6px;        
            padding: 12px;             
            margin: 10px 0 24px 0;
        ">
            <p style="font-size: 20px; color: #666; margin: 0;">Selected DAG</p>
            <p style="font-size: 22px; font-weight: bold; margin: 5px 0 0 0; color: #111;">🎯 {current_dag}</p>
        </div>
        """,
        unsafe_allow_html=True
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
##################################################
# FAILURES PER DAG  +  FAILURE TREND  (side by side)
##################################################
left_col, right_col = st.columns([1, 3])

# ---- LEFT (25%): Failures per DAG ----
with left_col:
    st.subheader("📊 Failures per DAG")

    failed_runs = failed.drop_duplicates(subset=["dag_id", "run_date"])
    if failed_runs.empty:
        st.success("No failures in range.")
    else:
        fail_counts = (
            failed_runs.groupby("dag_id")
            .size()
            .reset_index(name="failures")
            .sort_values("failures", ascending=False)
        )
        max_f = fail_counts["failures"].max()

        n = len(fail_counts)
        rows_html = ""
        for i, (_, r) in enumerate(fail_counts.iterrows(), start=1):
            pct = max(16, round(r["failures"] / max_f * 100))  # floor so small bars stay visible
            rows_html += (
                f'<div class="fpd-row" style="position:relative;display:flex;align-items:center;'
                f'gap:11px;padding:11px 14px;'
                f'background:linear-gradient(to right,#e7f1fb {pct}%,#ffffff {pct}%);">'
                f'<span style="display:flex;align-items:center;justify-content:center;width:20px;'
                f'height:20px;border-radius:50%;background:#f1f3f5;color:#868e96;font-size:11px;'
                f'font-weight:600;flex-shrink:0;">{i}</span>'
                f'<span style="flex:1;font-size:14px;color:#212529;white-space:nowrap;'
                f'overflow:hidden;text-overflow:ellipsis;">{r["dag_id"]}</span>'
                f'<span style="font-size:14px;font-weight:700;color:#1971c2;flex-shrink:0;'
                f'min-width:18px;text-align:right;">{r["failures"]}</span>'
                f'</div>'
            )

        st.markdown(
            "<style>"
            ".fpd-card{border:1px solid #ececec;border-radius:12px;overflow:hidden;"
            "background:#fff;box-shadow:0 1px 3px rgba(0,0,0,0.05);font-family:sans-serif;}"
            ".fpd-row+.fpd-row{border-top:1px solid #f1f3f5;}"
            ".fpd-row:hover{filter:brightness(0.985);}"
            "</style>"
            f'<div class="fpd-card">{rows_html}</div>',
            unsafe_allow_html=True,
        )

# ---- RIGHT (75%): Failure Trend ----
with right_col:
    st.subheader("📉 Failure Trend")

    if not dag_runs.empty:
        trend = (
            dag_runs.groupby("run_date")
            .agg(failed_tasks=("num_failed_tasks", "sum"))
            .reset_index()
        )
        fig_trend = px.line(
            trend, x="run_date", y="failed_tasks", title="Failure Trend Over Time"
        )
        fig_trend.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_trend, use_container_width=True)

st.divider()


##################################################
# SUMMARY TABLES
##################################################

c1, c2 = st.columns(2)

with c1:
    st.subheader("Recent Failures")
    st.dataframe(
        failed[['dag_id', 'run_date', 'task_id', 'start_time', 'end_time']].sort_values("run_date", ascending=False).head(10),

        use_container_width=True
    )
# print(dag_runs.columns)
# with c2:
#     st.subheader("Recent DAG Runs")
#     st.dataframe(
#         dag_runs.sort_values("run_date", ascending=False).head(10),
#         use_container_width=True
#     )

with c2:
    st.subheader("Recent DAG Runs")

    recent_runs = (
        dag_runs.sort_values("run_date", ascending=False)
        .head(10)
        .copy()
    )

    recent_runs["Status"] = recent_runs["is_success"].map(
        lambda x: "✅ Success" if x else "❌ Failed"
    )

    st.dataframe(
        recent_runs[
            [
                "dag_id",
                "run_date",
                "num_successful_tasks",
                "num_failed_tasks",
                "Status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )