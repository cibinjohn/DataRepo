from pathlib import Path
import pandas as pd
import streamlit as st

DATA_DIR = Path("data/outputs")

@st.cache_data
def load_data():

    failed_dag_runs = pd.read_csv(
        DATA_DIR / "failed_dag_runs.csv"
    )

    dag_analysis_status = pd.read_csv(
        DATA_DIR / "dag_analysis_status.csv"
    )

    dataops_results = pd.read_csv(
        DATA_DIR / "dataops_assistant_results.csv"
    )

    dag_runs = pd.read_csv(
        DATA_DIR / "dag_runs.csv"
    )

    task_runs = pd.read_csv(
        DATA_DIR / "task_runs.csv"
    )

    dag_runs["run_date"] = pd.to_datetime(dag_runs["run_date"])
    task_runs["run_date"] = pd.to_datetime(task_runs["run_date"])
    failed_dag_runs["run_date"] = pd.to_datetime(failed_dag_runs["run_date"])

    return {
        "failed_dag_runs": failed_dag_runs,
        "dag_analysis_status": dag_analysis_status,
        "dataops_results": dataops_results,
        "dag_runs": dag_runs,
        "task_runs": task_runs
    }