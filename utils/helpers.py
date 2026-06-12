from datetime import datetime
import pandas as pd


def format_duration(start_time, end_time):
    """
    Returns duration in seconds.
    """

    start = pd.to_datetime(start_time)
    end = pd.to_datetime(end_time)

    return (end - start).total_seconds()


def get_task_status(error_code):
    return "Failed" if error_code == 1 else "Success"


def safe_text(value):

    if pd.isna(value):
        return ""

    return str(value)


def create_failure_key(dag_id, run_date):

    return f"{dag_id}__{run_date}"


def parse_failure_signals(signal_text):

    if not signal_text:
        return []

    return [
        s.strip()
        for s in str(signal_text).split(",")
    ]


def get_latest_successful_run(
    task_runs,
    dag_id,
    task_id,
    before_date
):

    df = task_runs[
        (task_runs["dag_id"] == dag_id)
        &
        (task_runs["task_id"] == task_id)
        &
        (task_runs["error_code"] == 0)
    ]

    df["run_date"] = pd.to_datetime(
        df["run_date"]
    )

    before_date = pd.to_datetime(
        before_date
    )

    df = df[
        df["run_date"] < before_date
    ]

    if len(df) == 0:
        return None

    return df.sort_values(
        "run_date"
    ).iloc[-1]