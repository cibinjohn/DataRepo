import pandas as pd


def calculate_failure_rate(
    dag_runs,
    dag_id
):

    dag_data = dag_runs[
        dag_runs["dag_id"] == dag_id
    ]

    if len(dag_data) == 0:
        return 0

    failures = (
        dag_data["is_success"] == False
    ).sum()

    return round(
        failures / len(dag_data) * 100,
        2
    )


def get_recurring_failures(
    task_runs,
    dag_id,
    task_id,
    lookback_days=90
):

    df = task_runs[
        (task_runs["dag_id"] == dag_id)
        &
        (task_runs["task_id"] == task_id)
        &
        (task_runs["error_code"] == 1)
    ]

    return len(df)


def retry_success_rate(
    task_runs,
    dag_id,
    task_id
):

    df = task_runs[
        (task_runs["dag_id"] == dag_id)
        &
        (task_runs["task_id"] == task_id)
    ]

    retry_runs = df[
        df["attempt"] > 1
    ]

    if len(retry_runs) == 0:
        return 0

    successes = retry_runs[
        retry_runs[
            "has_task_succeeded_on_retry"
        ]
        == True
    ]

    return round(
        len(successes)
        / len(retry_runs)
        * 100,
        2
    )


def get_most_failing_tasks(
    task_runs,
    top_n=20
):

    failures = task_runs[
        task_runs["error_code"] == 1
    ]

    return (
        failures
        .groupby("task_id")
        .size()
        .reset_index(name="failures")
        .sort_values(
            "failures",
            ascending=False
        )
        .head(top_n)
    )


def classify_failure(
    failure_signals,
    retry_success_rate
):

    signal_text = str(
        failure_signals
    ).lower()

    if "oom" in signal_text:
        return "Infrastructure"

    if "timeout" in signal_text:
        return "Infrastructure"

    if "file not found" in signal_text:
        return "Data Issue"

    if retry_success_rate > 70:
        return "Transient"

    return "Systemic"


def get_task_duration_stats(
    task_runs,
    dag_id,
    task_id
):

    df = task_runs[
        (task_runs["dag_id"] == dag_id)
        &
        (task_runs["task_id"] == task_id)
    ].copy()

    df["start_time"] = pd.to_datetime(
        df["start_time"]
    )

    df["end_time"] = pd.to_datetime(
        df["end_time"]
    )

    df["duration_sec"] = (
        df["end_time"]
        - df["start_time"]
    ).dt.total_seconds()

    return {
        "avg_duration": round(
            df["duration_sec"].mean(),
            2
        ),
        "max_duration": round(
            df["duration_sec"].max(),
            2
        ),
        "min_duration": round(
            df["duration_sec"].min(),
            2
        )
    }


def compare_against_history(
    current_log_summary,
    historical_log_summary
):

    current_words = set(
        str(current_log_summary)
        .lower()
        .split()
    )

    historical_words = set(
        str(historical_log_summary)
        .lower()
        .split()
    )

    added = current_words - historical_words

    removed = historical_words - current_words

    return {
        "added": list(added),
        "removed": list(removed)
    }


def compute_dag_health_score(
    dag_runs,
    dag_id
):

    dag_data = dag_runs[
        dag_runs["dag_id"] == dag_id
    ]

    if len(dag_data) == 0:
        return 100

    success_rate = (
        dag_data["is_success"]
        .mean()
        * 100
    )

    return round(success_rate, 2)