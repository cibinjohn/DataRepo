"""
Generate 1 month of internally-consistent dummy data for the DataOps RCA assistant.

Join graph guaranteed by construction:
  dag_runs (is_success=False)  --(dag_id, run_date)-->  failed_dag_runs
  failed_dag_runs              --(dag_id, run_date)-->  dag_analysis_status (dag_run_date)
  dag_analysis_status(Completed)-->                     dataops_assistant_results
  task_runs is the full superset of every attempt (success + failure).
"""

import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

START_DATE = datetime(2026, 5, 12)        # ~1 month ending recently
N_DAYS = 31
DATES = [START_DATE + timedelta(days=i) for i in range(N_DAYS)]
TODAY = START_DATE + timedelta(days=N_DAYS - 1)   # "now" for analysis backlog logic

# ---------------------------------------------------------------- DAG catalog
DAGS = {
    "daily_sales_etl":        dict(hour=2, fail=0.12, retries=2,
        tasks=["extract_orders", "extract_returns", "transform_sales", "dq_check_sales", "load_fact_sales"]),
    "customer_360_pipeline":  dict(hour=1, fail=0.18, retries=3,
        tasks=["ingest_crm", "ingest_web_events", "resolve_identity", "build_profiles", "publish_features"]),
    "inventory_sync":         dict(hour=3, fail=0.10, retries=2,
        tasks=["pull_warehouse_api", "normalize_sku", "reconcile_stock", "write_delta"]),
    "marketing_attribution":  dict(hour=4, fail=0.15, retries=2,
        tasks=["load_ad_spend", "load_conversions", "join_touchpoints", "attribute_revenue"]),
    "finance_reconciliation": dict(hour=5, fail=0.08, retries=3,
        tasks=["extract_ledger", "extract_payments", "reconcile", "gl_validation", "export_report"]),
    "clickstream_ingest":     dict(hour=0, fail=0.20, retries=2,
        tasks=["read_kafka_offsets", "parse_events", "sessionize", "write_bronze", "write_silver"]),
    "ml_feature_store":       dict(hour=6, fail=0.14, retries=3,
        tasks=["fetch_training_data", "compute_features", "validate_features", "materialize_online"]),
    "data_quality_checks":    dict(hour=7, fail=0.09, retries=1,
        tasks=["load_snapshots", "run_expectations", "summarize_results"]),
    "revenue_reporting":      dict(hour=8, fail=0.07, retries=2,
        tasks=["aggregate_daily_revenue", "currency_convert", "build_dashboards_tbl"]),
    "user_segmentation":      dict(hour=9, fail=0.13, retries=2,
        tasks=["load_activity", "compute_rfm", "assign_segments", "push_to_crm"]),
}

# ------------------------------------------------------------ failure archetypes
FAILURES = {
    "executor_oom": dict(
        signals=["OutOfMemoryError", "ExecutorLostFailure", "Container killed by YARN exceeding memory limits"],
        stack=("java.lang.OutOfMemoryError: Java heap space\n"
               "\tat org.apache.spark.sql.execution.aggregate.HashAggregateExec.doExecute(HashAggregateExec.scala:101)\n"
               "\tat org.apache.spark.sql.execution.exchange.ShuffleExchangeExec.prepareShuffleDependency(...)\n"
               "\tat org.apache.spark.scheduler.Task.run(Task.scala:139)"),
        summary="Executor lost after exceeding container memory limits during a wide shuffle/aggregation; consistent with data skew or under-provisioned executor memory.",
        line="ERROR Executor: Exception in task 47.0 in stage 12.0 (TID 1893) java.lang.OutOfMemoryError: Java heap space",
        transient=True),
    "source_missing": dict(
        signals=["FileNotFoundException", "Path does not exist", "0 records read from source"],
        stack=("org.apache.spark.sql.AnalysisException: Path does not exist: s3://lake/raw/{src}/dt=...\n"
               "\tat org.apache.spark.sql.execution.datasources.DataSource.checkAndGlobPathIfNecessary(...)\n"
               "\tat org.apache.spark.sql.DataFrameReader.load(DataFrameReader.scala:309)"),
        summary="The upstream source partition for this run date was missing/empty, so the task aborted before processing any records.",
        line="ERROR DataSource: Path does not exist for the current run partition; 0 input files matched.",
        transient=False),
    "schema_drift": dict(
        signals=["AnalysisException", "cannot resolve column", "Schema mismatch detected"],
        stack=("org.apache.spark.sql.AnalysisException: cannot resolve 'customer_id' given input columns: [cust_id, ...]\n"
               "\tat org.apache.spark.sql.catalyst.analysis.CheckAnalysis.failAnalysis(...)\n"
               "\tat org.apache.spark.sql.Dataset.select(Dataset.scala:1685)"),
        summary="Incoming schema drifted (a referenced column was renamed/dropped), raising an AnalysisException during the transform step.",
        line="ERROR Analyzer: cannot resolve column 'customer_id' — upstream schema changed.",
        transient=False),
    "uc_permission": dict(
        signals=["PERMISSION_DENIED", "User does not have SELECT on table", "Unauthorized"],
        stack=("com.databricks.sql.managedcatalog.PermissionDeniedException: PERMISSION_DENIED: "
               "User 'sp-dataops-prod' does not have SELECT on Table 'main.curated.{src}'\n"
               "\tat com.databricks.sql.managedcatalog.ManagedCatalogClientImpl.checkPrivilege(...)"),
        summary="The service principal was missing a Unity Catalog grant (SELECT/MODIFY) on a required table; likely a grant regression after a catalog change.",
        line="ERROR UnityCatalog: PERMISSION_DENIED for service principal on a referenced table.",
        transient=False),
    "conn_timeout": dict(
        signals=["SQLTransientConnectionException", "Connection timed out", "Read timed out"],
        stack=("java.sql.SQLTransientConnectionException: HikariPool-1 - Connection is not available, request timed out after 30000ms\n"
               "\tat com.zaxxer.hikari.pool.HikariPool.getConnection(HikariPool.java:213)\n"
               "\tat org.apache.spark.sql.execution.datasources.jdbc.JDBCRDD.compute(...)"),
        summary="Connection to the upstream source database timed out after retries, pointing to a network blip or source unavailability rather than a code defect.",
        line="ERROR JDBCRDD: Connection timed out after 30000ms while reading from source DB.",
        transient=True),
    "dq_violation": dict(
        signals=["ConstraintViolation", "Null values in non-nullable column", "Expectation failed: not_null"],
        stack=("com.databricks.sql.expectations.ExpectationFailedException: Expectation 'not_null(order_id)' failed: 4123 violating rows\n"
               "\tat com.databricks.pipelines.execution.core.ExpectationEvaluator.evaluate(...)"),
        summary="A data-quality expectation failed because a non-nullable key column contained nulls in the latest batch, halting the pipeline by design.",
        line="ERROR Expectations: not_null(order_id) failed with 4123 violating rows.",
        transient=False),
    "shuffle_fetch": dict(
        signals=["FetchFailedException", "Shuffle block fetch failed", "Connection reset by peer"],
        stack=("org.apache.spark.shuffle.FetchFailedException: Failed to connect to executor-7.cluster:35421\n"
               "\tat org.apache.spark.storage.ShuffleBlockFetcherIterator.throwFetchFailedException(...)\n"
               "\tat org.apache.spark.scheduler.ShuffleMapTask.runTask(ShuffleMapTask.scala:99)"),
        summary="A shuffle fetch failed after an executor/node was lost mid-stage; the stage was retried and ultimately exhausted retries.",
        line="ERROR ShuffleBlockFetcherIterator: Failed to fetch shuffle block from a lost executor.",
        transient=True),
    "task_timeout": dict(
        signals=["TimeoutException", "SLA missed", "Task exceeded execution timeout"],
        stack=("java.util.concurrent.TimeoutException: Task exceeded the configured execution_timeout of 3600s\n"
               "\tat org.apache.airflow.executors.TaskRunner.enforceTimeout(...)"),
        summary="The task exceeded its execution timeout, consistent with growing data volume or cluster resource contention during the run window.",
        line="ERROR TaskRunner: Task exceeded execution_timeout (3600s) and was terminated.",
        transient=True),
    "disk_full": dict(
        signals=["No space left on device", "DiskQuotaExceeded", "IOException during spill"],
        stack=("java.io.IOException: No space left on device\n"
               "\tat java.base/sun.nio.ch.FileDispatcherImpl.write0(Native Method)\n"
               "\tat org.apache.spark.storage.DiskBlockObjectWriter.write(...)"),
        summary="The job ran out of local disk while spilling during a heavy shuffle/sort, exhausting node storage.",
        line="ERROR DiskBlockObjectWriter: No space left on device while spilling shuffle data.",
        transient=False),
    "sensor_timeout": dict(
        signals=["UpstreamFailedException", "Sensor timeout", "Dependency partition not found"],
        stack=("airflow.exceptions.AirflowSensorTimeout: Sensor timed out waiting for partition dt=... on upstream dataset\n"
               "\tat airflow.sensors.base.BaseSensorOperator.execute(...)"),
        summary="A sensor timed out waiting for an upstream dataset partition that never landed, blocking all downstream work.",
        line="ERROR Sensor: Timed out waiting for upstream partition to arrive.",
        transient=False),
}
TRANSIENT_KEYS = [k for k, v in FAILURES.items() if v["transient"]]
ALL_KEYS = list(FAILURES.keys())


def fmt(ts):
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def build_log(dag, task, st, et, attempt, outcome, fkey=None):
    """outcome in {'success','fail'}"""
    dur = int((et - st).total_seconds())
    L = [
        f"{fmt(st)} INFO  [{dag}.{task}] TaskInstance started — attempt {attempt}",
        f"{fmt(st + timedelta(seconds=2))} INFO  Acquiring Spark session on job cluster (Photon enabled, DBR 15.4 LTS)",
        f"{fmt(st + timedelta(seconds=5))} INFO  Reading inputs from Unity Catalog main.curated.* ",
    ]
    if outcome == "success":
        L += [
            f"{fmt(st + timedelta(seconds=max(6, dur // 2)))} INFO  Stage 1 complete — input rows processed",
            f"{fmt(et - timedelta(seconds=2))} INFO  Writing output Delta table (MERGE) ",
            f"{fmt(et)} INFO  [{dag}.{task}] Task completed successfully in {dur}s (exit_code=0)",
        ]
    else:
        f = FAILURES[fkey]
        line = f["line"].replace("{src}", task)
        stack = f["stack"].replace("{src}", task)
        L += [
            f"{fmt(st + timedelta(seconds=max(6, dur // 2)))} INFO  Stage 1 running…",
            f"{fmt(et - timedelta(seconds=3))} {line}",
            *(f"{fmt(et - timedelta(seconds=2))} ERROR   {ln}" if i == 0
              else f"                       {ln}"
              for i, ln in enumerate(stack.split('\n'))),
            f"{fmt(et)} ERROR [{dag}.{task}] Task failed on attempt {attempt} (exit_code=1)",
        ]
    return "\n".join(L)


# ---------------------------------------------------------------- simulate
task_runs = []          # every attempt
dag_runs = []           # one row per dag per day
failure_events = []     # one per (failed dag run, failing task) -> drives failed_dag_runs

TRANSIENT_PROB = 0.07   # prob an otherwise-successful task has a transient retry

for d in DATES:
    for dag, cfg in DAGS.items():
        tasks = cfg["tasks"]
        cur = d.replace(hour=cfg["hour"], minute=random.randint(0, 30), second=random.randint(0, 59))
        # day-level noise on failure probability
        p_fail = min(0.95, max(0.0, np.random.normal(cfg["fail"], 0.03)))
        run_fails = random.random() < p_fail
        fail_idx = random.randint(0, len(tasks) - 1) if run_fails else None
        # bias failing archetype toward something plausible for the task name
        chosen_fkey = random.choice(ALL_KEYS) if run_fails else None

        num_success = 0
        is_success = True

        for i, task in enumerate(tasks):
            if run_fails and i > fail_idx:
                break  # downstream skipped after a hard failure

            dur = timedelta(seconds=random.randint(180, 1500))

            if run_fails and i == fail_idx:
                # fails every attempt
                n_att = cfg["retries"] + 1
                for a in range(1, n_att + 1):
                    st = cur
                    et = st + dur
                    task_runs.append(dict(
                        dag_id=dag, run_date=d.date().isoformat(), task_id=task,
                        start_time=fmt(st), end_time=fmt(et),
                        log=build_log(dag, task, st, et, a, "fail", chosen_fkey),
                        attempt=a, has_task_succeeded_on_retry=False, error_code=1))
                    cur = et + timedelta(seconds=random.randint(60, 300))  # retry backoff
                # terminal failed attempt drives failed_dag_runs
                term = task_runs[-1]
                failure_events.append(dict(
                    dag_id=dag, run_date=d.date().isoformat(), task_id=task,
                    start_time=term["start_time"], end_time=term["end_time"],
                    log=term["log"], fkey=chosen_fkey, dt=d))
                is_success = False
                break
            else:
                # ultimately succeeds; maybe a transient retry first
                transient = random.random() < TRANSIENT_PROB
                a = 1
                if transient:
                    tkey = random.choice(TRANSIENT_KEYS)
                    n_fail = random.randint(1, 2)
                    for _ in range(n_fail):
                        st = cur
                        et = st + timedelta(seconds=random.randint(120, 600))
                        task_runs.append(dict(
                            dag_id=dag, run_date=d.date().isoformat(), task_id=task,
                            start_time=fmt(st), end_time=fmt(et),
                            log=build_log(dag, task, st, et, a, "fail", tkey),
                            attempt=a, has_task_succeeded_on_retry=True, error_code=1))
                        cur = et + timedelta(seconds=random.randint(60, 240))
                        a += 1
                st = cur
                et = st + dur
                task_runs.append(dict(
                    dag_id=dag, run_date=d.date().isoformat(), task_id=task,
                    start_time=fmt(st), end_time=fmt(et),
                    log=build_log(dag, task, st, et, a, "success"),
                    attempt=a, has_task_succeeded_on_retry=transient, error_code=0))
                cur = et + timedelta(seconds=random.randint(5, 60))
                num_success += 1

        dag_runs.append(dict(
            dag_id=dag, run_date=d.date().isoformat(),
            num_successful_tasks=num_success,
            num_failed_tasks=0 if is_success else 1,
            is_success=is_success))

# ---------------------------------------------------------------- failed_dag_runs
failed_dag_runs = []
for fe in failure_events:
    f = FAILURES[fe["fkey"]]
    failed_dag_runs.append(dict(
        dag_id=fe["dag_id"], run_date=fe["run_date"], task_id=fe["task_id"],
        start_time=fe["start_time"], end_time=fe["end_time"],
        log=fe["log"], log_summary=f["summary"],
        stack_trace=f["stack"].replace("{src}", fe["task_id"]),
        failure_signals=json.dumps(f["signals"])))

# ---------------------------------------------------------------- dag_analysis_status
# Recent failures (last 2 days) more likely still "Not Started".
analysis_status = []
for fe in failure_events:
    age_days = (TODAY - fe["dt"]).days
    if age_days <= 2:
        status = "Completed" if random.random() < 0.20 else "Not Started"
    elif age_days <= 4:
        status = "Completed" if random.random() < 0.55 else "Not Started"
    else:
        # small steady backlog so the UI always has both states to render
        status = "Completed" if random.random() < 0.88 else "Not Started"
    analysis_run = fe["dt"] + timedelta(days=1, hours=random.randint(0, 6))
    analysis_status.append(dict(
        dag_id=fe["dag_id"], dag_run_date=fe["run_date"],
        run_date=fmt(analysis_run), status=status, _fkey=fe["fkey"], _dt=fe["dt"]))

# ---------------------------------------------------------------- dataops_assistant_results
dag_runs_df = pd.DataFrame(dag_runs)


def agent_version(dt):
    day = (dt - START_DATE).days
    return "v0" if day < 10 else ("v1" if day < 21 else "v2")


def prior_fail_count(dag_id, dt):
    return int(((dag_runs_df.dag_id == dag_id)
                & (~dag_runs_df.is_success)
                & (pd.to_datetime(dag_runs_df.run_date) < pd.Timestamp(dt))).sum())


def make_rca(dag, task, dt, fkey):
    f = FAILURES[fkey]
    prior = prior_fail_count(dag, dt)
    sig = ", ".join(f"`{s}`" for s in f["signals"])
    return (
f"""## Root Cause Analysis — `{dag}`
**DAG run date:** {dt.date().isoformat()}  |  **Failing task:** `{task}`  |  **Generated by:** dataops-assistant {agent_version(dt)}

### Summary
The `{dag}` run failed at task `{task}`. {f['summary']}

### Detected failure signals
{sig}

### Root cause
The terminal attempt surfaced **{f['signals'][0]}**. {f['summary']} Downstream tasks were skipped once `{task}` exhausted its retries.

### Historical context
`{dag}` has failed **{prior}** time(s) earlier in the last 30 days. {'This is a recurring failure mode for this DAG and warrants a durable fix rather than a re-run.' if prior >= 3 else 'This appears to be an isolated/intermittent occurrence in the trailing window.'}

### Recommended actions
1. Inspect the terminal task log for `{task}` and confirm the {f['signals'][0]} signal.
2. {'Re-run the DAG — transient infrastructure cause likely cleared.' if f['transient'] else 'Apply a targeted fix before re-running; a blind retry will likely fail again.'}
3. Add/adjust monitoring on `{task}` to alert on early signs of `{f['signals'][0]}`.
""")


results = []
for a in analysis_status:
    if a["status"] == "Completed":
        # find the failing task for this dag run
        fe = next(x for x in failure_events
                  if x["dag_id"] == a["dag_id"] and x["run_date"] == a["dag_run_date"])
        results.append(dict(
            dag_id=a["dag_id"], dag_run_date=a["dag_run_date"],
            rca_report=make_rca(a["dag_id"], fe["task_id"], a["_dt"], a["_fkey"]),
            agent_version=agent_version(a["_dt"])))

# strip helper cols from analysis_status
for a in analysis_status:
    a.pop("_fkey", None); a.pop("_dt", None)

# ---------------------------------------------------------------- write CSVs
out = "data/outputs"
import os
os.makedirs(out, exist_ok=True)

frames = {
    "failed_dag_runs": pd.DataFrame(failed_dag_runs),
    "dag_analysis_status": pd.DataFrame(analysis_status),
    "dataops_assistant_results": pd.DataFrame(results),
    "dag_runs": pd.DataFrame(dag_runs),
    "task_runs": pd.DataFrame(task_runs),
}
for name, df in frames.items():
    df.to_csv(f"{out}/{name}.csv", index=False)

# ---------------------------------------------------------------- report
print("ROW COUNTS")
for name, df in frames.items():
    print(f"  {name:28s} {len(df):6d} rows   cols={list(df.columns)}")

tot = len(dag_runs_df)
fails = int((~dag_runs_df.is_success).sum())
print(f"\nDAG runs: {tot} total, {fails} failed ({fails/tot:.0%})")
sc = pd.DataFrame(analysis_status).status.value_counts().to_dict()
print("Analysis status:", sc)
print("Completed RCAs:", len(results))
print("Agent versions:", pd.DataFrame(results).agent_version.value_counts().to_dict())
print("Failure types used:",
      pd.Series([fe["fkey"] for fe in failure_events]).value_counts().to_dict())