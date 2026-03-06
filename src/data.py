"""
src/data.py
DuckDB view-only over S3 or local parquet/csv: define a VIEW (merge on time),
then SELECT only the configured time window. No materialization on disk.
Returns a pandas DataFrame for that window plus tmp_sensors and sensors lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import pandas as pd


def _is_s3(path: str) -> bool:
    return path.strip().lower().startswith("s3://")


def _ensure_httpfs(conn: duckdb.DuckDBPyConnection, file_paths: List[str]) -> None:
    if any(_is_s3(p) for p in file_paths):
        conn.execute("INSTALL httpfs")
        conn.execute("LOAD httpfs")


def _set_s3_credentials(
    conn: duckdb.DuckDBPyConnection,
    access_key_id: str,
    secret_access_key: str,
    region: Optional[str] = None,
) -> None:
    """Configure DuckDB to use the given AWS key and secret for S3 (CREATE SECRET)."""
    key_esc = access_key_id.replace("'", "''")
    secret_esc = secret_access_key.replace("'", "''")
    region_val = region or "us-east-1"
    region_esc = region_val.replace("'", "''")
    conn.execute(
        f"CREATE OR REPLACE SECRET s3_creds (TYPE s3, PROVIDER config, "
        f"KEY_ID '{key_esc}', SECRET '{secret_esc}', REGION '{region_esc}')"
    )


def _read_fn_and_path(path: str) -> Tuple[str, str]:
    """Return (read_function, path_for_duckdb). Path may be s3:// or local."""
    p = Path(path)
    # DuckDB read_parquet/read_csv accept s3:// and local; globs work in path
    if path.strip().lower().startswith("s3://"):
        return ("read_parquet", path)
    if p.suffix.lower() == ".parquet":
        return ("read_parquet", str(p.as_posix()))
    return ("read_csv", str(p.as_posix()))


def _escape_sql_string(s: str) -> str:
    return s.replace("'", "''")


def _create_merged_view(
    conn: duckdb.DuckDBPyConnection,
    file_paths: List[str],
    time_column: str,
) -> None:
    """Create VIEW merged_data as the JOIN of all sources on time. No CREATE TABLE."""
    tc = time_column
    if len(file_paths) == 1:
        read_fn, path = _read_fn_and_path(file_paths[0])
        path_esc = _escape_sql_string(path)
        conn.execute(
            f"CREATE VIEW merged_data AS SELECT * FROM {read_fn}('{path_esc}')"
        )
        return

    # Multiple sources: one view per source, then JOIN on time
    views: List[str] = []
    for i, fp in enumerate(file_paths):
        read_fn, path = _read_fn_and_path(fp)
        path_esc = _escape_sql_string(path)
        vname = f"src_{i}"
        conn.execute(f"CREATE VIEW {vname} AS SELECT * FROM {read_fn}('{path_esc}')")
        views.append(vname)

    base = views[0]
    selects = [f"{base}.{tc} AS {tc}", f"{base}.* EXCLUDE ({tc})"]
    joins = []
    for v in views[1:]:
        selects.append(f"{v}.* EXCLUDE ({tc})")
        joins.append(
            f"INNER JOIN {v} ON date_trunc('second', CAST({base}.{tc} AS TIMESTAMP)) = date_trunc('second', CAST({v}.{tc} AS TIMESTAMP))"
        )
    sql = f"SELECT {', '.join(selects)} FROM {base} " + " ".join(joins)
    conn.execute(f"CREATE VIEW merged_data AS {sql}")


def _time_filter_sql(
    time_column: str,
    last_n_days: Union[float, None] = None,
    start_time: Union[str, None] = None,
    end_time: Union[str, None] = None,
) -> str:
    """Return SQL WHERE fragment for time window (no leading WHERE)."""
    if start_time is not None and end_time is not None:
        st = _escape_sql_string(start_time)
        et = _escape_sql_string(end_time)
        return f"CAST({time_column} AS TIMESTAMP) >= CAST('{st}' AS TIMESTAMP) AND CAST({time_column} AS TIMESTAMP) <= CAST('{et}' AS TIMESTAMP)"
    if last_n_days is not None:
        return f"CAST({time_column} AS TIMESTAMP) > (SELECT max(CAST({time_column} AS TIMESTAMP)) FROM merged_data) - INTERVAL '{int(last_n_days)} days'"
    return "1=1"


def load_data(
    cfg: dict,
    query_config: Union[dict, None] = None,
    aws_credentials: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    View-only: create DuckDB VIEW over S3/local paths (merge on time), then
    SELECT only the configured time window. No materialization.

    cfg: data config dict with keys: files, time_column, tmp_suffix, exclude_columns;
         may include "aws" with access_key_id, secret_access_key, region.
    query_config: optional dict with last_n_days, start_time, end_time (overrides cfg.query if present).
    aws_credentials: optional dict with access_key_id, secret_access_key, region (overrides cfg.aws if present).

    Returns:
        df: DataFrame for the time window only
        tmp_sensors: list of temperature feature columns
        sensors: list of target sensor columns
    """
    data_cfg = cfg.get("data", cfg) if isinstance(cfg.get("data"), dict) else cfg
    files = data_cfg.get("files", [])
    tc = data_cfg.get("time_column", "time")
    tmp_suffix = data_cfg.get("tmp_suffix", "_t")
    exclude_columns = data_cfg.get(
        "exclude_columns", ["dt", "time", "month", "hour", "datetime"]
    )

    if not files:
        raise ValueError("data.files cannot be empty.")

    conn = duckdb.connect(database=":memory:")
    _ensure_httpfs(conn, files)

    # S3 credentials: param overrides config
    aws = aws_credentials if aws_credentials is not None else cfg.get("aws") or {}
    if isinstance(aws, dict):
        key = aws.get("access_key_id")
        secret = aws.get("secret_access_key")
        region = aws.get("region")
    else:
        key = secret = region = None
    if key and secret and any(_is_s3(p) for p in files):
        _set_s3_credentials(conn, key, secret, region)

    _create_merged_view(conn, files, tc)

    # Query config: prefer explicit query_config, else cfg["query"]
    q = query_config if query_config is not None else cfg.get("query", {})
    if not isinstance(q, dict):
        q = {}
    last_n_days = q.get("last_n_days")
    start_time = q.get("start_time")
    end_time = q.get("end_time")

    where = _time_filter_sql(
        tc, last_n_days=last_n_days, start_time=start_time, end_time=end_time
    )
    sql = f"SELECT * FROM merged_data WHERE {where} ORDER BY {tc}"
    df = conn.execute(sql).df()
    conn.close()

    if df.empty:
        raise ValueError(
            "Time window returned no rows. Check query.last_n_days or start_time/end_time."
        )

    if tc in df.columns:
        df[tc] = pd.to_datetime(df[tc])
    else:
        raise ValueError(f"Time column '{tc}' not found.")

    tmp_sensors = [c for c in df.columns if c.endswith(tmp_suffix)]
    sensors = [c for c in df.columns if c not in exclude_columns + tmp_sensors]

    if not tmp_sensors:
        raise ValueError(f"No column with suffix '{tmp_suffix}' found.")
    if len(tmp_sensors) > 1:
        raise ValueError("Only one temperature column is supported.")
    if not sensors:
        raise ValueError("No target sensor columns found after filtering.")

    return df, tmp_sensors, sensors
