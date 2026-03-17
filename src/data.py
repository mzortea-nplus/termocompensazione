"""
src/data.py
Legge i file CSV specificati nel config e li carica in un database
DuckDB in-memory. Restituisce un DataFrame pandas pronto per i modelli.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import duckdb
import pandas as pd

MY_CONST = 2


def _is_s3_path(path: str) -> bool:
    return path.strip().lower().startswith("s3://")


def _ensure_httpfs(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("INSTALL httpfs")
    conn.execute("LOAD httpfs")


def start_s3_connection(conn: duckdb.DuckDBPyConnection, s3_cfg: dict[str, Any]) -> None:
    """
    Configure DuckDB httpfs for S3 access.

    Explicit credentials only (no credential chain).
    """
    _ensure_httpfs(conn)

    region = s3_cfg.get("region", "eu-central-1")
    key_id = s3_cfg.get("key_id")
    secret = s3_cfg.get("secret")
    session_token = s3_cfg.get("session_token")
    if not key_id or not secret:
        raise ValueError(
            "Missing S3 credentials. Provide data.s3.key_id and data.s3.secret "
            "(or pass them via CLI in evaluation.py)."
        )

    token_sql = f", SESSION_TOKEN '{session_token}'" if session_token else ""
    conn.execute(
        f"""
        CREATE OR REPLACE SECRET s3_secret (
            TYPE s3,
            PROVIDER config,
            KEY_ID '{key_id}',
            SECRET '{secret}'{token_sql},
            REGION '{region}'
        )
        """
    )


def load_data(cfg: dict) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Legge i CSV, li unisce in DuckDB in-memory e restituisce:
        df          DataFrame completo con NaN imputati con la media
        tmp_sensors lista colonne temperatura (feature, suffisso _t)
        sensors     lista colonne sensore target
    """
    # ── 1. Connessione in memoria ──────────────────────────────────────────
    conn = duckdb.connect(database=":memory:")

    # ── 1b. Optional S3 configuration (EC2 friendly) ───────────────────────
    files = cfg.get("files", [])
    if any(_is_s3_path(p) for p in files):
        start_s3_connection(conn, cfg.get("s3", {}))

    # ── 2. Carica ogni CSV come vista, poi JOIN su time (come all_static.sql) ─
    tc = cfg["time_column"]
    views: List[str] = []
    for i, file_path in enumerate(cfg["files"]):
        view_name = f"csv_{i}"
        path_lower = str(file_path).lower()
        # If it's an S3 path, enforce parquet (also supports globs/prefixes without .parquet)
        read_function = (
            "read_parquet"
            if _is_s3_path(str(file_path)) or "s3" in path_lower or ".parquet" in path_lower
            else "read_csv"
        )
        try:
            conn.execute(
                f"CREATE VIEW {view_name} AS SELECT * FROM {read_function}('{file_path}')"
            )
        except Exception as e:
            print(f"Error: {e}")
        views.append(view_name)

    if len(views) == 1:
        join_query = f"SELECT * FROM {views[0]}"
    else:
        # Pipeline: first file = base, rest INNER JOIN on time
        base = views[0]
        selects = [f"{base}.{tc} AS {tc}", f"{base}.* EXCLUDE ({tc})"]
        joins = []
        for v in views[1:]:
            selects.append(f"{v}.* EXCLUDE ({tc})")
            joins.append(
                f"INNER JOIN {v} ON date_trunc('second', CAST({base}.{tc} AS TIMESTAMP)) = date_trunc('second', CAST({v}.{tc} AS TIMESTAMP))"
            )
        join_query = f"SELECT {', '.join(selects)} FROM {base} " + " ".join(joins)

    conn.execute(f"CREATE TABLE all_data AS {join_query}")

    # ── 3. Porta in pandas ─────────────────────────────────────────────────
    df: pd.DataFrame = conn.execute("SELECT * FROM all_data").df()
    conn.close()

    # ── 4. Converti la colonna tempo in datetime ───────────────────────────
    if tc in df.columns:
        df[tc] = pd.to_datetime(df[tc])
    else:
        raise ValueError(f"Colonna tempo '{tc}' non trovata nel CSV.")

    # ── 5. Identifica feature (tmp) e target (sensori) ────────────────────
    tmp_sensors = [c for c in df.columns if c.endswith(cfg["tmp_suffix"])]
    sensors = [c for c in df.columns if c not in cfg["exclude_columns"] + tmp_sensors]

    if not tmp_sensors:
        raise ValueError(f"Nessuna colonna con suffisso '{cfg['tmp_suffix']}' trovata.")
    elif len(tmp_sensors) > 1:
        raise ValueError("Only one temperature column is supported")
    if not sensors:
        raise ValueError("Nessuna colonna sensore target trovata dopo il filtraggio.")

    return df, tmp_sensors, sensors
