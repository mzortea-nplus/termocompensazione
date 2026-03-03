"""
src/data.py
Legge i file CSV specificati nel config e li carica in un database
DuckDB in-memory. Restituisce un DataFrame pandas pronto per i modelli.
"""

from pathlib import Path
from typing import List, Tuple

import duckdb
import pandas as pd

from src.config import AppConfig

MY_CONST = 2


def load_data(cfg: AppConfig) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Legge i CSV, li unisce in DuckDB in-memory e restituisce:
        df          DataFrame completo con NaN imputati con la media
        tmp_sensors lista colonne temperatura (feature, suffisso _t)
        sensors     lista colonne sensore target
    """
    # ── 1. Connessione in memoria ──────────────────────────────────────────
    conn = duckdb.connect(database=":memory:")

    # ── 2. Carica ogni CSV come vista, poi JOIN su time (come all_static.sql) ─
    tc = cfg.data.time_column
    views: List[str] = []
    for i, file_path in enumerate(cfg.data.files):
        p = Path(file_path)
        view_name = f"csv_{i}"
        read_function = "read_parquet" if p.suffix == ".parquet" else "read_csv"
        try:
            conn.execute(
                f"CREATE VIEW {view_name} AS SELECT * FROM {read_function}('{p.as_posix()}')"
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
    tmp_sensors = [c for c in df.columns if c.endswith(cfg.data.tmp_suffix)]
    sensors = [c for c in df.columns if c not in cfg.data.exclude_columns + tmp_sensors]

    if not tmp_sensors:
        raise ValueError(
            f"Nessuna colonna con suffisso '{cfg.data.tmp_suffix}' trovata."
        )
    if not sensors:
        raise ValueError("Nessuna colonna sensore target trovata dopo il filtraggio.")

    # ── 6. Override da config se specificato ──────────────────────────────
    if cfg.features.input:
        if cfg.features.input not in df.columns:
            raise ValueError(
                f"Feature input '{cfg.features.input}' non trovata nel dataset."
            )
        tmp_sensors = [cfg.features.input]
    if cfg.features.target:
        if cfg.features.target not in df.columns:
            raise ValueError(
                f"Feature target '{cfg.features.target}' non trovata nel dataset."
            )
        sensors = [cfg.features.target]

    # ── 7. Imputa NaN con la media di colonna ─────────────────────────────
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    return df, tmp_sensors, sensors
