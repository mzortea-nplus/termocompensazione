"""
src/pipeline.py
Single entrypoint: load config, DuckDB view + SELECT time window,
train/evaluate per sensor and algorithm, save results to storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import pandas as pd

from src.config import AppConfig, load_config
from src.data import load_data
from src.models import model_evaluation, model_training


RESULTS_SCHEMA = [
    "run_id",
    "run_timestamp",
    "sensor",
    "algorithm",
    "lag_time",
    "max_lag",
    "mse",
    "r2",
    "mae",
    "msle",
]


def _ensure_results_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save_results(results: List[dict], results_path: str, append: bool = True) -> None:
    """Write results to parquet; append to existing file if append=True."""
    _ensure_results_dir(results_path)
    df_new = pd.DataFrame(results)
    for col in RESULTS_SCHEMA:
        if col not in df_new.columns:
            df_new[col] = None
    df_new = df_new[RESULTS_SCHEMA]

    if append and Path(results_path).exists():
        df_existing = pd.read_parquet(results_path)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_parquet(results_path, index=False)


def run_pipeline(
    config_path: str | Path = "config/config.yaml",
    *,
    save_results: bool = True,
    append_results: bool = True,
    aws_credentials: Any = None,
) -> List[dict]:
    """
    Load config, create DuckDB view and SELECT time window, train and evaluate
    for each sensor and algorithm, return and optionally save metrics.

    aws_credentials: optional dict with access_key_id, secret_access_key, region (for S3).
                     Overrides config aws section if provided.

    Returns:
        List of result dicts (one per sensor per algorithm) with run_id, run_timestamp,
        sensor, algorithm, lag_time, max_lag, mse, r2, mae, msle.
    """
    cfg = load_config(config_path)
    run_id = str(uuid.uuid4())
    run_ts = datetime.now(timezone.utc).isoformat()

    if cfg.output.save_plots or cfg.debug_mode:
        _ensure_results_dir(cfg.output.plot_dir)

    # View + time window (no materialization); pass AWS key/secret if provided
    full_cfg = cfg.to_dict()
    df, tmp_sensors, sensors = load_data(
        full_cfg, aws_credentials=aws_credentials
    )
    df = df.sort_values(by=cfg.data.time_column).reset_index(drop=True)

    # Sampling interval for MLR
    dt = df[cfg.data.time_column].diff().dt.total_seconds()
    dt_val = float(dt.mode().iloc[0]) if not dt.mode().empty else float(dt.median())

    all_results: List[dict] = []

    for alg_cfg in cfg.algorithms:
        model_str = alg_cfg.algorithm
        params = alg_cfg.params.copy()
        params["dt"] = dt_val

        for sensor in sensors:
            # Impute NaNs for this sensor (model expects no NaN in y)
            df_work = df.copy()
            if df_work[sensor].isna().any():
                df_work = df_work.copy()
                df_work[sensor] = df_work[sensor].fillna(df_work[sensor].mean())

            model = model_training(
                df=df_work,
                tmp_cols=tmp_sensors,
                sig_col=sensor,
                time_col=cfg.data.time_column,
                model_str=model_str,
                model_params=params,
            )

            metrics, _, _, _ = model_evaluation(
                model=model,
                xdata=df_work[tmp_sensors].to_numpy(),
                ydata=df_work[sensor].to_numpy(),
                tdata=df_work[cfg.data.time_column].to_numpy(),
                debug_mode=cfg.debug_mode,
                filename=str(Path(cfg.output.plot_dir) / f"debug_{model_str}_{sensor}.png"),
            )

            row = {
                "run_id": run_id,
                "run_timestamp": run_ts,
                "sensor": sensor,
                "algorithm": model_str,
                "lag_time": params.get("lag_time"),
                "max_lag": params.get("max_lag"),
                "mse": metrics["mse"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
                "msle": metrics["msle"],
            }
            all_results.append(row)

    if save_results and all_results:
        _save_results(
            all_results,
            cfg.output.results_path,
            append=append_results,
        )

    return all_results
