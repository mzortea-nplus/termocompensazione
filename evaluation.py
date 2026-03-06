"""
Legacy evaluation script: load data (with optional query window), train and evaluate
per sensor, write all_metrics.csv. Prefer run_pipeline.py or flows.pipeline_flow for the full pipeline.
"""

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.data import load_data
from src.models import model_evaluation, model_training


def main() -> None:
    cfg = load_config("config/config.yaml")
    full = cfg.to_dict()

    # View + time window when query is set in config
    df, tmp_sensors, sensors = load_data(full)
    df = df.sort_values(by=cfg.data.time_column).reset_index(drop=True)

    dt = df[cfg.data.time_column].diff().dt.total_seconds()
    dt_val = float(dt.mode().iloc[0]) if not dt.mode().empty else float(dt.median())

    alg = cfg.algorithms[0]
    model_params = {
        "lag_time": alg.params.get("lag_time"),
        "max_lag": alg.params.get("max_lag"),
        "dt": dt_val,
    }

    all_metrics = []
    for s in sensors:
        print(f"Training model for {s}")
        df_work = df.copy()
        if df_work[s].isna().any():
            df_work[s] = df_work[s].fillna(df_work[s].mean())

        model = model_training(
            df=df_work,
            tmp_cols=tmp_sensors,
            sig_col=s,
            time_col=cfg.data.time_column,
            model_str=alg.algorithm,
            model_params=model_params,
        )

        metrics, _, _, _ = model_evaluation(
            model=model,
            xdata=df_work[tmp_sensors].to_numpy(),
            ydata=df_work[s].to_numpy(),
            tdata=df_work[cfg.data.time_column].to_numpy(),
            debug_mode=cfg.debug_mode,
            filename=str(Path(cfg.output.plot_dir) / f"debug_{alg.algorithm}.png"),
        )
        metrics["sensor"] = s
        all_metrics.append(metrics)
        print("Metrics", metrics)
        print("--------------------------------")

    all_metrics_df = pd.DataFrame(all_metrics)
    all_metrics_df.to_csv("all_metrics.csv", index=False)
    print("Wrote all_metrics.csv")


if __name__ == "__main__":
    main()
