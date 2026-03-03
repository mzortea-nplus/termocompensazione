import mlflow

from src.config import load_config
from src.data import load_data
from src.models import run_mlr
import numpy as np

mlflow.set_experiment("MLflow Quickstart")
mlflow.sklearn.autolog()
cfg = load_config("config.yaml")  # Load base configuration

compute_steps_allowed = lambda lag, freq: int(
    round(float(lag.replace("h", "")) / float(freq.replace("h", "")))
)

for lag_h in ["3h", "12h", "48h"]:
    for freq in ["0.25h", "0.5h", "1h"]:
        steps_allowed = compute_steps_allowed(lag_h, freq)
        print(f"MLR Test {lag_h} {freq} {steps_allowed}")
        with mlflow.start_run(run_name=f"MLR Test {lag_h} {freq} {steps_allowed}"):
            cfg.mlr.n_steps = steps_allowed
            cfg.mlr.lag_time = freq
            mlflow.log_param("n_steps", cfg.mlr.n_steps)
            mlflow.log_param("lag_time", cfg.mlr.lag_time)

            df, tmp_sensors, sensors = load_data(cfg)
            df = df.sort_values(by="time").reset_index(drop=True)

            for s in sensors:
                result = run_mlr(
                    df=df,
                    tmp_col=tmp_sensors[0],
                    sig_col=s,
                    time_col=cfg.data.time_column,
                    lag_time=cfg.mlr.lag_time,
                    n_steps=cfg.mlr.n_steps,
                    debug_mode=cfg.debug.debug_mode,
                )

                print(f"Sensor: {s}")
                for key, value in result.items():
                    print(f"{key}: {value}")
                    mlflow.log_metric(f"{s}_{key}", value)
