import mlflow

from src.config import load_config
from src.data import load_data
from src.models import model_training, model_prediction, model_evaluation
import numpy as np

cfg = load_config("config.yaml")  # Load base configuration

compute_steps_allowed = lambda lag, freq: int(
    round(float(lag.replace("h", "")) / float(freq.replace("h", "")))
)

for lag_h in ["3h", "12h", "48h"]:
    for freq in ["0.25h", "0.5h", "1h"]:
        steps_allowed = compute_steps_allowed(lag_h, freq)
        print(f"MLR Test {lag_h} {freq} {steps_allowed}")

        df, tmp_sensors, sensors = load_data(cfg)
        df = df.sort_values(by="time").reset_index(drop=True)
        

        for s in sensors:
            print(f"Training model for {s}")
            model_params = cfg.model.__dict__.copy()
            model_params["dt"] = df["time"].diff().dt.total_seconds().mode()[0]
            model = model_training(
                df=df,
                tmp_cols=tmp_sensors,
                sig_col=s,
                time_col=cfg.data.time_column,
                model_str="MLR",
                model_params=model_params,
            )

            scores, predictions, t_data, y_data = model_evaluation(
                model=model,
                xdata=df[tmp_sensors].to_numpy(),
                ydata=df[s].to_numpy(),
                tdata=df["time"].to_numpy(),
                debug_mode=cfg.debug.debug_mode,
                model_params={"max_lag_n": model.features_builder.max_lag_n},
            )

            for key, value in scores.items():
                print(f"{key}: {value}")
            print("--------------------------------")
