from src.data import load_data
from src.models import model_training, model_evaluation
import numpy as np
import yaml
import pandas as pd
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.set_experiment("test")

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

_df, tmp_sensors, sensors = load_data(cfg["data"])
_df = _df.sort_values(by="time").reset_index(drop=True)
delta_t = _df[cfg["data"]["time_column"]].diff().dt.total_seconds().mode()[0]

all_metrics = []

for algorithm in cfg["algorithms"]:

    if algorithm["algorithm"] == "MLR":
        df = _df.copy()
    else:
        df = _df[_df[cfg["data"]["time_column"]].dt.hour == algorithm["params"]["hour"]]
    tmp_data = df[tmp_sensors].to_numpy()

    for s in sensors:
        with mlflow.start_run(
            run_name=f"{algorithm['algorithm']}_{s}",
            experiment_id=experiment.experiment_id,
        ):
            mlflow.log_params(algorithm["params"])
            mlflow.log_params({"sensor": s})
            mlflow.log_params({"algorithm": algorithm["algorithm"]})
            print(f"Training model for {s}")
            print("Nans", df[s].isna().sum())
            df[s] = df[s].fillna(df[s].mean())
            params = algorithm["params"]
            params["dt"] = delta_t

            model = model_training(
                df=df,
                tmp_cols=tmp_sensors,
                sig_col=s,
                time_col=cfg["data"]["time_column"],
                model_str=algorithm["algorithm"],
                model_params=params,
            )

            metrics = model_evaluation(
                model=model,
                model_str=algorithm["algorithm"],
                xdata=tmp_data,
                ydata=df[s].to_numpy(),
                tdata=df[cfg["data"]["time_column"]].to_numpy(),
            )
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            metrics["sensor"] = s
            metrics["algorithm"] = algorithm["algorithm"]
            metrics["params"] = params
            all_metrics.append(metrics)

            print("Metrics", metrics)
            print("--------------------------------")


all_metrics_df = pd.DataFrame(all_metrics)
all_metrics_df.to_csv("all_metrics.csv", index=False)
mlflow.end_run()
