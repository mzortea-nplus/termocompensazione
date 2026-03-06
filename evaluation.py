from src.data import load_data
from src.models import model_training, model_evaluation
import numpy as np
import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

df, tmp_sensors, sensors = load_data(cfg["data"])
df = df.sort_values(by="time").reset_index(drop=True)

lag_time = cfg["algorithms"][0]["params"]["lag_time"]
max_lag = cfg["algorithms"][0]["params"]["max_lag"]
dt = df["time"].diff().dt.total_seconds().mode()[0]

for s in sensors:
    print(f"Training model for {s}")
    model_params = {
        "lag_time": lag_time,
        "max_lag": max_lag,
        "dt": dt,
    }
    print("Nans", df[s].isna().sum())
    df[s] = df[s].fillna(df[s].mean())
    model = model_training(
        df=df,
        tmp_cols=tmp_sensors,
        sig_col=s,
        time_col=cfg["data"]["time_column"],
        model_str=cfg["algorithms"][0]["algorithm"],
        model_params=model_params,
    )

    metrics, predictions, t_data, y_data = model_evaluation(
        model=model,
        xdata=df[tmp_sensors].to_numpy(),
        ydata=df[s].to_numpy(),
        tdata=df["time"].to_numpy(),
        debug_mode=cfg["debug_mode"] if "debug_mode" in cfg else False,
        filename=f"debug_{cfg['algorithms'][0]['algorithm']}.png",
    )

    print("Metrics", metrics)
    print("--------------------------------")
