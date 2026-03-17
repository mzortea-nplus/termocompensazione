from src.data import load_data
from src.models import model_training, model_evaluation
import numpy as np
import yaml
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description="Train/evaluate models")
parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
parser.add_argument("--s3-key-id", default=None, help="AWS access key id for DuckDB S3")
parser.add_argument("--s3-secret", default=None, help="AWS secret access key for DuckDB S3")
parser.add_argument("--s3-session-token", default=None, help="AWS session token (optional)")
parser.add_argument("--s3-region", default=None, help="AWS region override (optional)")
args = parser.parse_args()

with open(args.config, "r") as f:
    cfg = yaml.safe_load(f)

# Allow passing S3 credentials externally (CLI), without using credential chain.
data_cfg = cfg.setdefault("data", {})
s3_cfg = data_cfg.setdefault("s3", {})
if args.s3_key_id or args.s3_secret or args.s3_session_token or args.s3_region:
    s3_cfg["provider"] = "config"
if args.s3_key_id:
    s3_cfg["key_id"] = args.s3_key_id
if args.s3_secret:
    s3_cfg["secret"] = args.s3_secret
if args.s3_session_token:
    s3_cfg["session_token"] = args.s3_session_token
if args.s3_region:
    s3_cfg["region"] = args.s3_region

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
        metrics["sensor"] = s
        metrics["algorithm"] = algorithm["algorithm"]
        metrics["params"] = params
        all_metrics.append(metrics)

        print("Metrics", metrics)
        print("--------------------------------")


all_metrics_df = pd.DataFrame(all_metrics)
all_metrics_df.to_csv("all_metrics.csv", index=False)
