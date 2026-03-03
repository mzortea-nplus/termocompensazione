from src.config import load_config
from src.data import load_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

cfg = load_config("config.yaml")

df, tmp_sensors, sensors = load_data(cfg)
df["month"] = df["time"].dt.month
df_5 = df[df["time"].dt.hour == 5]

fit_params = []
for s in sensors:
    a, b = np.polyfit(df_5[tmp_sensors[0]], df_5[s], 1)
    fit_params.append(
        {
            "sensor": s,
            "slope": float(a),
            "intercept": float(b),
        }
    )
    for m in df["month"].unique():
        df_m = df_5[df_5["month"] == m]
        plt.scatter(df_m[tmp_sensors[0]], df_m[s], label=f"Month {m}")
    plt.plot(
        df_5[tmp_sensors[0]], a * df_5[tmp_sensors[0]] + b, color="red", label="Trend"
    )
    plt.legend()
    plt.close()
fit_params_df = pd.DataFrame(fit_params)
print(fit_params_df)
fit_params_df.to_csv("fit_params.csv", index=False)
