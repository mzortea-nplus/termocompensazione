"""
src/models.py
Implementa i modelli di regressione configurabili:
- LR  : Linear Regression semplice (X → y, punto per punto)
- MLR : Multiple Linear Regression con lag temporali
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_log_error,
)
from matplotlib import pyplot as plt


# ── MLR con lag ───────────────────────────────────────────────────────────────
def run_mlr(
    df: pd.DataFrame,
    tmp_col: str,
    sig_col: str,
    time_col: str,
    lag_time: str = "1H",
    n_steps: int = 10,
    debug_mode: bool = False,
) -> dict:
    """
    Multiple Linear Regression con lag temporali.
    Il DataFrame viene ricampionato a `lag_time`, poi vengono create
    colonne lag 0, 1, …, n_steps per la feature tmp_col.
    """

    df = df.sort_values(by=time_col).reset_index(drop=True)
    N = len(df)
    # Infer the ModelResultbase sampling frequency in seconds
    dt = df[time_col].diff().dt.total_seconds().mode()[0]

    # Convert lag_time string to seconds
    lag_seconds = pd.Timedelta(lag_time).total_seconds()

    # Calculate the step size (how many rows = one lag_time)
    step = int(lag_seconds / dt)

    # create lagged features
    t = df[time_col].to_numpy()[n_steps * step : N]
    X = np.zeros((N - n_steps * step, n_steps + 1))
    for lag in range(0, n_steps + 1):
        X[:, lag] = df[tmp_col].to_numpy()[lag * step : N - (n_steps - lag) * step]
    y = df[sig_col].to_numpy()[n_steps * step : N]

    n_train = int(0.67 * X.shape[0])
    X_train, X_test = np.split(X, [n_train])
    y_train, y_test = np.split(y, [n_train])
    t_train, t_test = np.split(t, [n_train])

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if debug_mode:
        plt.plot(t_train, y_train, label="Training")
        plt.plot(t_test, y_test, label="True")
        plt.plot(t_test, y_pred, label="Predicted")
        plt.legend()
        plt.savefig(f"debug_{sig_col}.png")
        plt.close()
    return {
        "mse": mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "msle": mean_squared_log_error(y_test, y_pred),
    }
