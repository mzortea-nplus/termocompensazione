"""
src/models.py
Implementa i modelli di regressione configurabili:
- LR  : Linear Regression semplice (X → y, punto per punto)
- MLR : Multiple Linear Regression con lag temporali
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_squared_log_error,
)
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# MLR columns builder
class MLRFeaturesBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, lag_time: str, max_lag: str, dt: float):
        # TODO: add validation for lag_time, max_lag, dt

        # lag times as string
        self.lag_time = lag_time
        self.max_lag = max_lag
        # lag times in seconds
        self.lag_seconds = pd.Timedelta(lag_time).total_seconds()
        self.max_lag_seconds = pd.Timedelta(max_lag).total_seconds()
        # sampling time in seconds
        self.dt = dt
        # lag steps
        self.lag_n = int(self.lag_seconds / self.dt)
        self.max_lag_n = int(self.max_lag_seconds / self.dt)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.step = int(self.lag_seconds / self.dt)
        if X.shape[1] > 1:
            raise ValueError("Only one temperature column is supported")
        else:
            tmp = X[:, 0]
            arrs = []
            for lag in range(0, self.max_lag_n + 1):
                arrs.append(
                    tmp[
                        lag * self.step : X.shape[0]
                        - (self.max_lag_n - lag) * self.step
                    ]
                )
            arrs = np.array(arrs)
            return arrs.transpose()


def model_training(
    df: pd.DataFrame,
    tmp_cols: list[str],
    sig_col: str,
    time_col: str,
    model_str: str,
    model_params: dict,
) -> dict:
    """
    WARNING
    Since MLR include lagged features, data will be truncated and reshaped.
    All of this is already built-in in the code: features are reshaped by the features builder,
    while time and temperature are truncated within this function.
    """

    if len(tmp_cols) > 1:
        raise ValueError("Only one temperature column is supported")
    else:
        tmp_col = tmp_cols[0]

    if model_str == "MLR":
        features_builder = MLRFeaturesBuilder(
            lag_time=model_params["lag_time"],
            max_lag=model_params["max_lag"],
            dt=model_params["dt"],
        )
    else:
        raise ValueError(f"Model {model_str} not supported")

    training_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("features", features_builder),
            ("regressor", LinearRegression()),
        ]
    )

    xdata = df[tmp_col].to_numpy().reshape(-1, 1)
    ydata = df[sig_col].to_numpy()[features_builder.max_lag_n :]

    training_pipeline.fit(xdata, ydata)
    return training_pipeline


def model_prediction(
    model: Pipeline,
    xdata: np.ndarray,
) -> np.ndarray:
    """
    WARNING
    Since MLR include lagged features, data will be truncated and reshaped.
    This is already built-in in the code: features are reshaped by the features builder,
    while time and temperature must be handled externally.
    """
    return model.predict(xdata)


def model_evaluation(
    model: Pipeline,
    xdata: np.ndarray,
    ydata: np.ndarray,
    tdata: np.ndarray,
    debug_mode: bool = False,
    model_params: dict = None,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the model performance.
    """

    predictions = model_prediction(model, xdata)

    if debug_mode:
        plt.plot(t_data, y_data, label="True")
        plt.plot(t_data, predictions, label="Predicted")
        plt.legend()
        plt.savefig(f"debug_{model.name}.png")
        plt.close()

    metrics = {
        "mse": mean_squared_error(y_data, predictions),
        "r2": r2_score(y_data, predictions),
        "mae": mean_absolute_error(y_data, predictions),
        "msle": mean_squared_log_error(y_data, predictions),
    }

    return metrics, predictions, t_data, y_data
