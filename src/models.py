"""
src/models.py
Implementa i modelli di regressione configurabili:
- LR  : Linear Regression semplice (X → y, punto per punto)
- MLR : Multiple Linear Regression con lag temporali
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from src.config import AppConfig


# ── Risultato di un modello ────────────────────────────────────────────────────


@dataclass
class ModelResult:
    name: str
    y_pred: np.ndarray
    y_true: np.ndarray
    t: pd.DatetimeIndex  # indice temporale allineato a y_pred/y_true
    rmse: float
    r2: float
    model: list[LinearRegression]  # oggetto sklearn per ispezione
    feature_col: str
    target_col: str
    filename: str

    def summary(self) -> str:
        return (
            f"[{self.name}]  RMSE={self.rmse:.4f}  R²={self.r2:.4f}"
            f"  |  feature={self.feature_col}  target={self.target_col}"
        )

    def sort_by_time(self) -> None:
        # sort by time
        idx = np.argsort(self.t)
        self.y_pred = self.y_pred[idx]
        self.y_true = self.y_true[idx]
        self.t = self.t[idx]


# ── Helper ─────────────────────────────────────────────────────────────────────


def _metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return rmse, r2


# ── MLR con lag ───────────────────────────────────────────────────────────────


def run_mlr(
    df: pd.DataFrame,
    tmp_col: str,
    sig_col: str,
    time_col: str,
    filename: str,
    resample_freq: str = "1H",
    max_lag: int = 10,
) -> ModelResult:
    """
    Multiple Linear Regression con lag temporali.
    Il DataFrame viene ricampionato a `resample_freq`, poi vengono create
    colonne lag 0, 1, …, max_lag per la feature tmp_col.
    """
    # Ricampionamento
    df_h = (
        df[[time_col, tmp_col, sig_col]]
        .set_index(time_col)
        .resample(resample_freq)
        .mean()
    )

    # Matrice di design con lag
    X = np.column_stack(
        [df_h[tmp_col].shift(lag).to_numpy() for lag in range(0, max_lag + 1)]
    )
    y = df_h[sig_col].to_numpy()

    # Rimuovi righe con NaN (generati dai lag)
    valid = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[valid]
    y = y[valid]
    t = df_h.index[valid]
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    rmse, r2 = _metrics(y, y_pred)
    result = ModelResult(
        name=f"MLR (lag {max_lag}, {resample_freq})",
        y_pred=y_pred,
        y_true=y,
        t=t,
        rmse=rmse,
        r2=r2,
        model=model,
        feature_col=tmp_col,
        target_col=sig_col,
        filename=filename,
    )
    result.resample_freq = resample_freq
    print(result.summary())
    return result


# ── Dispatcher ────────────────────────────────────────────────────────────────


def run_models(
    df: pd.DataFrame,
    tmp_sensors: list[str],
    sensors: list[str],
    cfg: AppConfig,
    filename: str,
) -> list[ModelResult]:
    """
    Esegue i modelli abilitati nel config e restituisce la lista dei risultati.
    Usa sempre il primo tmp_sensor e il primo sensor come coppia di default.
    """
    tmp_col = tmp_sensors[0]
    sig_col = sensors[0]
    time_col = cfg.data.time_column

    results: list[ModelResult] = []

    if cfg.algorithms.multiple_linear_regression:
        for mlr in cfg.mlr:
            results.append(
                run_mlr(
                    df,
                    tmp_col,
                    sig_col,
                    time_col,
                    filename,
                    resample_freq=mlr.resample_freq,
                    max_lag=mlr.max_lag,
                )
            )

    return results
