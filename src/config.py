"""
src/config.py
Carica e valida il file YAML di configurazione.
"""

from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class DataConfig:
    files: List[str]
    time_column: str = "time"
    tmp_suffix: str = "_t"
    exclude_columns: List[str] = field(
        default_factory=lambda: ["dt", "time", "month", "hour"]
    )


@dataclass
class FeaturesConfig:
    input: Optional[str] = None
    target: Optional[str] = None


@dataclass
class AlgorithmsConfig:
    linear_regression: bool = True
    multiple_linear_regression: bool = True


@dataclass
class MLRConfig:
    resample_freq: str = "1H"
    max_lag: int = 10
    window: int = 10


@dataclass
class OutputConfig:
    show_plots: bool = True
    save_plots: bool = False
    plot_dir: str = "output/plots"


@dataclass
class AppConfig:
    data: DataConfig
    features: FeaturesConfig
    algorithms: AlgorithmsConfig
    mlr: list[MLRConfig]
    output: OutputConfig


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Legge il file YAML e restituisce un AppConfig validato."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Data
    d = raw.get("data", {})
    data = DataConfig(
        files=d.get("files", []),
        time_column=d.get("time_column", "time"),
        tmp_suffix=d.get("tmp_suffix", "_t"),
        exclude_columns=d.get("exclude_columns", ["dt", "time", "month", "hour"]),
    )
    if not data.files:
        raise ValueError("'data.files' non può essere vuoto nel config.")

    # Features
    feat_raw = raw.get("features", {})
    features = FeaturesConfig(
        input=feat_raw.get("input") or None,
        target=feat_raw.get("target") or None,
    )

    # Algorithms
    alg_raw = raw.get("algorithms", {})
    algorithms = AlgorithmsConfig(
        linear_regression=alg_raw.get("linear_regression", True),
        multiple_linear_regression=alg_raw.get("multiple_linear_regression", True),
    )

    # MLR params
    mlr_raw = raw.get("mlr", {})
    mlr_configs = []
    for config in mlr_raw:
        mlr_configs.append(
            MLRConfig(
                resample_freq=config.get("resample_freq"),
                max_lag=config.get("max_lag"),
                window=config.get("window"),
            )
        )
    # Output
    out_raw = raw.get("output", {})
    output = OutputConfig(
        show_plots=out_raw.get("show_plots", True),
        save_plots=out_raw.get("save_plots", False),
        plot_dir=out_raw.get("plot_dir", "output/plots"),
    )

    return AppConfig(
        data=data,
        features=features,
        algorithms=algorithms,
        mlr=mlr_configs,
        output=output,
    )
