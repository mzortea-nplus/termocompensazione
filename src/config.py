"""
src/config.py
Load and validate the YAML configuration for the data analysis pipeline.
Supports S3 paths (s3://) and local paths; query section for time window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml


@dataclass
class DataConfig:
    """Data source and column configuration."""

    files: List[str]  # parquet/csv paths; may be s3:// or local
    time_column: str = "time"
    tmp_suffix: str = "_t"
    exclude_columns: List[str] = field(
        default_factory=lambda: ["dt", "time", "month", "hour", "datetime"]
    )


@dataclass
class QueryConfig:
    """Time window for SELECT from view. Only one of last_n_days or (start_time, end_time) used."""

    last_n_days: Optional[float] = None  # e.g. 7
    start_time: Optional[str] = None  # ISO or parseable datetime
    end_time: Optional[str] = None


@dataclass
class AlgorithmConfig:
    """Single algorithm entry from config."""

    algorithm: str
    params: dict


@dataclass
class OutputConfig:
    """Output paths and options."""

    show_plots: bool = False
    save_plots: bool = True
    plot_dir: str = "output/plots"
    results_path: str = "output/results.parquet"  # for pipeline results


@dataclass
class AppConfig:
    """Full application configuration."""

    data: DataConfig
    query: QueryConfig
    algorithms: List[AlgorithmConfig]
    output: OutputConfig
    debug_mode: bool = False

    def to_dict(self) -> dict:
        """For code that expects a dict (e.g. data layer)."""
        return {
            "data": {
                "files": self.data.files,
                "time_column": self.data.time_column,
                "tmp_suffix": self.data.tmp_suffix,
                "exclude_columns": self.data.exclude_columns,
            },
            "query": {
                "last_n_days": self.query.last_n_days,
                "start_time": self.query.start_time,
                "end_time": self.query.end_time,
            },
            "algorithms": [
                {"algorithm": a.algorithm, "params": a.params}
                for a in self.algorithms
            ],
            "output": {
                "show_plots": self.output.show_plots,
                "save_plots": self.output.save_plots,
                "plot_dir": self.output.plot_dir,
                "results_path": self.output.results_path,
            },
            "debug_mode": self.debug_mode,
        }


def load_config(path: str | Path = "config/config.yaml") -> AppConfig:
    """Read config file and return validated AppConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    # Data
    d = raw.get("data", {})
    data = DataConfig(
        files=d.get("files", []),
        time_column=d.get("time_column", "time"),
        tmp_suffix=d.get("tmp_suffix", "_t"),
        exclude_columns=d.get("exclude_columns", ["dt", "time", "month", "hour", "datetime"]),
    )
    if not data.files:
        raise ValueError("data.files cannot be empty.")

    # Query (time window)
    q = raw.get("query", {})
    query = QueryConfig(
        last_n_days=q.get("last_n_days"),
        start_time=q.get("start_time"),
        end_time=q.get("end_time"),
    )

    # Algorithms
    alg_list = raw.get("algorithms", [])
    algorithms = [
        AlgorithmConfig(
            algorithm=item.get("algorithm", "MLR"),
            params=item.get("params", {}),
        )
        for item in alg_list
    ]

    # Output
    out = raw.get("output", {})
    output = OutputConfig(
        show_plots=out.get("show_plots", False),
        save_plots=out.get("save_plots", True),
        plot_dir=out.get("plot_dir", "output/plots"),
        results_path=out.get("results_path", "output/results.parquet"),
    )

    return AppConfig(
        data=data,
        query=query,
        algorithms=algorithms,
        output=output,
        debug_mode=raw.get("debug_mode", False),
    )
