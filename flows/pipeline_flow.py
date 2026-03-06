"""
Prefect flow: view + SELECT time window -> train -> evaluate -> save.
Run on-demand (test) or with 15-minute schedule via --schedule.
"""

import sys
from pathlib import Path

from prefect import flow

SRC_DIR = Path(__file__).parent.parent
sys.path.append(str(SRC_DIR))

from src.pipeline import run_pipeline


@flow(
    name="data-analysis-pipeline", description="Load from view, train MLR, save results"
)
def data_analysis_pipeline(
    config_path: str = "config/config.yaml",
    save_results: bool = True,
    append_results: bool = True,
):
    """Run the full pipeline: DuckDB view + time window -> train -> evaluate -> save."""
    return run_pipeline(
        config_path=Path(config_path),
        save_results=save_results,
        append_results=append_results,
    )


if __name__ == "__main__":
    # On-demand (test) run:
    #   python flows/pipeline_flow.py
    #   python -m flows.pipeline_flow
    # Scheduled (every 15 min), Prefect 3:
    #   python flows/pipeline_flow.py --schedule
    import sys

    if "--schedule" in sys.argv:
        data_analysis_pipeline.serve(
            name="pipeline-15min",
            interval=900,  # seconds; Prefect 3
        )
    else:
        data_analysis_pipeline()
