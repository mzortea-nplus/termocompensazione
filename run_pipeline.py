"""
Run the data analysis pipeline: view + time window -> train -> evaluate -> save results.
Usage: python run_pipeline.py [config/config.yaml]
"""

import sys
from pathlib import Path

from src.pipeline import run_pipeline


def main() -> None:
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    results = run_pipeline(config_path, save_results=True, append_results=True)
    print(f"Run complete: {len(results)} result(s) written to output.")
    for r in results[:5]:
        print(r)
    if len(results) > 5:
        print("...")


if __name__ == "__main__":
    main()
