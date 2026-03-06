"""
Run the data analysis pipeline: view + time window -> train -> evaluate -> save results.
Usage:
  python run_pipeline.py [config/config.yaml]
  python run_pipeline.py --aws-access-key-id KEY --aws-secret-access-key SECRET [--aws-region REGION]
"""

import argparse
import sys
from pathlib import Path

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data analysis pipeline.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--aws-access-key-id",
        default=None,
        help="AWS access key ID for S3 (overrides config and env)",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        default=None,
        help="AWS secret access key for S3 (overrides config and env)",
    )
    parser.add_argument(
        "--aws-region",
        default=None,
        help="AWS region for S3 (e.g. eu-west-1)",
    )
    args = parser.parse_args()

    aws_credentials = None
    if args.aws_access_key_id and args.aws_secret_access_key:
        aws_credentials = {
            "access_key_id": args.aws_access_key_id,
            "secret_access_key": args.aws_secret_access_key,
            "region": args.aws_region,
        }

    results = run_pipeline(
        args.config_path,
        save_results=True,
        append_results=True,
        aws_credentials=aws_credentials,
    )
    print(f"Run complete: {len(results)} result(s) written to output.")
    for r in results[:5]:
        print(r)
    if len(results) > 5:
        print("...")


if __name__ == "__main__":
    main()
