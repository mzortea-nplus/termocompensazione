# Data analysis pipeline

Pipeline: load data (S3 or local) via a **DuckDB view** (no materialization), **SELECT** only the configured time window, run **MLR** per sensor, and **save results** to parquet for visualization.

## Pipeline stages

1. **View** – DuckDB defines a VIEW over parquet/CSV (S3 or local). No `CREATE TABLE`; data is not materialized on disk.
2. **Query** – Each run executes `SELECT ... FROM view WHERE time IN (window)`. Only that window is streamed into memory.
3. **Model** – Train and evaluate Multiple Linear Regression (MLR) per sensor (see `src/MLR.py`).
4. **Store** – Append results (run_id, timestamp, sensor, algorithm, mse, r2, mae, msle) to `output/results.parquet`.

Data can be updated every 15 minutes; the view sees current S3/local data on each run. Scheduling is optional via Prefect.

## Config

Config files live in the `config/` folder. Edit `config/config.yaml`:

- **data.files** – List of parquet/CSV paths. Use `s3://bucket/prefix/*.parquet` for S3 (requires AWS credentials in env or IAM).
- **query** – Time window: `last_n_days: 7` or `start_time` / `end_time`.
- **algorithms** – List of `algorithm` and `params` (e.g. MLR with `lag_time`, `max_lag`).
- **output.results_path** – Where to append results (default `output/results.parquet`).

## Run

From the project root:

```bash
# One-off run (uses config/config.yaml by default)
python run_pipeline.py
# or with custom config
python run_pipeline.py config/config.s3.yaml
```

## Prefect (optional, 15-minute schedule)

```bash
pip install prefect
# On-demand
python -m flows.pipeline_flow

# Start scheduler (every 15 min)
python -m flows.pipeline_flow --schedule
```

Or use `prefect deploy` / external scheduler to run the flow on a schedule.

## Running with AWS S3

1. **Point config at S3** – In `config/config.yaml` (or a copy), set `data.files` to your S3 paths (globs supported). You can copy `config/config.s3.example.yaml` to `config/config.s3.yaml`, replace `YOUR-BUCKET` and `your-prefix`, then run with `python run_pipeline.py config/config.s3.yaml`.

```yaml
data:
  files:
    - "s3://your-bucket/prefix/csv_est/*.parquet"
    - "s3://your-bucket/prefix/csv_incl/*.parquet"
    - "s3://your-bucket/prefix/csv_spost/*.parquet"
    - "s3://your-bucket/prefix/csv_temp/*.parquet"
  time_column: "time"
  tmp_suffix: "_t"
  exclude_columns: ["dt", "time", "month", "hour", "datetime"]
```

2. **Set AWS credentials** (one of):

   - **Key and secret as parameters** (config or CLI):

     In `config/config.yaml` (or your S3 config), add an `aws` section:

     ```yaml
     aws:
       access_key_id: "YOUR_ACCESS_KEY"
       secret_access_key: "YOUR_SECRET_KEY"
       region: "eu-west-1"
     ```

     Or pass them on the command line (CLI overrides config):

     ```bash
     python run_pipeline.py --aws-access-key-id KEY --aws-secret-access-key SECRET --aws-region eu-west-1
     ```

   - **Environment variables** (e.g. in PowerShell):

     ```powershell
     $env:AWS_ACCESS_KEY_ID = "your-access-key"
     $env:AWS_SECRET_ACCESS_KEY = "your-secret-key"
     $env:AWS_REGION = "eu-west-1"
     python run_pipeline.py
     ```

   - **IAM role** – On EC2/ECS/Lambda with an IAM role, no config or env needed; DuckDB uses the default credential chain.

3. **Run the pipeline** – Same as local:

   ```bash
   python run_pipeline.py
   # or
   python -m flows.pipeline_flow
   ```

   The pipeline will load the httpfs extension, define a view over your S3 paths, and SELECT only the time window (e.g. `query.last_n_days: 7`), so only that portion is streamed from S3.

## Environment

- **S3**: set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_REGION`, or use an IAM role on EC2.
- **Python**: 3.10+ recommended.

## Dependencies

See `requirements.txt`: duckdb, pandas, scikit-learn, pyarrow, pyyaml, prefect.

Install: `pip install -r requirements.txt`

## Results schema

Each run appends rows to the results store with:

| Column         | Description                    |
|----------------|--------------------------------|
| run_id         | Unique run id                 |
| run_timestamp  | ISO timestamp of the run      |
| sensor         | Target sensor column name     |
| algorithm      | e.g. MLR                      |
| lag_time       | Algorithm param               |
| max_lag        | Algorithm param               |
| mse, r2, mae, msle | Metrics                   |

Visualization (out of scope here) can read `output/results.parquet` and filter by `run_timestamp` or `run_id`.
