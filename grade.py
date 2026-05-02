from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
REQUEST_FIELDS = ["pickup_zone", "dropoff_zone", "requested_at", "passenger_count"]


def run(
    predict_fn,
    input_path: Path,
    output_csv: Path | None,
    sample_n: int | None,
) -> None:
    df = pd.read_parquet(input_path)
    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"Predicting {len(df):,} rows from {input_path.name}...", file=sys.stderr)

    preds = np.empty(len(df), dtype=np.float64)
    records = df[REQUEST_FIELDS].to_dict("records")
    for i, req in enumerate(records):
        preds[i] = predict_fn(req)

    if output_csv is not None:
        if "row_idx" in df.columns:
            row_idx = df["row_idx"].to_numpy()
        else:
            row_idx = np.arange(len(df), dtype=np.int64)
        pd.DataFrame({"row_idx": row_idx, "prediction": preds}).to_csv(output_csv, index=False)
        print(f"Wrote {len(preds):,} predictions to {output_csv}", file=sys.stderr)

    if "duration_seconds" not in df.columns:
        if output_csv is None:
            raise SystemExit(
                "Local grading needs a `duration_seconds` column in the parquet."
            )
        return

    truth = df["duration_seconds"].to_numpy()
    mae = float(np.mean(np.abs(preds - truth)))
    if not np.isfinite(mae):
        raise SystemExit(f"Non-finite MAE ({mae}) — predictions contain NaN/Inf.")
    print(f"MAE: {mae:.1f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="ETA challenge scoring harness")
    parser.add_argument(
        "input_parquet",
        nargs="?",
        help="Input parquet (with output_csv for grader / Docker mode)",
    )
    parser.add_argument(
        "output_csv",
        nargs="?",
        help="Predictions CSV (grader / Docker mode)",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Model artifact path (repo-relative or absolute); sets ETA_MODEL_PATH",
    )
    parser.add_argument(
        "--sample",
        "--train-size",
        dest="sample_n",
        type=int,
        default=50_000,
        help="Local dev: max rows from dev.parquet (default 50000; 0 = all rows)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Local dev: also write predictions CSV to this path",
    )
    args = parser.parse_args()

    if args.model is not None:
        mp = Path(args.model)
        resolved = mp.resolve() if mp.is_absolute() else (REPO_ROOT / mp).resolve()
        os.environ["ETA_MODEL_PATH"] = str(resolved)
        print(f"Using model: {resolved}", file=sys.stderr)

    import predict as predict_mod

    importlib.reload(predict_mod)
    predict_fn = predict_mod.predict

    has_in = args.input_parquet is not None
    has_out = args.output_csv is not None
    if has_in ^ has_out:
        parser.error("Provide both input_parquet and output_csv, or neither for local dev.")

    if has_in and has_out:
        run(predict_fn, Path(args.input_parquet), Path(args.output_csv), None)
        return

    sample_n = None if args.sample_n <= 0 else args.sample_n
    local_csv = Path(args.output) if args.output else None
    run(predict_fn, DATA_DIR / "dev.parquet", local_csv, sample_n)


if __name__ == "__main__":
    main()
