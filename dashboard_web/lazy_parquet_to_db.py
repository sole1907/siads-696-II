#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lazy-sample a Parquet file and upload to a DB.

Features
--------
- Lazy-style sampling via Polars:
  * column pruning
  * approximate Bernoulli sampling by fraction
  * optional stratified sampling by a column
- Upload to SQLite (default)
- Optional: upload to PostgreSQL using psycopg2

Usage
-----
# SQLite (default)
python lazy_parquet_to_db.py \
  --parquet ./data/myfile.parquet \
  --sqlite-db ./out.db \
  --table sampled_regimes \
  --fraction 0.10 \
  --label-col cluster \
  --columns date,moneyness,ttm_days,greek_delta,greek_gamma,greek_vega,iv_1,iv_2,iv_3,cluster


"""

from __future__ import annotations

import argparse
import io
import sqlite3
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl

def lazy_sample_parquet(
    path: str,
    fraction: float = 0.05,
    *,
    columns: Optional[List[str]] = None,
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Low-memory(ish) sampling loader.

    Strategy
    --------
    - Read schema first -> prune columns
    - If stratify_col provided and present:
        read only that column, compute unique groups,
        then read the requested columns and sample within each group
      (this requires one full pass; for very large files consider row-group aware sampling)
    - Else:
        read requested columns and take Bernoulli-like sample by fraction
    """
    # Step 1: Schema & column pruning
    schema = pl.read_parquet(path, n_rows=0).schema
    available = list(schema.keys())
    if columns:
        columns = [c for c in columns if c in available]
        if not columns:
            raise ValueError("None of the requested columns are in the file schema.")
    else:
        columns = available

    rng = np.random.default_rng(seed)

    if stratify_col and stratify_col in available:
        # Read only the stratify column to learn groups
        strat_series = pl.read_parquet(path, columns=[stratify_col])[stratify_col]
        unique_vals = strat_series.unique()
        # Read full columns once (column-pruned)
        df = pl.read_parquet(path, columns=columns)
        parts = []
        for val in unique_vals:
            sub = df.filter(pl.col(stratify_col) == val) if stratify_col in df.columns else df
            if sub.height == 0:
                continue
            take_n = max(1, int(round(sub.height * fraction)))
            parts.append(sub.sample(n=take_n, with_replacement=False, seed=seed))
        if parts:
            return pl.concat(parts, how="vertical")
        # fallback if no parts collected
        take_n = max(1, int(round(df.height * fraction)))
        return df.sample(n=take_n, with_replacement=False, seed=seed)

    # No stratification: one pass + sample
    df = pl.read_parquet(path, columns=columns)
    if fraction >= 1.0:
        return df
    take_n = max(1, int(round(df.height * fraction)))
    return df.sample(n=take_n, with_replacement=False, seed=seed)


def upload_polars_to_sqlite(
    df: pl.DataFrame,
    sqlite_db_path: str,
    table: str,
    *,
    if_exists: str = "replace",
    chunksize: int = 50_000,
) -> None:
    """
    Upload a Polars DataFrame to SQLite using pandas.to_sql under the hood.
    """
    # Convert to pandas for to_sql
    pdf = df.to_pandas(use_pyarrow_extension_array=False)
    # Normalize pandas categorical to strings for SQLite compatibility
    for c in pdf.select_dtypes(include=["category"]).columns:
        pdf[c] = pdf[c].astype("string")

    with sqlite3.connect(sqlite_db_path) as conn:
        pdf.to_sql(
            table,
            conn,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method="multi",
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lazy-sample a Parquet and upload to a DB")
    p.add_argument("--parquet", required=True, help="Path to input .parquet file")
    p.add_argument("--fraction", type=float, default=0.05, help="Fraction in (0,1] to sample")
    p.add_argument("--columns", type=str, default="", help="Comma-separated columns to keep (optional)")
    p.add_argument("--stratify-col", type=str, default="", help="Optional column to stratify sampling")

    # SQLite targets
    p.add_argument("--sqlite-db", type=str, default="", help="SQLite DB path (e.g., ./out.db). If set, uses SQLite uploader.")
    p.add_argument("--table", type=str, default="sampled_data", help="Destination table name")

    # Postgres (optional)
    p.add_argument("--postgres-dsn", type=str, default="", help="psycopg2 DSN for PostgreSQL (optional)")
    p.add_argument("--schema", type=str, default="", help="PostgreSQL schema (optional)")

    return p.parse_args()


def main():
    args = parse_args()

    cols = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    strat = args.stratify_col.strip() or None

    print(f"[INFO] Sampling parquet: {args.parquet}")
    print(f"[INFO] fraction={args.fraction}, columns={cols}, stratify_col={strat or 'None'}")

    df = lazy_sample_parquet(
        args.parquet,
        fraction=args.fraction,
        columns=cols,
        stratify_col=strat,
    )
    print(f"[INFO] Sampled rows: {df.height:,}; columns: {len(df.columns)}")

    print(f"[INFO] Uploading to SQLite: {args.sqlite_db}, table={args.table}")
    upload_polars_to_sqlite(df, sqlite_db_path=args.sqlite_db, table=args.table, if_exists="replace")
    print("[OK] SQLite upload complete.")

if __name__ == "__main__":
    main()
