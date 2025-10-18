#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lazy-sample a Parquet file and upload to PostgreSQL.

Example (Render Postgres):
python ./dashboard_web/lazy_parquet_to_db.py \
  --parquet ./data/path.parquet \
  --postgres-uri "postgresql+psycopg2://user:host@domain/db" \
  --table tablename \
  --fraction 0.05 \
  --if-exists replace \
  --create-index "date,cluster"
"""

from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# Lazy Parquet sampler (Polars)


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
        read pruned columns once, sample within each group
      (single pass; good for medium/large files)
    - Else:
        read pruned columns and sample Bernoulli-like by fraction
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

    # Single pass read (column-pruned)
    df = pl.read_parquet(path, columns=columns)

    # Stratified sampling path
    if stratify_col and stratify_col in df.columns:
        parts: list[pl.DataFrame] = []
        unique_vals = df.select(pl.col(stratify_col)).unique().to_series().to_list()
        for val in unique_vals:
            sub = df.filter(pl.col(stratify_col) == val)
            if sub.height == 0:
                continue
            take_n = max(1, int(round(sub.height * fraction)))
            parts.append(sub.sample(n=take_n, with_replacement=False, seed=seed))
        if parts:
            return pl.concat(parts, how="vertical")
        # fallback
        take_n = max(1, int(round(df.height * fraction)))
        return df.sample(n=take_n, with_replacement=False, seed=seed)

    # Non-stratified path
    if fraction >= 1.0:
        return df
    take_n = max(1, int(round(df.height * fraction)))
    return df.sample(n=take_n, with_replacement=False, seed=seed)



# Postgres upload helpers


import pandas as pd
import numpy as np
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

def make_engine_from_args(
    *,
    postgres_uri: str | None = None,
    user: str | None = None,
    password: str | None = None,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    require_ssl: bool = True,
    echo: bool = False,
) -> Engine:
    q = "keepalives=1&keepalives_idle=30&keepalives_interval=10&keepalives_count=5"
    def add_params(uri: str) -> str:
        sep = "&" if "?" in uri else "?"
        return f"{uri}{sep}{q}"
    if postgres_uri:
        uri = postgres_uri
        if require_ssl and "sslmode=" not in uri:
            uri += ("&" if "?" in uri else "?") + "sslmode=require"
        if "keepalives=" not in uri:
            uri = add_params(uri)
    else:
        if not all([user, password, host, port, database]):
            raise ValueError("Postgres params missing and no --postgres-uri provided.")
        uri = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
        if require_ssl:
            uri += "?sslmode=require"
        uri = add_params(uri)
    return create_engine(uri, pool_pre_ping=True, echo=echo)


def sanitize_pdf(pdf: pd.DataFrame) -> pd.DataFrame:
    # categoricals -> pandas string -> object
    for c in pdf.select_dtypes(include=["category"]).columns:
        pdf[c] = pdf[c].astype("string")
    for c in pdf.select_dtypes(include=["string"]).columns:
        pdf[c] = pdf[c].astype("object")

    # tz-aware datetimes -> UTC -> naive
    for c in pdf.columns:
        if pd.api.types.is_datetime64tz_dtype(pdf[c]):
            pdf[c] = pdf[c].dt.tz_convert("UTC").dt.tz_localize(None)

    # pandas sometimes keeps ms/µs—fine for PG; ensure NaNs/±Inf -> NULL
    pdf = pdf.replace([np.inf, -np.inf], np.nan).where(pd.notna(pdf), None)
    return pdf

def infer_dtype_map(pdf: pd.DataFrame) -> dict:
    """
    Map a few known columns to stable PG types to avoid surprises from pandas' defaults.
    Extend as needed for your schema.
    """
    dtype_map: dict[str, sa.types.TypeEngine] = {}

    # integers
    for c in pdf.select_dtypes(include=["int32", "int64", "Int64"]).columns:
        # year/cluster fit in INTEGER
        if c in ("year", "cluster"):
            dtype_map[c] = sa.Integer()
        else:
            dtype_map[c] = sa.BigInteger() if pdf[c].max() and pd.to_numeric(pdf[c], errors="coerce").max() > 2**31-1 else sa.Integer()

    # floats -> DOUBLE PRECISION
    for c in pdf.select_dtypes(include=["float32", "float64"]).columns:
        dtype_map[c] = sa.Float(precision=53)  # maps to DOUBLE PRECISION

    # dates/timestamps
    for c in pdf.columns:
        if pd.api.types.is_datetime64_any_dtype(pdf[c]):
            dtype_map[c] = sa.DateTime()  # TIMESTAMP WITHOUT TIME ZONE

    # text-like objects
    for c in pdf.select_dtypes(include=["object"]).columns:
        dtype_map[c] = sa.Text()

    return dtype_map

def to_postgres_safe(
    pdf: pd.DataFrame,
    engine: Engine,
    *,
    table: str,
    schema: str | None = None,
    if_exists: str = "replace",     # 'append' or 'replace'
    chunksize: int = 20_000,
    drop_first: bool = False,       # set True to force a clean slate before append
) -> None:
    pdf = sanitize_pdf(pdf)
    dtype_map = infer_dtype_map(pdf)

    try:
        with engine.begin() as conn:
            if drop_first:
                fq = f'"{schema}".\"{table}\"' if schema else f'"{table}"'
                conn.execute(text(f"DROP TABLE IF EXISTS {fq}"))
            pdf.to_sql(
                name=table,
                con=conn,
                schema=schema,
                if_exists=if_exists,      # 'replace' also works, but drop_first+append can be clearer
                index=False,
                method="multi",
                chunksize=chunksize,
                dtype=dtype_map,
            )
    except DBAPIError as e:
        # e.orig is the underlying psycopg2 error with the real cause
        raise RuntimeError(f"Postgres upload failed (DBAPI): {getattr(e, 'orig', e)}") from e
    except Exception as e:
        # Add context (column dtypes) to generic exceptions
        raise RuntimeError(f"Postgres upload failed: {e}. Dtypes={pdf.dtypes.astype(str).to_dict()}") from e


def create_indexes(
    engine: Engine,
    *,
    table: str,
    schema: str | None,
    index_cols: List[str],
) -> None:
    if not index_cols:
        return
    fq = f'"{schema}".\"{table}\"' if schema else f'"{table}"'
    with engine.begin() as conn:
        for col in index_cols:
            col_q = col.replace('"', '')  # naive sanitize
            idx_name = f'idx_{table}_{col_q}'.replace('.', '_')
            sql = text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON {fq} ("{col_q}")')
            conn.execute(sql)


# CLI args

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lazy-sample a Parquet file and upload to PostgreSQL.")
    p.add_argument("--parquet", required=True, help="Path to input .parquet file")
    p.add_argument("--fraction", type=float, default=0.05, help="Sample fraction in (0,1], default 0.05")
    p.add_argument("--columns", type=str, default="", help="Comma-separated columns to keep (optional)")
    p.add_argument("--stratify-col", type=str, default="", help="Optional column name to stratify sampling")

    # Destination
    p.add_argument("--table", required=True, help="Destination table name")
    p.add_argument("--schema", type=str, default=None, help="Optional schema (e.g., public)")
    p.add_argument("--if-exists", choices=["replace", "append"], default="replace", help="Table write mode")
    p.add_argument("--chunksize", type=int, default=50_000, help="Batch size for to_sql(multi)")

    # Indexes
    p.add_argument("--create-index", type=str, default="", help="Comma-separated column list to index after load")

    # Connection: either full URI or parts
    p.add_argument("--postgres-uri", type=str, default="", help="Full SQLAlchemy URI (postgresql+psycopg2://user:pass@host:port/db)")
    p.add_argument("--postgres-user", type=str, default="")
    p.add_argument("--postgres-password", type=str, default="")
    p.add_argument("--postgres-host", type=str, default="")
    p.add_argument("--postgres-port", type=int, default=5432)
    p.add_argument("--postgres-db", type=str, default="")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cols = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    strat = args.stratify_col.strip() or None
    index_cols = [c.strip() for c in args.create_index.split(",") if c.strip()]

    print(f"[INFO] Sampling parquet: {args.parquet}")
    print(f"[INFO] fraction={args.fraction}, stratify_col={strat or 'None'}, columns={cols or 'ALL'}")

    df_polars = lazy_sample_parquet(
        path=args.parquet,
        fraction=max(0.0, min(1.0, args.fraction)),
        columns=cols,
        stratify_col=strat,
    )
    print(f"[INFO] Sampled rows: {df_polars.height:,}; columns: {len(df_polars.columns)}")

    # Convert to pandas (for to_sql)
    pdf = df_polars.to_pandas(use_pyarrow_extension_array=False)

    # Make engine
    engine = make_engine_from_args(
    postgres_uri=args.postgres_uri or None,
    user=args.postgres_user or None,
    password=args.postgres_password or None,
    host=args.postgres_host or None,
    port=args.postgres_port or None,
    database=args.postgres_db or None,
    require_ssl=True,  
    echo=False,         
)
    print(f"[INFO] Uploading to Postgres: table={args.table}, schema={args.schema or '(default)'} if_exists={args.if_exists}")

    to_postgres_safe(
    pdf,
    engine,
    table=args.table,
    schema=args.schema,
    if_exists=args.if_exists,   # 'replace' while testing
    chunksize=2000,             # drop from 50k -> 2k
)
    print("[OK] Postgres upload complete.")

    if index_cols:
        print(f"[INFO] Creating indexes on: {', '.join(index_cols)}")
        create_indexes(engine, table=args.table, schema=args.schema, index_cols=index_cols)
        print("[OK] Index creation complete.")

    # Done
    print("[DONE]")


if __name__ == "__main__":
    main()
