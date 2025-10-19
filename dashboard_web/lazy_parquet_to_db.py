#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Downsample (daily-stratified) a Parquet file and upload to PostgreSQL.

Example (Render Postgres):
python ./dashboard_web/lazy_parquet_to_db.py \
  --parquet ./data/SPY_pca_kmeans.parquet \
  --postgres-uri "postgresql+psycopg2://USER:PASS@HOST:PORT/DB" \
  --table sampled_regimes \
  --downsample-pct 50 \
  --if-exists replace \
  --create-index "date"
"""

from __future__ import annotations

import argparse
import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError

# Parquet loaders
def lazy_sample_parquet(
    path: str,
    fraction: float = 0.05,
    *,
    columns: Optional[List[str]] = None,
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> pl.DataFrame:
    """(Kept for fallback) Low-memory-ish sampler by fraction; optional stratify_col."""
    schema = pl.read_parquet(path, n_rows=0).schema
    available = list(schema.keys())
    if columns:
        columns = [c for c in columns if c in available]
        if not columns:
            raise ValueError("None of the requested columns are in the file schema.")
    else:
        columns = available

    df = pl.read_parquet(path, columns=columns)

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
        take_n = max(1, int(round(df.height * fraction)))
        return df.sample(n=take_n, with_replacement=False, seed=seed)

    if fraction >= 1.0:
        return df
    take_n = max(1, int(round(df.height * fraction)))
    return df.sample(n=take_n, with_replacement=False, seed=seed)


def downsample_dataframe(
    df: pl.DataFrame,
    target_size: int,
    random_state: int = 42
) -> pl.DataFrame:
    """
    Downsample to a target percent (1–100) of rows **within each 'date' group**.
    Requires a 'date' column (Date/Datetime); else raise.
    """
    if not (1 <= target_size <= 100):
        raise ValueError("target_size must be between 1 and 100")
    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column for stratification.")

    fraction = target_size / 100.0

    def _sample_group(g: pl.DataFrame) -> pl.DataFrame:
        n_rows = g.height
        n_sample = max(1, int(n_rows * fraction))
        return g.sample(n=n_sample, with_replacement=False, shuffle=True, seed=random_state)

    out = df.group_by("date").map_groups(_sample_group)
    print(f"[INFO] Downsampled (by date) from {df.height:,} → {out.height:,} rows "
          f"({target_size}% per-day).")
    return out

# Postgres helpers (engine, schema creation, COPY streaming, indexes)
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

def create_table_like(engine: Engine, table: str, pdf: pd.DataFrame,
                      schema: Optional[str] = None, if_exists: str = "fail") -> None:
    fq = f'"{schema}".\"{table}\"' if schema else f'"{table}"'
    dtype_map = {
        "int64": "BIGINT", "int32": "INTEGER",
        "float64": "DOUBLE PRECISION", "float32": "REAL",
        "object": "TEXT", "bool": "BOOLEAN",
        "datetime64[ns]": "TIMESTAMP", "datetime64[ms]": "TIMESTAMP",
    }
    cols_sql = []
    for c, dt in pdf.dtypes.items():
        pg = dtype_map.get(str(dt), "TEXT")
        cols_sql.append(f'"{c}" {pg}')
    cols_clause = ", ".join(cols_sql)

    with engine.begin() as conn:
        if if_exists == "replace":
            conn.execute(text(f"DROP TABLE IF EXISTS {fq}"))
        if if_exists in ("replace", "fail"):
            conn.execute(text(f"CREATE TABLE {fq} ({cols_clause})"))
from datetime import datetime, timedelta

from datetime import datetime, timedelta

def coerce_and_filter_since(
    df: pl.DataFrame,
    *,
    since_years: int = 5,
    since_date: str | None = None
) -> pl.DataFrame:
    if "date" not in df.columns:
        print("[WARN] 'date' column not found; skipping date filter.")
        return df

    # Coerce to proper datetime or date
    dt_type = df.schema["date"]
    if dt_type not in (pl.Date, pl.Datetime):
        # Try parsing strings/objects to Datetime
        try:
            df = df.with_columns(
                pl.col("date").str.strptime(pl.Datetime, strict=False, exact=False).alias("date")
            )
            dt_type = pl.Datetime
        except Exception:
            # Last resort: attempt a non-strict cast
            df = df.with_columns(pl.col("date").cast(pl.Datetime, strict=False).alias("date"))
            dt_type = pl.Datetime

    # Build Python cutoff (literal)
    if since_date:
        cutoff_py = datetime.fromisoformat(since_date)
    else:
        cutoff_py = datetime.utcnow() - timedelta(days=since_years * 365)

    # If column is Date (not Datetime), compare to date(); else compare to datetime
    if dt_type == pl.Date:
        cutoff_lit = cutoff_py.date()
    else:
        cutoff_lit = cutoff_py

    before = df.height
    df = df.filter(pl.col("date") >= cutoff_lit)
    print(
        f"[INFO] Date filter: kept since {since_date or cutoff_py.date()} — "
        f"{before:,} → {df.height:,} rows."
    )
    return df


def copy_chunks_to_postgres(
    engine: Engine,
    table: str,
    pdf: pd.DataFrame,
    schema: Optional[str] = None,
    chunksize: int = 200_000,
) -> None:
    """Stream upload via COPY FROM STDIN in CSV chunks."""
    fq = f'"{schema}".\"{table}\"' if schema else f'"{table}"'
    cols_list = ", ".join(f'"{c}"' for c in pdf.columns)
    copy_sql = f"COPY {fq} ({cols_list}) FROM STDIN WITH (FORMAT CSV, HEADER TRUE)"

    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        start = 0
        n = len(pdf)
        while start < n:
            chunk = pdf.iloc[start:start + chunksize]
            buf = io.StringIO()
            chunk.to_csv(buf, index=False)
            buf.seek(0)
            cur.copy_expert(copy_sql, buf)
            start += chunksize
        cur.close()
        raw.commit()
    finally:
        try:
            raw.close()
        except Exception:
            pass


def create_indexes(engine: Engine, *, table: str, schema: str | None,
                   index_cols: List[str]) -> None:
    if not index_cols:
        return
    fq = f'"{schema}".\"{table}\"' if schema else f'"{table}"'
    with engine.begin() as conn:
        for col in index_cols:
            col_q = col.replace('"', '')
            idx_name = f'idx_{table}_{col_q}'.replace('.', '_')
            conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON {fq} ("{col_q}")'))


def to_postgres_safe(
    engine: Engine,
    pdf: pd.DataFrame,
    *,
    table: str,
    schema: Optional[str] = None,
    if_exists: str = "replace",         # replace for first load; append later
    use_copy_threshold: int = 500_000,  # COPY for large tables
    copy_chunksize: int = 200_000,
    to_sql_chunksize: int = 50_000,
) -> int:
    """Create/replace target table and upload via COPY (large) or to_sql (smaller)."""
    # tz-aware -> naive UTC; categories -> string
    for c in pdf.columns:
        if isinstance(pdf[c].dtype, pd.DatetimeTZDtype):
            pdf[c] = pdf[c].dt.tz_convert("UTC").dt.tz_localize(None)
        if str(pdf[c].dtype) == "category":
            pdf[c] = pdf[c].astype("string")

    pdf = pdf.replace([np.inf, -np.inf], np.nan).where(pd.notna(pdf), None)
    nrows = len(pdf)

    create_table_like(
        engine, table=table, pdf=pdf.head(0),
        schema=schema, if_exists=("replace" if if_exists == "replace" else "fail"),
    )

    if nrows >= use_copy_threshold:
        copy_chunks_to_postgres(
            engine, table=table, pdf=pdf,
            schema=schema, chunksize=copy_chunksize
        )
    else:
        pdf.to_sql(
            name=table, con=engine, schema=schema,
            if_exists="append", index=False,
            method="multi", chunksize=to_sql_chunksize,
        )
    return nrows

# CLI & main


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downsample (by-date) a Parquet file and upload to PostgreSQL.")
    p.add_argument("--parquet", required=True, help="Path to input .parquet file")
    p.add_argument("--since-years", type=int, default=5,
               help="Keep only rows with date >= today - N years (default: 5). Ignored if --since-date is set.")
    p.add_argument("--since-date", type=str, default="",
               help='Override cutoff date as YYYY-MM-DD (e.g., "2020-01-01").')
    p.add_argument("--downsample-pct", type=int, default=0,
                   help="Percent to keep per 'date' group (1–100). If >0, used instead of --fraction.")
    p.add_argument("--fraction", type=float, default=0.0,
                   help="(Fallback) Sample fraction in (0,1]; ignored if --downsample-pct>0.")
    p.add_argument("--columns", type=str, default="", help="Comma-separated columns to keep (optional)")
    p.add_argument("--stratify-col", type=str, default="", help="(Fallback) Optional column to stratify sampling")

    # Destination
    p.add_argument("--table", required=True, help="Destination table name")
    p.add_argument("--schema", type=str, default=None, help="Optional schema (e.g., public)")
    p.add_argument("--if-exists", choices=["replace", "append"], default="replace", help="Table write mode")

    # Indexes
    p.add_argument("--create-index", type=str, default="", help="Comma-separated columns to index after load")

    # Connection: URI or parts
    p.add_argument("--postgres-uri", type=str, default="")
    p.add_argument("--postgres-user", type=str, default="")
    p.add_argument("--postgres-password", type=str, default="")
    p.add_argument("--postgres-host", type=str, default="")
    p.add_argument("--postgres-port", type=int, default=5432)
    p.add_argument("--postgres-db", type=str, default="")

    # Performance knobs
    p.add_argument("--copy-threshold", type=int, default=500_000, help="Rows ≥ threshold use COPY streaming")
    p.add_argument("--copy-chunksize", type=int, default=200_000, help="Rows per COPY chunk")
    p.add_argument("--to-sql-chunksize", type=int, default=50_000, help="Rows per to_sql batch")

    return p.parse_args()

import re

# Build the target feature list (case-insensitive)
_TARGET_FEATURES_CANON = [
    "impl_volatility",
    "delta",
    "theta",
    "vega",
    "moneyness",
    "volume",
    "open_interest",
    "vol",
    "volume_ma5",
    "prc",
    "price_diff_1d",
    "price_diff_2d",
    "price_diff_3d",
    "price_diff_5d",
    "price_diff_8d",
    "price_diff_34d",
]

def select_limited_columns(schema_cols: list[str]) -> list[str]:
    """
    Limit to the requested features + any PCA components + date + cluster.
    Matching is case-insensitive and tolerant of naming like Impl_Volatility vs impl_volatility.
    PCA columns recognized if they match:
        - pca_component_<n>
        - pca<n>
        - pc<n>
    """
    lower_map = {c.lower(): c for c in schema_cols}  # map lowercase -> original
    selected: list[str] = []

    # 1) core features (case-insensitive)
    for canon in _TARGET_FEATURES_CANON:
        if canon in lower_map:
            selected.append(lower_map[canon])
        else:
            # allow variants with underscores/casing from the user list, e.g., "Impl_Volatility"
            # normalize by removing non-alnum + lowercasing
            norm = canon
            for c in schema_cols:
                if re.sub(r"[^a-z0-9]+", "", c.lower()) == re.sub(r"[^a-z0-9]+", "", canon):
                    selected.append(c)
                    break

    # 2) PCA components
    pca_patterns = [
        re.compile(r"^pca_component_\d+$", re.IGNORECASE),
        re.compile(r"^pca\d+$", re.IGNORECASE),
        re.compile(r"^pc\d+$", re.IGNORECASE),
    ]
    for c in schema_cols:
        lc = c.lower()
        if any(p.search(lc) for p in pca_patterns):
            selected.append(c)

    # 3) always include date and cluster if present
    for must in ("date", "cluster"):
        if must in lower_map:
            selected.append(lower_map[must])

    # preserve input order but drop duplicates
    seen = set()
    ordered = []
    for c in schema_cols:  # keep original schema order
        if c in selected and c not in seen:
            seen.add(c); ordered.append(c)

    return ordered

def main() -> None:
    args = parse_args()

    cols = [c.strip() for c in args.columns.split(",") if c.strip()] if args.columns else None
    strat = args.stratify_col.strip() or None
    index_cols = [c.strip() for c in args.create_index.split(",") if c.strip()]

    # Build engine
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

    # Read Parquet (column-pruned)
    schema_cols = list(pl.read_parquet(args.parquet, n_rows=0).schema.keys())
    read_cols = [c for c in cols or schema_cols if c in schema_cols]
    # Limit columns to: your feature list + any PCA components + date + cluster
    read_cols = select_limited_columns(schema_cols)
    print(f"[INFO] Limiting columns to requested feature set ({len(read_cols)} cols).")

    print(f"[INFO] Reading parquet: {args.parquet}")
    print(f"[INFO] Columns: {read_cols}")

    df_full = pl.read_parquet(args.parquet, columns=read_cols)

    df_full = coerce_and_filter_since(
            df_full,
            since_years=args.since_years,
            since_date=(args.since_date or None),
        )

    # Prefer downsample (by date) if requested and possible
    if args.downsample_pct and args.downsample_pct > 0:
        if "date" in df_full.columns:
            print(f"[INFO] Using daily downsampling: {args.downsample_pct}% per-day")
            df_use = downsample_dataframe(df_full, target_size=args.downsample_pct, random_state=42)
        else:
            print("[WARN] 'date' column not found; falling back to fraction sampler.")
            fraction = args.fraction if 0.0 < args.fraction <= 1.0 else 1.0
            df_use = lazy_sample_parquet(
                path=args.parquet,
                fraction=fraction,
                columns=read_cols,
                stratify_col=strat,
            )
    else:
        # Fallback: fraction-based sampler (or all rows if fraction is 0/invalid)
        fraction = args.fraction if 0.0 < args.fraction <= 1.0 else 1.0
        print(f"[INFO] Using fraction sampler: fraction={fraction}")
        df_use = lazy_sample_parquet(
            path=args.parquet,
            fraction=fraction,
            columns=read_cols,
            stratify_col=strat,
        )

    print(f"[INFO] Rows selected: {df_use.height:,}; columns: {len(df_use.columns)}")

    # Convert to pandas and upload
    pdf = df_use.to_pandas(use_pyarrow_extension_array=False)

    print(f"[INFO] Uploading to Postgres: table={args.table}, schema={args.schema or '(default)'} "
          f"if_exists={args.if_exists}")
    uploaded = to_postgres_safe(
        engine=engine,
        pdf=pdf,
        table=args.table,
        schema=args.schema,
        if_exists=args.if_exists,
        use_copy_threshold=args.copy_threshold,
        copy_chunksize=args.copy_chunksize,
        to_sql_chunksize=args.to_sql_chunksize,
    )
    print(f"[OK] Upload complete. Rows uploaded: {uploaded:,}")

    if index_cols:
        print(f"[INFO] Creating indexes on: {', '.join(index_cols)}")
        create_indexes(engine, table=args.table, schema=args.schema, index_cols=index_cols)
        print("[OK] Index creation complete.")


if __name__ == "__main__":
    main()
