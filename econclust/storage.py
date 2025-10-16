from __future__ import annotations
import os
from io import BytesIO
from typing import Optional
import fsspec
import polars as pl

def fs_open(path: str, mode: str = "wb"):
    # Works with s3://, gs://, abfs://, file://, etc.
    return fsspec.open(path, mode).open()

def write_bytes_fs(path: str, data: bytes) -> None:
    with fs_open(path, "wb") as f:
        f.write(data)

def write_parquet_fs(df: pl.DataFrame, path: str) -> None:
    # Write to bytes buffer then upload
    buf = BytesIO()
    df.write_parquet(buf)
    buf.seek(0)
    write_bytes_fs(path, buf.read())

def lake_to_local(lake_path: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    with fs_open(lake_path, "rb") as src, open(local_path, "wb") as dst:
        dst.write(src.read())
