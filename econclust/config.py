from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import os, yaml

@dataclass
class Paths:
    lake_root: str
    local_root: str
    input_parquet: str
    output_prefix: str  # e.g. "spy" or "qqq"

@dataclass
class Params:
    features: List[str]
    date_col: Optional[str] = None
    pca_variance: float = 0.95
    scan_max_cap: int = 20
    n_jobs: int = 8
    silhouette_sample_size: int = 5000
    minibatch_iter: int = 100
    stability_bootstraps: int = 6
    stability_sample_frac: float = 0.8
    final_kmeans_n_init: int = 20

@dataclass
class Config:
    paths: Paths
    params: Params

def load_config(path: str | None) -> Config:
    # Allow override with env vars for lake/local roots
    if path and os.path.exists(path):
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    else:
        raw = {}

    paths = raw.get("paths", {})
    params = raw.get("params", {})

    lake_root = os.environ.get("LAKE_ROOT", paths.get("lake_root", "s3://my-bucket/your-prefix"))
    local_root = os.environ.get("LOCAL_ROOT", paths.get("local_root", "./artifacts"))
    input_parquet = paths.get("input_parquet", "s3://my-bucket/your-prefix/input.parquet")
    output_prefix = paths.get("output_prefix", "asset")

    features = params.get("features", [])
    date_col = params.get("date_col", None)

    return Config(
        paths=Paths(
            lake_root=lake_root,
            local_root=local_root,
            input_parquet=input_parquet,
            output_prefix=output_prefix,
        ),
        params=Params(
            features=features,
            date_col=date_col,
            pca_variance=float(params.get("pca_variance", 0.95)),
            scan_max_cap=int(params.get("scan_max_cap", 20)),
            n_jobs=int(params.get("n_jobs", 8)),
            silhouette_sample_size=int(params.get("silhouette_sample_size", 5000)),
            minibatch_iter=int(params.get("minibatch_iter", 100)),
            stability_bootstraps=int(params.get("stability_bootstraps", 6)),
            stability_sample_frac=float(params.get("stability_sample_frac", 0.8)),
            final_kmeans_n_init=int(params.get("final_kmeans_n_init", 20)),
        ),
    )
