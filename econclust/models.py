from __future__ import annotations
from typing import Optional
import numpy as np, polars as pl
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import fcluster
from fastcluster import linkage

def cluster_from_X(
    X: np.ndarray,
    df: pl.DataFrame,
    *,
    method: str,                   # "kmeans" | "ward"
    n_clusters: Optional[int] = None,
    n_init: int | str = "auto",
    random_state: int = 42,
    ward_threshold: Optional[float] = None,
) -> pl.DataFrame:
    method = method.lower()
    if method == "kmeans":
        assert n_clusters is not None, "n_clusters required for KMeans."
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init, algorithm="elkan")
        labels = km.fit_predict(X)
        print(f"KMeans inertia: {km.inertia_:.2f}")
        return df.with_columns(pl.Series("cluster", labels))
    elif method == "ward":
        assert ward_threshold is not None, "ward_threshold required for Ward."
        Z = linkage(X, method="ward")
        labels = fcluster(Z, t=ward_threshold, criterion="distance") - 1
        return df.with_columns(pl.Series("cluster", labels))
    else:
        raise ValueError(f"Unknown method: {method}")
