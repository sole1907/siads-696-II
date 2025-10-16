from __future__ import annotations
from typing import List, Tuple
import math, numpy as np, polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def _prep_features(df: pl.DataFrame, feature_cols: List[str], standardize: bool = True,
                   dtype: np.dtype = np.float32) -> np.ndarray:
    X = (df.select([pl.col(c).cast(pl.Float32).fill_null(0.0) for c in feature_cols])
           .to_numpy().astype(dtype, copy=False))
    if standardize:
        X = StandardScaler(copy=False).fit_transform(X)
    return X

def apply_pca_preprocessing(
    df: pl.DataFrame, numeric_features: List[str], n_components: float = 0.95, standardize: bool = True
) -> Tuple[pl.DataFrame, PCA, List[str]]:
    X = _prep_features(df, numeric_features, standardize=standardize, dtype=np.float32)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=42)
    Z = pca.fit_transform(X).astype(np.float32, copy=False)
    pca_cols = [f"pca_component_{i+1}" for i in range(Z.shape[1])]
    pca_df = pl.DataFrame(Z, schema=pca_cols)
    out = pl.concat([df, pca_df], how="horizontal")
    print(f"PCA reduced {len(numeric_features)} → {len(pca_cols)} comps (explained={pca.explained_variance_ratio_.sum():.3f})")
    return out, pca, pca_cols

def auto_k_range(df: pl.DataFrame, feature_cols: List[str], max_cap: int = 20) -> range:
    n_samples = df.height
    n_features = len(feature_cols)
    upper = int(min(max_cap, max(3, math.sqrt(n_samples/100)), n_features*2))
    upper = max(upper, 4)
    print(f"Auto-selected k_range = range(2, {upper}) (samples={n_samples:,}, features={n_features})")
    return range(2, upper)
