from __future__ import annotations
from typing import List, Tuple
import math, numpy as np, polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import polars as pl
from typing import Union
import holidays

def temporal_feature_engineering(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """
    Add temporal features to a Polars DataFrame or LazyFrame.
    """
    df = df.with_columns(
        pl.col(date_col).dt.month().alias(f"{date_col}_month"),
        pl.col(date_col).dt.day().alias(f"{date_col}_day"),
        pl.col(date_col).dt.quarter().alias(f"{date_col}_quarter"),
        pl.col(date_col).dt.weekday().alias(f"{date_col}_weekday"),
    )
    
    us_holidays = holidays.US(years=df[date_col].dt.year().unique().to_list())
    df = df.with_columns([
        pl.col(date_col).is_in(list(us_holidays)).alias(f"{date_col}_is_holiday"),
        (pl.col(date_col) - pl.duration(days=1)).is_in(list(us_holidays)).alias(f"{date_col}_is_holiday_eve"),
    ])
    return df

def _prep_features(df: pl.DataFrame, feature_cols: List[str], standardize: bool = True,
                   dtype: np.dtype = np.float32) -> np.ndarray:
    df = temporal_feature_engineering(df, date_col='date')
    X = (df.select([pl.col(c).cast(pl.Float32).fill_null(0.0) for c in feature_cols])
           .to_numpy().astype(dtype, copy=False))
    if standardize:
        X = StandardScaler(copy=False).fit_transform(X)
    # add temporal features to numpy array X if needed
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

def downsample_dataframe(
    df: pl.DataFrame,
    target_size: int,
    random_state: int = 42
) -> pl.DataFrame:
    """
    Downsample a Polars DataFrame to a target percentage of rows using daily stratification.

    Args:
        df: Polars DataFrame with a 'date' column (Date or Datetime).
        target_size: Percentage of rows to keep within each day (1-100).
        random_state: RNG seed for reproducibility.

    Returns:
        A downsampled DataFrame with the same columns, sampled within each 'date' group.
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

    # Polars 1.x: use `group_by`, not `groupby`
    downsampled = df.group_by("date").map_groups(_sample_group)

    print(f"Downsampled DataFrame from {df.height:,} to {downsampled.height:,} rows.")
    return downsampled
