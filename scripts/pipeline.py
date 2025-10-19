import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import logging
import os

import numpy as np
import polars as pl
import pandas as pd

from econclust import (
    load_config,
    _prep_features,
    apply_pca_preprocessing,
    auto_k_range,
    scan_kmeans,
    scan_ward,
    optimize_w_weights_by_stability_fast,
    optimize_w_weights_by_stability_ward,
    cluster_from_X,
    write_parquet_fs,
    lake_to_local,
    write_bytes_fs,
    downsample_dataframe,
    temporal_feature_engineering
)
from econclust.viz import plot_k_scan_to_fs, plot_ward_scan_to_fs, plot_dendrogram_to_fs, export_comparison_plots

# Logging & path utils

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_out_paths(lake_root: str, local_root: str, run_name: str | None = None):
    """
    Returns (lake_plots_dir, local_frames_dir, lake_frames_dir, local_frames_dir) with directories created.
    - lake_plots_dir:  <lake_root>/plots[/run_name]
    - local_frames_dir: <local_root>/frames[/run_name]
    - lake_frames_dir:  <lake_root>/frames[/run_name]
    - local_plots_dir:  <local_root>/plots[/run_name]
    """
    lake_root = Path(lake_root).resolve()
    local_root = Path(local_root).resolve()

    lake_plots = ensure_dir(lake_root / "plots")
    lake_frames = ensure_dir(lake_root / "frames")
    local_plots = ensure_dir(local_root / "plots")
    local_frames = ensure_dir(local_root / "frames")

    if run_name:
        lake_plots = ensure_dir(lake_plots / run_name)
        lake_frames = ensure_dir(lake_frames / run_name)
        local_frames = ensure_dir(local_frames / run_name)
        local_plots = ensure_dir(local_plots / run_name)

    return lake_plots, local_frames, lake_frames, local_plots


# Main pipeline

def main():
    ap = argparse.ArgumentParser(description="Unified clustering pipeline")
    ap.add_argument("--config", type=str, default="configs/default.yml")
    ap.add_argument("--algo", choices=["kmeans", "ward", "both"], default="both")
    ap.add_argument("--input", type=str, nargs='+', default=None, help="One or more input parquet files")
    ap.add_argument("--downsample", type=int, default=None, help="Downsample percentage (1-100)")
    ap.add_argument("--features", nargs='+', default=None, help="Feature columns")
    ap.add_argument("--pca", type=float, default=None, help="PCA variance (e.g. 0.95)")
    ap.add_argument("--standardize", type=str, default=None, help="Standardize features (true/false)")
    ap.add_argument("--lake-root", type=str, default=None, help="Lake root directory")
    ap.add_argument("--local-root", type=str, default=None, help="Local output directory")
    ap.add_argument("--k-max", type=int, default=None, help="Max k for scan")
    ap.add_argument("--n-jobs", type=int, default=None, help="Number of jobs")
    ap.add_argument("--silh-cap", type=int, default=None, help="Silhouette sample size cap")
    ap.add_argument("--export-plots", action="store_true", help="Export plots")
    ap.add_argument("--export-frame", action="store_true", help="Export clustered frame")
    ap.add_argument("--use-pca", action="store_true", help="Run PCA and cluster on PCA space (legacy)")
    ap.add_argument("--thresholds", type=str, default="30,40,50,60,70,80,90,100", help="Ward thresholds CSV")
    ap.add_argument("--run-name", type=str, default=None, help="Optional run name (subfolder for outputs)")
    ap.add_argument("--log-level", type=str, default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    ap.add_argument("--sample-size", type=int, default=None, help="Optional sample size for training")
    args = ap.parse_args()

    setup_logging(args.log_level)

    cfg = load_config(args.config)
    paths, P = cfg.paths, cfg.params

    # Override config with CLI args if provided
    input_parquets = args.input if args.input else (
        paths.input_parquet if isinstance(paths.input_parquet, list) else [paths.input_parquet]
    )
    features = args.features if args.features else P.features
    downsample_pct = args.downsample if args.downsample is not None else 100
    pca_variance = args.pca if args.pca is not None else P.pca_variance
    standardize = args.standardize.lower() == "true" if args.standardize is not None else True
    lake_root = args.lake_root if args.lake_root else paths.lake_root
    local_root = args.local_root if args.local_root else paths.local_root
    k_max = args.k_max if args.k_max is not None else P.scan_max_cap
    n_jobs = args.n_jobs if args.n_jobs is not None else P.n_jobs
    silh_cap = args.silh_cap if args.silh_cap is not None else P.silhouette_sample_size

    # Resolve output dirs (create if missing)
    lake_plots_dir, local_frames_dir, lake_frames_dir, local_plots_dir  = resolve_out_paths(
        lake_root, local_root, args.run_name
    )

    # Basic environment info
    logging.info("CWD: %s", os.getcwd())
    logging.info("Config loaded: %s", args.config)
    logging.info("Algo: %s | use_pca: %s | thresholds: %s", args.algo, args.use_pca, args.thresholds)

    # Load
    # Validate and load each file
    dfs = []
    for path_str in input_parquets:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Input parquet not found: {path.resolve()}")
        logging.info("Reading: %s", path.resolve())
        dfs.append(pl.read_parquet(str(path)))

    # Merge all into one DataFrame
    df = pl.concat(dfs, how="vertical")

    # Optional sampling
    if args.sample_size is not None and args.sample_size < df.height:
        logging.info("Sampling %d records from full dataset with proportional stratification on year and secid...", args.sample_size)
    
        for col in ["year", "secid"]:
            if col not in df.columns:
                raise ValueError(f"Stratified sampling requires '{col}' column in the dataset.")
    
        df = df.with_columns([
            pl.col("year").cast(pl.Int32),
            (pl.col("year").cast(pl.Utf8) + "_" + pl.col("secid").cast(pl.Utf8)).alias("year_secid")
        ])
    
        total_rows = df.height
    
        df = (
            df.group_by("year_secid", maintain_order=True)
            .map_groups(lambda group: group.sample(n=int(args.sample_size * group.height / total_rows), seed=42))
        )

    logging.info("Input shape: rows=%d, cols=%d", df.height, len(df.columns))
    
    # Downsample if requested
    if downsample_pct < 100:
        logging.info("Downsampling data to %d%% of rows...", downsample_pct)
        df = downsample_dataframe(df, target_size=downsample_pct)
        logging.info("Downsampled shape: rows=%d, cols=%d", df.height, len(df.columns))

    # Columns check
    feat_cols = list(features)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    # 1) Optional PCA
    if args.use_pca or args.pca is not None:
        
        df, pca, feat_cols = apply_pca_preprocessing(df, features, n_components=pca_variance)
        # When using PCA, features are already standardized for the PCA step; don't re-standardize for X
        standardize_for_X = False
        suffix = "pca"
    else:
        standardize_for_X = standardize
        suffix = "raw"

    X = _prep_features(df, feat_cols, standardize=standardize_for_X)
    logging.info("Feature matrix X: shape=%s, dtype=%s", X.shape, X.dtype)

    # Output base names
    base = f"{paths.output_prefix}_{suffix}"

    # Pre-build file paths
    lake_plot_kmeans = lake_plots_dir / f"{base}_kmeans_scan.png"
    lake_plot_ward = lake_plots_dir / f"{base}_ward_scan.png"

    local_plot_kmeans = local_plots_dir / f"{base}_kmeans_scan.png"
    local_plot_ward = local_plots_dir / f"{base}_ward_scan.png"

    lake_parquet_kmeans = lake_frames_dir / f"{base}_kmeans.parquet"
    lake_parquet_ward = lake_frames_dir / f"{base}_ward.parquet"

    local_parquet_kmeans = local_frames_dir / f"{base}_kmeans.parquet"
    local_parquet_ward = local_frames_dir / f"{base}_ward.parquet"

    summary = []
    
    # KMeans pipeline
    if args.algo in ("kmeans", "both"):
        logging.info("[KMeans] Auto-selecting k-range...")
        k_range = auto_k_range(df, feat_cols, max_cap=k_max)

        logging.info("[KMeans] Scanning k over %s", list(k_range))
        res_km = scan_kmeans(
            df=None,
            feature_cols=None,
            X=X,
            k_range=k_range,
            n_jobs=n_jobs,
            silhouette_sample_size=silh_cap,
            use_minibatch=True,
            mbk_max_iter=P.minibatch_iter,
        )

        logging.info("[KMeans] Optimizing weights via stability...")
        a, b, k_star, info = optimize_w_weights_by_stability_fast(
            ks=list(res_km.ks),
            res=res_km,
            X=X,
            n_init=P.final_kmeans_n_init,
            bootstraps=P.stability_bootstraps,
            sample_frac=P.stability_sample_frac,
            n_jobs_stability=n_jobs,
        )
        k_final = int(k_star)
        logging.info("[KMeans] Final k = %d (mean ARI=%.4f)", k_final, info["best"]["score"])

        # Final fit
        out_km = cluster_from_X(X, df, method="kmeans", n_clusters=k_final, n_init=P.final_kmeans_n_init)

        # one hot encode cluster column
        out_km_encoded = out_km.to_dummies(columns=["cluster"])

        # Save to lake, copy to local
        write_parquet_fs(out_km, str(lake_parquet_kmeans))
        lake_to_local(str(lake_parquet_kmeans), str(local_parquet_kmeans))
        logging.info("Saved clustered frame → lake=%s | local=%s", lake_parquet_kmeans, local_parquet_kmeans)

        # Build encoded file paths
        lake_parquet_kmeans_encoded = str(lake_parquet_kmeans).replace(".parquet", "_encoded.parquet")
        local_parquet_kmeans_encoded = str(local_parquet_kmeans).replace(".parquet", "_encoded.parquet")
        
        # Save encoded frame
        write_parquet_fs(out_km_encoded, lake_parquet_kmeans_encoded)
        lake_to_local(lake_parquet_kmeans_encoded, local_parquet_kmeans_encoded)
        logging.info("Saved encoded clustered frame → lake=%s | local=%s", lake_parquet_kmeans_encoded, local_parquet_kmeans_encoded)

        summary.append(
            {
                "Ticker": paths.output_prefix,
                "Method": "KMeans",
                "Best Parameter": f"k = {k_final}",
                "Number of Clusters": out_km.select(pl.col("cluster").n_unique()).item(),
                "Silhouette Score": res_km.silhouette[0],
                "Calinski-Harabasz Score": res_km.calinski_harabasz[0],
                "Davies-Bouldin Score": res_km.davies_bouldin[0],
                "Wasserstein distance": res_km.wasserstein[0],
                "Stability (ARI)": info["best"]["score"]
            }
        )

        # Plots → lake + local (helper saves both)
        plot_k_scan_to_fs(res_km, str(lake_plot_kmeans), str(local_plot_kmeans))
        logging.info("Saved KMeans scan plot to lake_dir=%s | local_dir=%s", lake_plots_dir, local_plots_dir)

    # ---------------
    # Ward pipeline
    # ---------------
    if args.algo in ("ward", "both"):
        thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
        logging.info("[Ward] Scanning thresholds: %s", thresholds)

        res_w = scan_ward(X, thresholds, sample_size=silh_cap)

        logging.info("[Ward] Optimizing weights via stability...")
        a, b, t_star, info_w = optimize_w_weights_by_stability_ward(
            thresholds=thresholds,
            res=res_w,
            X=X,
            bootstraps=P.stability_bootstraps,
            sample_frac=P.stability_sample_frac,
            n_jobs_stability=n_jobs,
        )
        t_final = float(t_star)
        logging.info("[Ward] Final threshold = %.4f (mean ARI=%.4f)", t_final, info_w["best"]["score"])

        out_w = cluster_from_X(X, df, method="ward", ward_threshold=t_final)

        # one hot encode cluster column
        out_w_encoded = out_w.to_dummies(columns=["cluster"])

        # Save to lake, copy to local
        write_parquet_fs(out_w, str(lake_parquet_ward))
        lake_to_local(str(lake_parquet_ward), str(local_parquet_ward))
        logging.info("Saved clustered frame → lake=%s | local=%s", lake_parquet_ward, local_parquet_ward)

        # Build encoded file paths
        lake_parquet_ward_encoded = str(lake_parquet_ward).replace(".parquet", "_encoded.parquet")
        local_parquet_ward_encoded = str(local_parquet_ward).replace(".parquet", "_encoded.parquet")
        
        # Save encoded frame
        write_parquet_fs(out_w_encoded, lake_parquet_ward_encoded)
        lake_to_local(lake_parquet_ward_encoded, local_parquet_ward_encoded)
        logging.info("Saved encoded clustered frame → lake=%s | local=%s", lake_parquet_ward_encoded, local_parquet_ward_encoded)

        # Plots → lake + local
        plot_ward_scan_to_fs(res_w, str(lake_plot_ward), str(local_plot_ward))
        logging.info("Saved Ward scan plot to lake_dir=%s | local_dir=%s", lake_plots_dir, local_plots_dir)

        summary.append(
            {
                "Ticker": paths.output_prefix,
                "Method": "Ward Linkage",
                "Best Parameter": f"threshold = {t_final:.2f}",
                "Number of Clusters": out_w.select(pl.col("cluster").n_unique()).item(),
                "Silhouette Score": res_w.silhouette[0],
                "Calinski-Harabasz Score": res_w.calinski_harabasz[0],
                "Davies-Bouldin Score": res_w.davies_bouldin[0],
                "Wasserstein distance": res_w.wasserstein[0],
                "Stability (ARI)": info_w["best"]["score"]
            }
        )
        
        # Dendrogram plot
        plot_dendrogram_to_fs(
            X=X,
            threshold=t_final,
            lake_path=str(lake_plot_ward).replace(".png", "_dendrogram.png"),
            local_path=str(local_plot_ward).replace(".png", "_dendrogram.png"),
            show_contracted=True,
            leaf_font_size=10,
        )
        logging.info("Saved Ward dendrogram plot to lake_dir=%s | local_dir=%s", lake_plots_dir, local_plots_dir)

    # Summary
    logging.info("Generating comparison summary and plots...")

    lake_frame_path=str(lake_frames_dir / f"{paths.output_prefix}_{suffix}_summary.csv")
    local_frame_path=str(local_frames_dir / f"{paths.output_prefix}_{suffix}_summary.csv")
    summary_df = pd.DataFrame(summary) 
    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    write_bytes_fs(lake_frame_path, csv_bytes)
    lake_to_local(lake_frame_path, local_frame_path)

    logging.info("Saved comparison summary.")
    
    if args.algo == "both":
        raw_metrics = {
            f"{paths.output_prefix}_KMeans": [
                res_km.silhouette[0],
                res_km.calinski_harabasz[0],
                1 / res_km.davies_bouldin[0],
                1 / res_km.wasserstein[0]
            ],
            f"{paths.output_prefix}_Ward": [
                res_w.silhouette[0],
                res_w.calinski_harabasz[0],
                1 / res_w.davies_bouldin[0],
                1 / res_w.wasserstein[0]
            ]
        }

        export_comparison_plots(
            raw_metrics=raw_metrics,
            clustered_frames = {
                f"{paths.output_prefix}_KMeans": out_km,
                f"{paths.output_prefix}_Ward": out_w
            },
            lake_plot_path=str(lake_plots_dir / f"{paths.output_prefix}_{suffix}_comparison.png"),
            local_plot_path=str(local_plots_dir / f"{paths.output_prefix}_{suffix}_comparison.png"),
        )
        
        logging.info("Saved comparison plots.")

    logging.info("DONE.")


if __name__ == "__main__":
    main()
