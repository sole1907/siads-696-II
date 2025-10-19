from __future__ import annotations
from typing import List
import numpy as np, matplotlib.pyplot as plt
from io import BytesIO
from .scan import KScanResult, ScanResultWard
from .storage import write_bytes_fs, lake_to_local
from scipy.cluster.hierarchy import linkage, dendrogram
import os

def _finalize_to_bytes() -> bytes:
    buf = BytesIO(); plt.tight_layout(); plt.savefig(buf, dpi=150, bbox_inches="tight"); plt.close(); buf.seek(0)
    return buf.read()

def plot_k_scan_to_fs(res: KScanResult, lake_path: str, local_path: str) -> None:
    ks = np.array(res.ks)

    def _panel(vals, title, ylabel, key):
        if any(v is not None for v in vals):
            arr = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
            plt.figure(figsize=(7,5)); plt.plot(ks, arr, marker="o")
            plt.title(title); plt.xlabel("k"); plt.ylabel(ylabel); plt.grid(True, alpha=0.3)
            if key in res.best_by: plt.axvline(res.best_by[key], linestyle="--", alpha=0.6)
            data = _finalize_to_bytes(); write_bytes_fs(lake_path.replace(".png", f"_{key}.png"), data)
            lake_to_local(lake_path.replace(".png", f"_{key}.png"), local_path.replace(".png", f"_{key}.png"))

    _panel(res.inertia, "Elbow (Inertia) vs k", "Inertia (lower is better)", "elbow_inertia")
    _panel(res.silhouette, "Silhouette vs k (higher is better)", "Silhouette", "silhouette")
    _panel(res.calinski_harabasz, "Calinski–Harabasz vs k (higher is better)", "Calinski–Harabasz", "calinski_harabasz")
    _panel(res.davies_bouldin, "Davies–Bouldin vs k (lower is better)", "Davies–Bouldin", "davies_bouldin")
    _panel(res.wasserstein, "Wasserstein-1 vs k (lower is better)", "Wasserstein-1", "wasserstein")

def plot_ward_scan_to_fs(res: ScanResultWard, lake_path: str, local_path: str) -> None:
    th = np.array(res.thresholds)

    def _panel(vals, title, ylabel, key):
        if any(v is not None for v in vals):
            arr = np.array([np.nan if v is None else float(v) for v in vals], dtype=float)
            plt.figure(figsize=(7,5)); plt.plot(th, arr, marker="o")
            plt.title(title); plt.xlabel("Distance Threshold"); plt.ylabel(ylabel); plt.grid(True, alpha=0.3)
            if key in res.best_by: plt.axvline(res.best_by[key], linestyle="--", alpha=0.6)
            data = _finalize_to_bytes(); write_bytes_fs(lake_path.replace(".png", f"_{key}.png"), data)
            lake_to_local(lake_path.replace(".png", f"_{key}.png"), local_path.replace(".png", f"_{key}.png"))

    _panel(res.silhouette, "Silhouette vs Threshold", "Silhouette", "silhouette")
    _panel(res.calinski_harabasz, "Calinski–Harabasz vs Threshold", "Calinski–Harabasz", "calinski_harabasz")
    _panel(res.davies_bouldin, "Davies–Bouldin vs Threshold", "Davies–Bouldin", "davies_bouldin")
    _panel(res.wasserstein, "Wasserstein vs Threshold", "Wasserstein", "wasserstein")


def plot_dendrogram_to_fs(
    X: np.ndarray,
    threshold: float,
    lake_path: str,
    local_path: str,
    *,
    sample_size: int = 10000,      # subsample for readability & speed
    max_leaves: int = 30,          # show last p merged clusters
    random_state: int = 42,
    orientation: str = "top",      # "top" | "left" | "right" | "bottom"
    show_contracted: bool = True,
    leaf_font_size: int = 10,
) -> None:
    """
    Ward dendrogram (truncated) → write to data lake and mirror locally.

    - Subsamples X (without replacement) if X is larger than `sample_size`
    - Uses truncate_mode="lastp" to keep the figure interpretable
    - Draws a horizontal line at `threshold`
    """
    rng = np.random.RandomState(random_state)
    if X.shape[0] > sample_size:
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_plot = X[idx]
    else:
        X_plot = X

    # Linkage on the (sub)sample for visualization
    Z = linkage(X_plot, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        truncate_mode="lastp",
        p=max_leaves,
        leaf_rotation=90 if orientation in ("top", "bottom") else 0,
        leaf_font_size=leaf_font_size,
        show_contracted=show_contracted,
        orientation=orientation,
    )
    # Horizontal/vertical threshold line depending on orientation
    if orientation in ("top", "bottom"):
        plt.axhline(y=threshold, color="r", linestyle="--", label=f"Threshold = {threshold:.2f}")
        plt.ylabel("Distance")
        plt.xlabel("Merged clusters")
    else:
        plt.axvline(x=threshold, color="r", linestyle="--", label=f"Threshold = {threshold:.2f}")
        plt.xlabel("Distance")
        plt.ylabel("Merged clusters")

    plt.title("Ward Linkage Dendrogram (truncated)")
    plt.legend(loc="best")
    data = _finalize_to_bytes()

    # Save to lake and mirror locally
    write_bytes_fs(lake_path, data)
    lake_to_local(lake_path, local_path)

def export_comparison_plots(
    raw_metrics: dict,
    clustered_frames: dict,
    lake_plot_path: str,
    local_plot_path: str,
) -> None:
    """
    Generate comparison plots (bar + radar) to lake and local paths.
    Works dynamically for any ticker prefix (e.g., SPY, QQQ, etc.).
    """

    # --- Cluster Size Bar Chart ---
    def convert_sizes(pl_df):
        df = pl_df.to_pandas()
        return df["cluster"].value_counts().sort_index()

    tickers = sorted(set(k.split("_")[0] for k in clustered_frames.keys()))
    cluster_sizes = {}

    for ticker in tickers:
        try:
            km_key = f"{ticker}_KMeans"
            ward_key = f"{ticker}_Ward"
            km_sizes = convert_sizes(clustered_frames[km_key])
            ward_sizes = convert_sizes(clustered_frames[ward_key])
            all_clusters = sorted(set(km_sizes.index).union(ward_sizes.index))
            cluster_sizes[ticker] = {
                "x": np.arange(len(all_clusters)),
                "kmeans": km_sizes.reindex(all_clusters, fill_value=0),
                "ward": ward_sizes.reindex(all_clusters, fill_value=0),
                "labels": all_clusters
            }
        except KeyError:
            continue  # Skip if either clustering method is missing

    fig, axs = plt.subplots(1, len(cluster_sizes), figsize=(7 * len(cluster_sizes), 5), sharey=True)
    if len(cluster_sizes) == 1:
        axs = [axs]  # Ensure iterable if single subplot

    for i, (ticker, sizes) in enumerate(cluster_sizes.items()):
        axs[i].bar(sizes["x"] - 0.2, sizes["kmeans"].values, width=0.4, label="KMeans", color="blue")
        axs[i].bar(sizes["x"] + 0.2, sizes["ward"].values, width=0.4, label="Ward Linkage", color="green")
        axs[i].set_xticks(sizes["x"])
        axs[i].set_xticklabels(sizes["labels"])
        axs[i].set_title(f"{ticker} Cluster Sizes")
        axs[i].set_xlabel("Cluster ID")
        axs[i].set_ylabel("Number of Samples")
        axs[i].legend()

    plt.suptitle("Cluster Size Comparison by Ticker")
    plt.tight_layout()
    bar_bytes = _finalize_to_bytes()
    write_bytes_fs(lake_plot_path.replace(".png", "_bars.png"), bar_bytes)
    lake_to_local(lake_plot_path.replace(".png", "_bars.png"), local_plot_path.replace(".png", "_bars.png"))

    # --- Radar Chart ---
    def normalize_all(metrics_dict):
        all_metrics = np.array(list(metrics_dict.values()))
        transposed = all_metrics.T
        normalized = []
        for metric_values in transposed:
            min_val = np.min(metric_values)
            max_val = np.max(metric_values)
            range_val = max_val - min_val if max_val != min_val else 1.0
            normalized.append([(v - min_val) / range_val for v in metric_values])
        return dict(zip(metrics_dict.keys(), np.array(normalized).T.tolist()))

    def plot_radar(labels, m1, m2, title):
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        m1 += m1[:1]
        m2 += m2[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, m1, label="KMeans", color="blue")
        ax.fill(angles, m1, alpha=0.25, color="blue")
        ax.plot(angles, m2, label="Ward Linkage", color="green")
        ax.fill(angles, m2, alpha=0.25, color="green")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.legend(loc="upper right")
        plt.tight_layout()

    normalized = normalize_all(raw_metrics)
    labels = ["Silhouette", "Calinski-Harabasz", "1 / Davies-Bouldin", "1 / Wasserstein"]

    for ticker in tickers:
        try:
            plot_radar(
                labels,
                normalized[f"{ticker}_KMeans"],
                normalized[f"{ticker}_Ward"],
                f"Clustering Quality Comparison - {ticker}"
            )
            radar_bytes = _finalize_to_bytes()
            write_bytes_fs(lake_plot_path.replace(".png", f"_radar.png"), radar_bytes)
            lake_to_local(
                lake_plot_path.replace(".png", f"_radar.png"),
                local_plot_path.replace(".png", f"_radar.png")
            )
        except KeyError:
            continue  # Skip if metrics are missing
