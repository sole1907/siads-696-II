from __future__ import annotations
from typing import List
import numpy as np, matplotlib.pyplot as plt
from io import BytesIO
from .scan import KScanResult, ScanResultWard
from .storage import write_bytes_fs, lake_to_local
from scipy.cluster.hierarchy import linkage, dendrogram

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