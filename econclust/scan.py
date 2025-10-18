from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable, Dict, Tuple
import numpy as np, pandas as pd, polars as pl
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score
from scipy.cluster.hierarchy import fcluster
from fastcluster import linkage
from scipy.stats import wasserstein_distance
from .features import _prep_features
from itertools import product
from multiprocessing import Manager
import time

# ---------- Common helpers ----------
def _kneedle(x: np.ndarray, y: np.ndarray) -> int:
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    p1 = np.array([x_n[0], y_n[0]]); p2 = np.array([x_n[-1], y_n[-1]])
    v = p2 - p1; v /= np.linalg.norm(v) + 1e-12
    dists = [np.linalg.norm(np.array([x_n[i], y_n[i]]) - (p1 + np.dot(np.array([x_n[i], y_n[i]])-p1, v)*v))
             for i in range(len(x_n))]
    return int(np.argmax(dists))

def _kmeans_wasserstein(X: np.ndarray, labels: np.ndarray) -> float:
    k = int(np.max(labels)) + 1
    vals, sizes = [], []
    for c in range(k):
        idx = (labels == c); n_c = int(idx.sum())
        if n_c <= 1: continue
        Xc = X[idx]; mu = Xc.mean(axis=0)
        wj = [wasserstein_distance(Xc[:, j], np.full(n_c, mu[j], dtype=Xc.dtype)) for j in range(X.shape[1])]
        vals.append(float(np.mean(wj))); sizes.append(n_c)
    if not vals: return np.nan
    w = np.array(sizes, dtype=float); w /= w.sum()
    return float(np.dot(w, np.array(vals, dtype=float)))

# ---------- KMeans scan ----------
@dataclass
class KScanResult:
    ks: List[int]
    inertia: List[float]
    silhouette: List[Optional[float]]
    calinski_harabasz: List[Optional[float]]
    davies_bouldin: List[Optional[float]]
    wasserstein: List[Optional[float]]
    best_k: int
    best_by: Dict[str, int]

def scan_kmeans(
    df: pl.DataFrame | None,
    feature_cols: List[str] | None,
    *,
    X: np.ndarray | None = None,
    k_range: Iterable[int] = range(2, 13),
    standardize: bool = True,
    random_state: int = 42,
    n_init: int | str = "auto",
    n_jobs: int = 8,
    silhouette_sample_size: Optional[int] = 5000,
    use_minibatch: bool = True,
    mbk_max_iter: int = 200,
) -> KScanResult:
    if X is None:
        assert df is not None and feature_cols is not None
        X = _prep_features(df, feature_cols, standardize=standardize, dtype=np.float32)
    ks = list(k_range)
    KM = MiniBatchKMeans if use_minibatch else KMeans
    km_kwargs = dict(n_clusters=None, random_state=random_state, n_init=n_init)
    if use_minibatch:
        km_kwargs.update(dict(batch_size=10000, max_iter=mbk_max_iter, reassignment_ratio=0.01))
    else:
        km_kwargs["algorithm"] = "elkan"

    def _one(k: int):
        start = time.time()
        print(f"[KMeans] Fitting k={k}...")
        
        # with threadpool_limits(limits=1):
        km = KM(**{**km_kwargs, "n_clusters": k})
        labels = km.fit_predict(X)
        inertia = float(km.inertia_)
        if k >= 2 and len(set(labels)) > 1:
            sil = float(silhouette_score(X, labels,
                    sample_size=min(silhouette_sample_size, len(X)) if silhouette_sample_size else None,
                    random_state=random_state))
            ch  = float(calinski_harabasz_score(X, labels))
            db  = float(davies_bouldin_score(X, labels))
            w1  = _kmeans_wasserstein(X, labels)
        else:
            sil = ch = db = w1 = None

        elapsed = time.time() - start
        print(f"[KMeans] k={k} completed in {elapsed:.2f}s")
        return k, inertia, sil, ch, db, w1

    results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one)(k) for k in ks)
    results.sort(key=lambda t: t[0])
    inertia = [r[1] for r in results]; silh = [r[2] for r in results]
    chs = [r[3] for r in results]; dbs = [r[4] for r in results]; w1s = [r[5] for r in results]

    best_by: Dict[str,int] = {}
    best_by["elbow_inertia"]   = ks[_kneedle(np.array(ks), np.array(inertia))]
    if any(v is not None for v in silh): best_by["silhouette"] = ks[int(np.nanargmax(np.array([v if v is not None else -np.inf for v in silh])))]
    if any(v is not None for v in chs):  best_by["calinski_harabasz"] = ks[int(np.nanargmax(np.array([v if v is not None else -np.inf for v in chs])))]
    if any(v is not None for v in dbs):  best_by["davies_bouldin"]   = ks[int(np.nanargmin(np.array([v if v is not None else  np.inf for v in dbs])))]
    if any(v is not None for v in w1s):  best_by["wasserstein"]      = ks[int(np.nanargmin(np.array([v if v is not None else  np.inf for v in w1s])))]

    votes = list(best_by.values()); counts = pd.Series(votes).value_counts()
    best_k = int(counts.index[np.argmax(counts.values)])
    ties = counts[counts == counts.max()].index.tolist()
    if len(ties) > 1: best_k = int(min(ties))

    return KScanResult(ks, inertia, silh, chs, dbs, w1s, best_k, best_by)

def select_k_ensemble(res: KScanResult) -> int:
    return res.best_k

# ---------- Ward scan ----------
@dataclass
class ScanResultWard:
    thresholds: List[float]
    silhouette: List[Optional[float]]
    calinski_harabasz: List[Optional[float]]
    davies_bouldin: List[Optional[float]]
    wasserstein: List[Optional[float]]
    best_threshold: float
    best_by: Dict[str, float]

def scan_ward(
    X: np.ndarray,
    thresholds: Iterable[float],
    *,
    sample_size: Optional[int] = 5000,
    random_state: int = 42,
) -> ScanResultWard:
    rng = np.random.RandomState(random_state)
    if sample_size and sample_size < X.shape[0]:
        idx = rng.choice(X.shape[0], size=sample_size, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X

    Z = linkage(X_eval, method="ward")
    def metrics(t: float):
        labels = fcluster(Z, t=t, criterion="distance") - 1
        if len(set(labels)) <= 1:
            return None, None, None, None
        s = silhouette_score(X_eval, labels, random_state=random_state)
        c = calinski_harabasz_score(X_eval, labels)
        d = davies_bouldin_score(X_eval, labels)
        # simple compactness proxy
        vals = []
        for c_id in np.unique(labels):
            Xc = X_eval[labels == c_id]
            if Xc.shape[0] <= 1: continue
            mu = Xc.mean(axis=0)
            vals.append(np.mean(np.abs(Xc - mu)))
        w = float(np.mean(vals)) if vals else None
        return s, c, d, w

    th = list(map(float, thresholds))
    silh, chs, dbs, w1s = [], [], [], []
    for t in th:
        start = time.time()
        print(f"[Ward] Evaluating threshold={t:.2f}...")
        
        s, c, d, w = metrics(t)
        silh.append(s); chs.append(c); dbs.append(d); w1s.append(w)

        elapsed = time.time() - start
        print(f"[Ward] Threshold={t:.2f} completed in {elapsed:.2f}s")

    def best_idx_high(arr): 
        a = np.array([(-np.inf if v is None else v) for v in arr], dtype=float); 
        return int(np.nanargmax(a))
    def best_idx_low(arr): 
        a = np.array([( np.inf if v is None else v) for v in arr], dtype=float); 
        return int(np.nanargmin(a))

    best_by: Dict[str, float] = {}
    if any(v is not None for v in silh): best_by["silhouette"] = th[best_idx_high(silh)]
    if any(v is not None for v in chs):  best_by["calinski_harabasz"] = th[best_idx_high(chs)]
    if any(v is not None for v in dbs):  best_by["davies_bouldin"]   = th[best_idx_low(dbs)]
    if any(v is not None for v in w1s):  best_by["wasserstein"]      = th[best_idx_low(w1s)]

    votes = list(best_by.values())
    best_threshold = float(np.median(votes)) if votes else th[0]
    return ScanResultWard(th, silh, chs, dbs, w1s, best_threshold, best_by)

def select_threshold_ensemble(res: ScanResultWard) -> float:
    return float(res.best_threshold)

# ---------- Stability optimization (KMeans + Ward) ----------
def _normalize_metrics(silh, dbs):
    s = np.array([np.nan if v is None else float(v) for v in silh], dtype=float)
    d = np.array([np.nan if v is None else float(v) for v in dbs], dtype=float)
    v = np.isfinite(s) & np.isfinite(d)
    if np.any(v):
        s_min, s_max = np.nanmin(s[v]), np.nanmax(s[v]); d_min, d_max = np.nanmin(d[v]), np.nanmax(d[v])
        s_den = (s_max - s_min) if (s_max - s_min) > 0 else 1.0
        d_den = (d_max - d_min) if (d_max - d_min) > 0 else 1.0
    else:
        s_min = s_max = d_min = d_max = 0.0; s_den = d_den = 1.0
    s_n = np.full_like(s, np.nan); s_n[v] = (s[v]-s_min)/s_den
    d_n = np.full_like(d, np.nan); d_n[v] = (d[v]-d_min)/d_den
    return s_n, d_n, v

def optimize_w_weights_by_stability_fast(
    ks: List[int], res: KScanResult, *, X: np.ndarray, alphas=(0.25,0.5,1.0,2.0,4.0),
    betas=(0.25,0.5,1.0,2.0,4.0), random_state: int = 42, n_init: int | str = "auto",
    bootstraps: int = 3, sample_frac: float = 0.4, n_jobs_stability: int = 8, use_minibatch: bool = True, mbk_max_iter: int = 200,
) -> Tuple[float, float, int, Dict]:
    s_n, d_n, valid = _normalize_metrics(res.silhouette, res.davies_bouldin)
    ks_arr = np.array(ks); ks_valid = ks_arr[np.where(valid)[0]]
    if ks_valid.size == 0:
        raise ValueError("No valid (silhouette, DB) pairs to optimize over.")
    rng = np.random.RandomState(random_state)
    n = X.shape[0]
    m = max(2, int(round(sample_frac * n)))

    total = len(ks_valid) * bootstraps  
    manager = Manager()  
    counter = manager.Value("i", 0)  
    
    def compute_ari_for_k_b(k: int, b: int) -> Tuple[int, float]:
        start = time.time()  
        print(f"[Stability-KMeans] Computing ARI for k={k}, bootstrap={b}...") 
        
        KM = MiniBatchKMeans if use_minibatch else KMeans
        km_kwargs = dict(n_clusters=None, random_state=random_state, n_init=n_init)
        if use_minibatch:
            km_kwargs.update(dict(batch_size=10000, max_iter=mbk_max_iter, reassignment_ratio=0.01))
        else:
            km_kwargs["algorithm"] = "elkan"
        km0 = KM(**{**km_kwargs, "n_clusters": k})
        km0.fit(X)
        idx = rng.choice(n, size=m, replace=True); Xb = X[idx]
        kmb = KM(**{**km_kwargs, "n_clusters": k, "random_state": random_state+b+1})
        Lb  = kmb.fit_predict(Xb); L0b = km0.predict(Xb)
        ari = adjusted_rand_score(L0b, Lb)

        elapsed = time.time() - start  
        counter.value += 1
        percent = (counter.value / total) * 100
        print(f"[Stability-KMeans] k={k}, bootstrap={b} completed in {elapsed:.2f}s ({percent:.1f}% done)")  
        
        return k, ari

    results = Parallel(n_jobs=n_jobs_stability, prefer="processes")(
        delayed(compute_ari_for_k_b)(k, b) for k, b in product(ks_valid, range(bootstraps))
    )

    stability_scores = {}
    for k in ks_valid:
        aris_k = [ari for k_, ari in results if k_ == k]
        stability_scores[k] = float(np.mean(aris_k)) if aris_k else np.nan

    best = {"alpha": None, "beta": None, "k": None, "score": -np.inf}
    one_minus_s = 1.0 - s_n[valid]; d_vec = np.abs(d_n[valid])
    for a in alphas:
        for b in betas:
            wdist = a*np.abs(one_minus_s) + b*d_vec
            k_star = int(ks_valid[int(np.argmin(wdist))]); score = float(stability_scores.get(k_star, np.nan))
            if np.isfinite(score) and score > best["score"]:
                best.update({"alpha": a, "beta": b, "k": k_star, "score": score})
    print(f"Best stability {best['score']:.4f} at alpha={best['alpha']}, beta={best['beta']}, k={best['k']}")
    return float(best["alpha"]), float(best["beta"]), int(best["k"]), {"best": best, "stability_by_k": stability_scores}

def optimize_w_weights_by_stability_ward(
    thresholds: List[float], res: ScanResultWard, *, X: np.ndarray, alphas=(0.25,0.5,1.0,2.0,4.0),
    betas=(0.25,0.5,1.0,2.0,4.0), random_state: int = 42, bootstraps: int = 3, sample_frac: float = 0.4,
    n_jobs_stability: int = 8,
) -> Tuple[float, float, float, Dict]:
    s_n, d_n, valid = _normalize_metrics(res.silhouette, res.davies_bouldin)
    th_arr = np.array(thresholds); th_valid = th_arr[np.where(valid)[0]]
    if th_valid.size == 0:
        raise ValueError("No valid (silhouette, DB) pairs to optimize over.")
    rng = np.random.RandomState(random_state)

    Z0 = linkage(X, method="ward")
    n = X.shape[0]
    m = max(2, int(round(sample_frac * n))) 
    
    total = len(th_valid) * bootstraps  
    manager = Manager()  
    counter = manager.Value("i", 0)  
    
    def compute_ari_for_t_b(t: float, b: int) -> Tuple[float, float]:
        start = time.time()  
        print(f"[Stability-Ward] Computing ARI for threshold={t:.2f}, bootstrap={b}...") 
        
        L0 = fcluster(Z0, t=t, criterion="distance") - 1
        idx = rng.choice(n, size=m, replace=True)
        Xb = X[idx]
        L0b = L0[idx]
        Lb = fcluster(linkage(Xb, method="ward"), t=t, criterion="distance") - 1
        ari = adjusted_rand_score(L0b, Lb)

        elapsed = time.time() - start  
        counter.value += 1
        percent = (counter.value / total) * 100
        print(f"[Stability-Ward] threshold={t:.2f}, bootstrap={b} completed in {elapsed:.2f}s ({percent:.1f}% done)")
        
        return t, ari

    results = Parallel(n_jobs=n_jobs_stability, prefer="processes")(
        delayed(compute_ari_for_t_b)(t, b) for t, b in product(th_valid, range(bootstraps))
    )

    stability_scores = {}
    for t in th_valid:
        aris_t = [ari for t_, ari in results if t_ == t]
        stability_scores[t] = float(np.mean(aris_t)) if aris_t else np.nan


    best = {"alpha": None, "beta": None, "threshold": None, "score": -np.inf}
    one_minus_s = 1.0 - s_n[valid]; d_vec = np.abs(d_n[valid])
    for a in alphas:
        for b in betas:
            wdist = a*np.abs(one_minus_s) + b*d_vec
            t_star = float(th_valid[int(np.argmin(wdist))]); score = float(stability_scores.get(t_star, np.nan))
            if np.isfinite(score) and score > best["score"]:
                best.update({"alpha": a, "beta": b, "threshold": t_star, "score": score})
    print(f"Best stability {best['score']:.4f} at alpha={best['alpha']}, beta={best['beta']}, threshold={best['threshold']:.2f}")
    return float(best["alpha"]), float(best["beta"]), float(best["threshold"]), {"best": best, "stability_by_threshold": stability_scores}
