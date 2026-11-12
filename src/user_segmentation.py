"""
User Segmentation Module

Performs user layering with a multi-model candidate pool:
- K-Means          (elbow-based auto-k)
- Gaussian Mixture (soft, elliptical clusters)
- Hierarchical     (agglomerative, ward linkage)
- DBSCAN           (density-based, noise-tolerant)

Auto mode fits every eligible candidate and selects the winner by
silhouette score. Guardrails: silhouette needs >= 2 clusters with
>= 2 samples each; DBSCAN is O(n^2)-ish so it is skipped on large
data and rejected when it finds one cluster or too much noise.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

# DBSCAN's neighbor graph is O(n^2) — skip beyond this size
DBSCAN_MAX_SAMPLES = 10000
# Reject DBSCAN if more than this share of points is noise
DBSCAN_MAX_NOISE = 0.3
# Silhouette is O(n^2) — subsample beyond this cap
SILHOUETTE_SAMPLE_CAP = 3000

PARTITION_METHODS = ("kmeans", "gmm", "hierarchical")


class UserSegmentation:
    """Segment users into distinct groups based on behavior patterns.

    Main entry: auto_segment() — auto-selects features, cluster count,
    and clustering algorithm.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.labels_: np.ndarray = None
        self.profiles_: pd.DataFrame = None
        self.method_: str = ""          # algorithm actually used
        self.n_clusters_: int = 0
        self.silhouette_: float = None
        self._feature_cols: list = []

    # ── Main API ─────────────────────────────────────────────

    def auto_segment(self, n_clusters: int = None,
                     method: str = "auto") -> np.ndarray:
        """
        Segment users; auto-selects the clustering algorithm.

        Args:
            n_clusters: number of segments; None = auto (2-6 via elbow).
            method: 'kmeans' | 'gmm' | 'hierarchical' | 'dbscan' | 'auto'.
                    'auto' fits all eligible candidates and keeps the
                    one with the highest silhouette score.
        """
        self._feature_cols = self._pick_features()
        if len(self._feature_cols) < 2:
            print("  Warning: < 2 numeric features; using all numeric columns.")
            self._feature_cols = self.df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        if len(self._feature_cols) < 2:
            print("  Not enough features for clustering. Assigning single segment.")
            self.labels_ = np.zeros(len(self.df), dtype=int)
            self.method_ = "none"
            self.n_clusters_ = 1
            self.profiles_ = self._build_profiles()
            return self.labels_

        X = self.df[self._feature_cols].fillna(0).values

        if method == "dbscan":
            labels, _ = self._fit_dbscan(X)
            if labels is None:
                print("  DBSCAN ineligible — falling back to kmeans.")
                labels = self._fit_one("kmeans", X, n_clusters)
        elif method in PARTITION_METHODS:
            labels = self._fit_one(method, X, n_clusters)
        else:
            if method != "auto":
                print(f"  Unknown method '{method}' — falling back to auto.")
            labels = self._auto_select(X, n_clusters)

        self.labels_ = labels
        self.profiles_ = self._build_profiles()
        return self.labels_

    def get_segment_profiles(self) -> pd.DataFrame:
        """Return a summary table of each segment's characteristics."""
        if self.profiles_ is None:
            self._build_profiles()
        return self.profiles_

    def get_segment_summary(self) -> dict:
        """Return a dict summarizing each segment."""
        if self.profiles_ is None:
            self._build_profiles()
        result = {}
        for _, row in self.profiles_.iterrows():
            label = row.get("segment", row.name)
            result[label] = {
                "count": int(row["count"]),
                "pct": round(float(row["pct"]), 1),
            }
        return result

    # ── Model selection ──────────────────────────────────────

    def _auto_select(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """Fit every eligible candidate, keep the best silhouette."""
        candidates = {}
        for name in PARTITION_METHODS:
            labels, score = self._fit_one(name, X, n_clusters,
                                          return_score=True)
            if labels is not None and score is not None:
                candidates[name] = (labels, score)

        # DBSCAN needs no k but is O(n^2)-ish — small data only
        if len(X) <= DBSCAN_MAX_SAMPLES:
            labels, score = self._fit_dbscan(X)
            if labels is not None and score is not None:
                candidates["dbscan"] = (labels, score)

        if not candidates:
            print("  All models failed — falling back to single segment.")
            self.method_ = "none"
            self.n_clusters_ = 1
            self.silhouette_ = None
            return np.zeros(len(X), dtype=int)

        best = max(candidates, key=lambda k: candidates[k][1])
        labels, score = candidates[best]
        self.method_ = best
        self.silhouette_ = round(float(score), 4)
        self.n_clusters_ = len(set(labels))
        print(f"  Auto-selected {best}: {self.n_clusters_} segments, "
              f"silhouette={self.silhouette_}")
        return labels

    def _fit_one(self, name: str, X: np.ndarray, n_clusters: int,
                 return_score: bool = False):
        """Fit one partitional model. Shared k selection: elbow on
        K-Means inertia, reused as the k prior for the others."""
        if len(X) < 3:
            labels = np.zeros(len(X), dtype=int)
            self._set_meta(name, 1, None)
            return (labels, None) if return_score else labels

        k = n_clusters if n_clusters and n_clusters > 0 else None
        if k is None:
            k = self._find_best_k(X)
        k = max(2, min(k, len(X) - 1))

        if name == "kmeans":
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        elif name == "gmm":
            # 'tied' covariance avoids over-parameterizing small samples
            model = GaussianMixture(n_components=k, covariance_type="tied",
                                    random_state=42)
        else:  # hierarchical
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")

        labels = model.fit_predict(X)
        score = self._safe_silhouette(X, labels)
        self._set_meta(name, k, score)
        print(f"  [{name}] k={k}, silhouette="
              f"{None if score is None else round(score, 4)}")
        return (labels, score) if return_score else labels

    def _fit_dbscan(self, X: np.ndarray):
        """DBSCAN with eps auto-derived from the k-distance graph.
        Rejected when it yields <2 clusters or >DBSCAN_MAX_NOISE noise."""
        n = len(X)
        if n < 5:
            return None, None
        min_samples = max(3, min(2 * X.shape[1], n // 20))
        knn = NearestNeighbors(n_neighbors=min_samples)
        knn.fit(X)
        kdist = np.sort(knn.kneighbors(X)[0][:, -1])
        eps = float(np.percentile(kdist, 90))  # elbow heuristic
        if eps <= 0:
            return None, None

        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
        unique = set(labels)
        if len(unique) < 2:
            print("  [dbscan] single cluster — skipped")
            return None, None
        noise_ratio = float((labels == -1).mean())
        if noise_ratio > DBSCAN_MAX_NOISE:
            print(f"  [dbscan] noise={noise_ratio:.0%} > "
                  f"{DBSCAN_MAX_NOISE:.0%} — skipped")
            return None, None
        score = self._safe_silhouette(X[labels != -1], labels[labels != -1])
        self._set_meta("dbscan", len(unique), score)
        print(f"  [dbscan] eps={eps:.3f}, {len(unique)} clusters, "
              f"silhouette={None if score is None else round(score, 4)}")
        return labels, score

    def _safe_silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Silhouette with guardrails: >=2 clusters, >=2 samples each;
        subsample to cap the O(n^2) cost."""
        valid = labels >= 0
        X, labels = X[valid], labels[valid]
        if len(set(labels)) < 2 or len(X) < 4:
            return None
        if pd.Series(labels).value_counts().min() < 2:
            return None
        if len(X) > SILHOUETTE_SAMPLE_CAP:
            idx = np.random.RandomState(42).choice(
                len(X), SILHOUETTE_SAMPLE_CAP, replace=False)
            X, labels = X[idx], labels[idx]
            if pd.Series(labels).value_counts().min() < 2:
                return None
        try:
            return float(silhouette_score(X, labels))
        except Exception:
            return None

    def _set_meta(self, method: str, n_clusters: int, score):
        self.method_ = method
        self.n_clusters_ = n_clusters
        self.silhouette_ = round(float(score), 4) if score is not None else None

    # ── Internal helpers ─────────────────────────────────────

    def _pick_features(self) -> list:
        """Select numeric columns for clustering, prefer recency/frequency features."""
        numeric = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Prioritize recency/frequency-like columns if present
        priority = [c for c in numeric if any(
            kw in c.lower() for kw in ["recency", "freq", "count", "amount", "day"]
        )]
        if len(priority) >= 2:
            return priority[:8]
        return numeric[:10]  # cap at 10 features for simplicity

    def _find_best_k(self, X: np.ndarray, max_k: int = 6) -> int:
        """Simple elbow method: pick k with best inertia improvement."""
        if len(X) < 10:
            return 2

        max_k = min(max_k, len(X) - 1, 6)
        inertias = []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init="auto")
            km.fit(X)
            inertias.append(km.inertia_)

        # Pick k where the improvement drop is largest relative to next
        if len(inertias) < 3:
            return 2
        # Simple: use k=3 as reasonable default, or pick based on elbow
        improvements = [
            (inertias[i] - inertias[i + 1]) / inertias[i]
            for i in range(len(inertias) - 1)
        ]
        # Find k with diminishing returns (improvement < 10% of previous)
        for i in range(1, len(improvements)):
            if improvements[i] < improvements[i - 1] * 0.5:
                return i + 2  # +2 because i=0 corresponds to k=2
        return 3  # default

    def _build_profiles(self) -> pd.DataFrame:
        """Build a summary DataFrame: segment, count, pct, mean per feature."""
        if self.labels_ is None:
            return pd.DataFrame()

        df = self.df.copy()
        df["segment"] = self.labels_

        rows = []
        total = len(df)
        for seg in sorted(df["segment"].unique()):
            subset = df[df["segment"] == seg]
            row = {"segment": f"Segment_{seg}", "count": len(subset),
                   "pct": round(100 * len(subset) / total, 1)}
            for col in self._feature_cols:
                if col in df.columns:
                    row[col] = round(subset[col].mean(), 3)
            rows.append(row)

        self.profiles_ = pd.DataFrame(rows)
        return self.profiles_
