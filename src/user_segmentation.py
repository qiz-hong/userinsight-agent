"""
User Segmentation Module

Performs user layering using:
- RFM (Recency, Frequency, Monetary) analysis
- K-Means clustering
- Outputs user segment labels and segment profiles
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


class UserSegmentation:
    """Segment users into distinct groups based on behavior patterns.

    Main entry: auto_segment() — auto-selects features and cluster count.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.labels_: np.ndarray = None
        self.profiles_: pd.DataFrame = None
        self._feature_cols: list = []

    # ── Main API ─────────────────────────────────────────────

    def auto_segment(self, n_clusters: int = None) -> np.ndarray:
        """
        Automatically select numeric features and cluster count then
        run K-Means clustering.

        If n_clusters is None, auto-selects between 2-5 using elbow method.
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
            self.profiles_ = self._build_profiles()
            return self.labels_

        X = self.df[self._feature_cols].fillna(0).values

        if n_clusters is None:
            n_clusters = self._find_best_k(X)

        print(f"  Clustering on {len(self._feature_cols)} features → {n_clusters} segments")
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        self.labels_ = model.fit_predict(X)
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

    # ── Internal ──────────────────────────────────────────────

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
