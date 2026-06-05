"""
Visualization Module

Generates analysis charts using matplotlib and seaborn:
- User segment pie/bar charts
- Feature distribution histograms
- Correlation heatmaps
- Churn probability curves
- RFM scatter plots
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path


# Try to find a CJK font for Chinese character support
_CJK_FONT = None
for _f in fm.findSystemFonts():
    try:
        _fp = fm.FontProperties(fname=_f)
        _name = _fp.get_name().lower()
        if any(kw in _name for kw in ["simhei", "microsoft yahei", "noto sans cjk",
                                        "wenquanyi", "source han", "pingfang"]):
            _CJK_FONT = _fp
            break
    except Exception:
        pass


def _safe_title(ax, title: str, fontsize: int = 14):
    """Set title with CJK font fallback."""
    if _CJK_FONT:
        ax.set_title(title, fontproperties=_CJK_FONT, fontsize=fontsize)
    else:
        ax.set_title(title, fontsize=fontsize)


class Visualizer:
    """Generate and save analysis charts."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")

    # ── Public methods ────────────────────────────────────────

    def plot_segment_distribution(self, labels, save: bool = True) -> str:
        """Bar chart showing user count per segment."""
        if isinstance(labels, np.ndarray):
            labels = pd.Series(labels)
        counts = labels.value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = sns.color_palette("Set2", len(counts))
        bars = ax.bar(
            [f"Seg_{i}" for i in counts.index],
            counts.values,
            color=colors,
            edgecolor="white",
        )
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    str(val), ha="center", fontsize=11, fontweight="bold")

        _safe_title(ax, "User Segment Distribution")
        ax.set_xlabel("Segment")
        ax.set_ylabel("User Count")
        plt.tight_layout()
        return self._save("segment_distribution.png", fig)

    def plot_segment_profiles(self, profiles: pd.DataFrame, save: bool = True) -> str:
        """Grouped bar chart comparing mean features across segments."""
        if profiles is None or profiles.empty:
            return ""
        # Select only numeric columns (excluding 'count', 'pct')
        feat_cols = [c for c in profiles.columns
                     if c not in ("segment", "count", "pct")
                     and pd.api.types.is_numeric_dtype(profiles[c])]
        if not feat_cols:
            return ""

        fig, ax = plt.subplots(figsize=(10, 5))
        profiles.set_index("segment")[feat_cols].T.plot(kind="bar", ax=ax, colormap="Set2")
        _safe_title(ax, "Segment Profiles — Feature Comparison")
        ax.set_xlabel("Feature")
        ax.set_ylabel("Mean Value (scaled)")
        ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        return self._save("segment_profiles.png", fig)

    def plot_correlation_heatmap(self, df: pd.DataFrame, save: bool = True) -> str:
        """Correlation heatmap of numeric features."""
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return ""
        corr = numeric.corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, square=True, linewidths=0.5, ax=ax,
                    vmin=-1, vmax=1)
        _safe_title(ax, "Feature Correlation Heatmap")
        plt.tight_layout()
        return self._save("correlation_heatmap.png", fig)

    def plot_feature_importance(self, importance_df: pd.DataFrame, save: bool = True) -> str:
        """Horizontal bar chart of feature importance."""
        if importance_df is None or importance_df.empty:
            return ""
        df = importance_df.head(15)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = sns.color_palette("Blues_r", len(df))
        ax.barh(df["feature"], df["importance"], color=colors, edgecolor="white")
        _safe_title(ax, "Feature Importance")
        ax.set_xlabel("Importance")
        ax.invert_yaxis()
        plt.tight_layout()
        return self._save("feature_importance.png", fig)

    # ── Internal ──────────────────────────────────────────────

    def _save(self, filename: str, fig) -> str:
        """Save figure to output dir and close."""
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart saved: {path}")
        return str(path)
