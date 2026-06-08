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

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray,
        class_names: list = None, save: bool = True,
    ) -> str:
        """Confusion matrix heatmap showing prediction vs actual."""
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        if class_names is None:
            class_names = [str(c) for c in sorted(set(y_true))]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            linewidths=0.5, ax=ax,
        )
        _safe_title(ax, "Confusion Matrix — Prediction vs Actual")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        return self._save("confusion_matrix.png", fig)

    def plot_prediction_distribution(
        self, y_true: np.ndarray, y_proba: np.ndarray,
        target_name: str = "target", save: bool = True,
    ) -> str:
        """Histogram of predicted probabilities, colored by true class."""
        if y_proba is None or y_true is None:
            return ""

        fig, ax = plt.subplots(figsize=(8, 5))
        classes = sorted(set(y_true))
        colors = sns.color_palette("Set2", len(classes))

        for i, cls in enumerate(classes):
            mask = y_true == cls
            ax.hist(
                y_proba[mask], bins=20, alpha=0.6,
                color=colors[i], label=f"Actual Class {cls}",
                edgecolor="white",
            )
        ax.axvline(x=0.5, color="red", linestyle="--", linewidth=1.5, label="Decision Boundary")
        _safe_title(ax, f"Prediction Probability Distribution ({target_name})")
        ax.set_xlabel(f"Predicted Probability of Class 1 ({target_name})")
        ax.set_ylabel("User Count")
        ax.legend()
        plt.tight_layout()
        return self._save("prediction_distribution.png", fig)

    def plot_model_performance(
        self, metrics: dict, save: bool = True,
    ) -> str:
        """Simple summary card showing model metrics as a chart."""
        if not metrics:
            return ""

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.axis("off")

        lines = [
            f"Model: {metrics.get('model', 'N/A')}",
            f"Target: {metrics.get('target', 'N/A')}",
            f"Accuracy: {metrics.get('accuracy', 'N/A')}",
            f"CV Mean Score: {metrics.get('cv_mean', 'N/A')}",
            f"Test Samples: {metrics.get('test_samples', 'N/A')}",
            f"Classes: {metrics.get('classes', [])}",
        ]
        text = "\n".join(lines)
        ax.text(
            0.5, 0.5, text, transform=ax.transAxes,
            fontsize=14, verticalalignment="center", horizontalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F0FE", edgecolor="#4A90D9"),
        )
        _safe_title(ax, "Prediction Model Performance")
        plt.tight_layout()
        return self._save("model_performance.png", fig)

    # ── Internal ──────────────────────────────────────────────

    def _save(self, filename: str, fig) -> str:
        """Save figure to output dir and close."""
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Chart saved: {path}")
        return str(path)
