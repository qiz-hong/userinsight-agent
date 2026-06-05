"""
UserInsight Agent — Main Entry Point

Orchestrates the full data analysis pipeline:
1. Load data      2. Clean data       3. Feature engineering
4. Segmentation   5. Prediction       6. Visualization
7. Report generation
"""

import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from user_segmentation import UserSegmentation
from behavior_prediction import BehaviorPredictor
from visualization import Visualizer
from report_generator import ReportGenerator


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config not found: {config_path}, using defaults.")
        return _default_config()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_config() -> dict:
    """Return a minimal default config."""
    return {
        "data": {"file_path": "data/sample_user_data.csv", "user_id_column": "user_id"},
        "cleaning": {"fill_missing_strategy": "auto"},
        "output": {"dir": "output"},
        "prediction": {"target_column": None},
        "segmentation": {"n_clusters": None},
    }


def _step(n: int, total: int, desc: str):
    """Print a step header."""
    print(f"\n[{n}/{total}] {desc}")


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  UserInsight Agent — User Profiling & Behavior Analysis")
    print("=" * 60)

    # ── Bootstrap ────────────────────────────────────────────
    config = load_config()
    output_dir = config.get("output", {}).get("dir", "output")
    chart_paths = []
    segment_labels = None
    profiles = None
    importance_df = None

    # ── 1. Load ──────────────────────────────────────────────
    df = None
    try:
        _step(1, 7, "Loading data...")
        loader = DataLoader(config["data"]["file_path"])
        df = loader.load()
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return

    # ── 2. Clean ─────────────────────────────────────────────
    try:
        _step(2, 7, "Cleaning data...")
        cleaner = DataCleaner(df)
        strategy = config.get("cleaning", {}).get("fill_missing_strategy", "auto")
        cleaner.drop_duplicates().fill_missing(strategy)

        outlier_cols = config.get("cleaning", {}).get("outlier_columns") or None
        if outlier_cols:
            cleaner.remove_outliers(outlier_cols)

        cleaner.correct_dtypes()
        df_clean = cleaner.get_clean_data()
    except Exception as e:
        print(f"  WARNING during cleaning: {e}, continuing with raw data.")
        df_clean = df

    # ── 3. Feature Engineering ───────────────────────────────
    try:
        _step(3, 7, "Engineering features...")
        id_col = config.get("data", {}).get("user_id_column")
        engineer = FeatureEngineer(df_clean)
        engineer.auto_process(id_column=id_col)
        df_features = engineer.get_features()
    except Exception as e:
        print(f"  WARNING during feature engineering: {e}")
        df_features = df_clean.select_dtypes(include=["number"])

    # ── 4. Segmentation ──────────────────────────────────────
    try:
        _step(4, 7, "Segmenting users...")
        n_clusters = config.get("segmentation", {}).get("n_clusters")
        segmenter = UserSegmentation(df_features)
        segment_labels = segmenter.auto_segment(n_clusters=n_clusters)
        profiles = segmenter.get_segment_profiles()
    except Exception as e:
        print(f"  WARNING during segmentation: {e}")

    # ── 5. Prediction ────────────────────────────────────────
    try:
        _step(5, 7, "Training prediction model...")
        target_col = config.get("prediction", {}).get("target_column")
        predictor = BehaviorPredictor(df_features)
        predictor.auto_predict(target_column=target_col, segment_labels=segment_labels)
        importance_df = predictor.get_feature_importance()
    except Exception as e:
        print(f"  WARNING during prediction: {e}")
        predictor = None

    # ── 6. Visualization ─────────────────────────────────────
    try:
        _step(6, 7, "Generating charts...")
        viz = Visualizer(output_dir)

        if segment_labels is not None:
            p = viz.plot_segment_distribution(segment_labels)
            if p:
                chart_paths.append(p)

        if profiles is not None and not profiles.empty:
            p = viz.plot_segment_profiles(profiles)
            if p:
                chart_paths.append(p)

        p = viz.plot_correlation_heatmap(df_features)
        if p:
            chart_paths.append(p)

        if importance_df is not None and not importance_df.empty:
            p = viz.plot_feature_importance(importance_df)
            if p:
                chart_paths.append(p)
    except Exception as e:
        print(f"  WARNING during visualization: {e}")

    # ── 7. Report ────────────────────────────────────────────
    try:
        _step(7, 7, "Generating reports...")
        reporter = ReportGenerator(output_dir)

        # Build summary dict
        seg_summary = {}
        if segmenter is not None and segment_labels is not None:
            seg_summary = segmenter.get_segment_summary()

        pred_metrics = {}
        if predictor is not None:
            pred_metrics = predictor.get_metrics()

        report_summary = {
            "Data": f"{len(df)} rows, {len(df.columns)} columns",
            "Segments": seg_summary,
            "Model": pred_metrics,
        }

        reporter.print_console_summary(report_summary)

        # Excel
        reporter.generate_excel(
            segments_df=profiles,
            profiles_df=profiles,
            importance_df=importance_df,
            metrics=pred_metrics,
        )

        # HTML
        reporter.generate_html_summary(
            summary=report_summary,
            chart_paths=chart_paths,
        )
    except Exception as e:
        print(f"  WARNING during report generation: {e}")

    # ── Done ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Analysis complete! Results in '{output_dir}/'")
    if chart_paths:
        print(f"  Charts: {len(chart_paths)} generated")
    print("=" * 60)


if __name__ == "__main__":
    main()
