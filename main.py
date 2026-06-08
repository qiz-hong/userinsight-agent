"""
UserInsight Agent — Main Entry Point

Agent-powered analysis pipeline with LLM decision nodes:
0. LLM diagnoses data → cleaning plan
1. Load data      2. Clean data (LLM-guided)  3. Feature engineering
4. Segmentation (LLM-planned)  5. Prediction   6. Visualization
7. Report + LLM business insights
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
from llm_client import get_client
from llm_diagnoser import LLMDiagnoser
from llm_planner import LLMPlanner
from llm_insight import LLMInsight


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
        "llm": {"enabled": True},
    }


def _step(n: int, total: int, desc: str, agent: bool = False):
    """Print a step header. agent=True adds an '[Agent]' tag."""
    tag = "[Agent]" if agent else "      "
    print(f"\n[{n}/{total}] {tag} {desc}")


# ═══════════════════════════════════════════════════════════════
#  MAIN PIPELINE (also importable as main_pipeline())
# ═══════════════════════════════════════════════════════════════

def main_pipeline(file_path: str = None, api_key: str = None) -> dict:
    """
    Run the full analysis pipeline. Returns a dict with all results.
    Can be called from CLI or programmatically (e.g., from app.py).
    """
    config = load_config()
    output_dir = config.get("output", {}).get("dir", "output")
    llm_enabled = config.get("llm", {}).get("enabled", True)
    data_path = file_path or config["data"]["file_path"]

    # Init LLM client
    llm = get_client(api_key)
    if not llm.available:
        llm_enabled = False

    chart_paths = []
    segment_labels = None
    profiles = None
    importance_df = None
    llm_diagnosis = None
    llm_plan = None
    llm_insight = None

    total_steps = 8 if llm_enabled else 7
    step = 0

    print("=" * 60)
    print("  UserInsight Agent — User Profiling & Behavior Analysis")
    print(f"  LLM Decision Layer: {'ACTIVE' if llm_enabled else 'inactive (no API key)'}")
    print("=" * 60)

    # ═══ Step 1: Load ═══════════════════════════════════════
    df = None
    try:
        step += 1
        _step(step, total_steps, "Loading data...")
        loader = DataLoader(data_path)
        df = loader.load()
    except Exception as e:
        print(f"  ERROR loading data: {e}")
        return {"error": str(e)}

    # ═══ Step 1.5 (Agent): LLM Data Diagnosis ═════════════
    if llm_enabled:
        try:
            step += 1
            _step(step, total_steps,
                  "Agent is diagnosing data quality...", agent=True)
            diagnoser = LLMDiagnoser()
            llm_diagnosis = diagnoser.diagnose(df)
            if llm_diagnosis:
                for issue in llm_diagnosis.get("issues_found", []):
                    print(f"  Issue: {issue}")
                quality = llm_diagnosis.get("overall_quality", "?")
                print(f"  Quality: {quality}")
                # Apply LLM-suggested ID column
                llm_ids = llm_diagnosis.get("id_columns", [])
                if llm_ids and not config.get("data", {}).get("user_id_column"):
                    config.setdefault("data", {})["user_id_column"] = llm_ids[0]
                    print(f"  Agent detected ID: {llm_ids[0]}")
        except Exception as e:
            print(f"  Agent diagnosis skipped: {e}")

    # ═══ Step 2: Clean ═════════════════════════════════════
    try:
        step += 1
        _step(step, total_steps, "Cleaning data...")
        cleaner = DataCleaner(df)
        drop_dupes = config.get("cleaning", {}).get("drop_duplicates", True)
        if drop_dupes:
            cleaner.drop_duplicates()

        # Use LLM fill strategies if available, else config fallback
        if llm_diagnosis and llm_diagnosis.get("fill_strategy"):
            fill_map = llm_diagnosis["fill_strategy"]
            for col, strat in fill_map.items():
                if col in df.columns:
                    cleaner.fill_missing(strat)
        else:
            strategy = config.get("cleaning", {}).get("fill_missing_strategy", "auto")
            cleaner.fill_missing(strategy)

        # Outlier removal from LLM or config
        if llm_diagnosis and llm_diagnosis.get("outlier_columns"):
            cleaner.remove_outliers(llm_diagnosis["outlier_columns"])
        elif config.get("cleaning", {}).get("outlier_columns"):
            cleaner.remove_outliers(config["cleaning"]["outlier_columns"])

        cleaner.correct_dtypes()
        df_clean = cleaner.get_clean_data()
    except Exception as e:
        print(f"  WARNING in cleaning: {e}, continuing with raw data.")
        df_clean = df

    # ═══ Step 3: Feature Engineering ═══════════════════════
    try:
        step += 1
        _step(step, total_steps, "Engineering features...")
        id_col = config.get("data", {}).get("user_id_column")
        engineer = FeatureEngineer(df_clean)
        engineer.auto_process(id_column=id_col)
        df_features = engineer.get_features()
    except Exception as e:
        print(f"  WARNING in feature eng: {e}")
        df_features = df_clean.select_dtypes(include=["number"])

    # ═══ Step 3.5 (Agent): LLM Strategy Planning ══════════
    if llm_enabled:
        try:
            step += 1
            _step(step, total_steps,
                  "Agent is planning analysis strategy...", agent=True)
            planner = LLMPlanner()
            llm_plan = planner.plan(df_features)
            if llm_plan:
                seg_p = llm_plan.get("segmentation", {})
                pred_p = llm_plan.get("prediction", {})
                print(f"  Clusters: {seg_p.get('n_clusters', '?')} "
                      f"({seg_p.get('reasoning', '')})")
                print(f"  Model: {pred_p.get('model_preference', '?')} "
                      f"({pred_p.get('reasoning', '')})")
        except Exception as e:
            print(f"  Agent planning skipped: {e}")

    # ═══ Step 4: Segmentation ══════════════════════════════
    try:
        step += 1
        _step(step, total_steps, "Segmenting users...")
        n_clusters = config.get("segmentation", {}).get("n_clusters")
        if llm_plan and n_clusters is None:
            n_clusters = llm_plan.get("segmentation", {}).get("n_clusters")
        segmenter = UserSegmentation(df_features)
        segment_labels = segmenter.auto_segment(n_clusters=n_clusters)
        profiles = segmenter.get_segment_profiles()
    except Exception as e:
        print(f"  WARNING in segmentation: {e}")

    # ═══ Step 5: Prediction ════════════════════════════════
    try:
        step += 1
        _step(step, total_steps, "Training prediction model...")
        target_col = config.get("prediction", {}).get("target_column")
        if llm_plan and not target_col:
            target_col = llm_plan.get("prediction", {}).get("recommended_target")
        predictor = BehaviorPredictor(df_features)
        predictor.auto_predict(target_column=target_col, segment_labels=segment_labels)
        importance_df = predictor.get_feature_importance()
    except Exception as e:
        print(f"  WARNING in prediction: {e}")
        predictor = None

    # ═══ Step 6: Visualization ═════════════════════════════
    try:
        step += 1
        _step(step, total_steps, "Generating charts...")
        viz = Visualizer(output_dir)

        # Segment charts
        if segment_labels is not None:
            p = viz.plot_segment_distribution(segment_labels)
            if p:
                chart_paths.append(p)
        if profiles is not None and not profiles.empty:
            p = viz.plot_segment_profiles(profiles)
            if p:
                chart_paths.append(p)

        # Correlation heatmap
        p = viz.plot_correlation_heatmap(df_features)
        if p:
            chart_paths.append(p)

        # Feature importance
        if importance_df is not None and not importance_df.empty:
            p = viz.plot_feature_importance(importance_df)
            if p:
                chart_paths.append(p)

        # ══ Prediction charts ══
        if predictor is not None and predictor.y_test_ is not None:
            # Confusion matrix
            class_names = [f"Not {predictor._target_col}", predictor._target_col] \
                if hasattr(predictor, '_target_col') else None
            p = viz.plot_confusion_matrix(
                predictor.y_test_, predictor.y_pred_,
                class_names=class_names,
            )
            if p:
                chart_paths.append(p)

            # Prediction probability distribution
            if predictor.y_proba_ is not None:
                p = viz.plot_prediction_distribution(
                    predictor.y_test_, predictor.y_proba_,
                    target_name=predictor._target_col if hasattr(predictor, '_target_col') else "target",
                )
                if p:
                    chart_paths.append(p)

            # Model performance card
            p = viz.plot_model_performance(predictor.get_metrics())
            if p:
                chart_paths.append(p)

    except Exception as e:
        print(f"  WARNING in visualization: {e}")

    # ═══ Step 6.5 (Agent): LLM Business Insight ═══════════
    if llm_enabled:
        try:
            step += 1
            _step(step, total_steps,
                  "Agent is generating business insights...", agent=True)
            seg_summary = segmenter.get_segment_summary() if (
                segmenter is not None and segment_labels is not None
            ) else {}
            pred_metrics = predictor.get_metrics() if predictor else {}
            fi_text = importance_df.head(5).to_string() if (
                importance_df is not None and not importance_df.empty
            ) else ""

            insight_gen = LLMInsight()
            llm_insight = insight_gen.generate(
                segments_summary=seg_summary,
                model_metrics=pred_metrics,
                feature_importance=fi_text,
                data_summary=f"{len(df)} rows, {len(df.columns)} cols",
            )
            if llm_insight:
                print(f"  {llm_insight.get('headline', '')}")
        except Exception as e:
            print(f"  Agent insight skipped: {e}")

    # ═══ Step 7: Report ════════════════════════════════════
    try:
        step += 1
        _step(step, total_steps, "Generating reports...")
        reporter = ReportGenerator(output_dir)

        seg_summary = segmenter.get_segment_summary() if (
            segmenter is not None and segment_labels is not None
        ) else {}
        pred_metrics = predictor.get_metrics() if predictor else {}

        report_summary = {
            "Data": f"{len(df)} rows, {len(df.columns)} cols",
            "Segments": seg_summary,
            "Model": pred_metrics,
        }
        if llm_insight:
            report_summary["Agent Headline"] = llm_insight.get("headline", "")
            for i, f in enumerate(llm_insight.get("top_findings", []), 1):
                report_summary[f"Finding {i}"] = f
            for i, r in enumerate(llm_insight.get("recommendations", []), 1):
                report_summary[f"Recommendation {i}"] = r

        reporter.print_console_summary(report_summary)
        reporter.generate_excel(
            segments_df=profiles, profiles_df=profiles,
            importance_df=importance_df, metrics=pred_metrics,
        )
        reporter.generate_html_summary(
            summary=report_summary, chart_paths=chart_paths,
        )
    except Exception as e:
        print(f"  WARNING in report generation: {e}")

    # ── Done ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Analysis complete! Results in '{output_dir}/'")
    if chart_paths:
        print(f"  Charts: {len(chart_paths)} generated")
    if llm_insight:
        print(f"  Agent insights: generated")
    print("=" * 60)

    return {
        "profiles": profiles,
        "segment_labels": segment_labels,
        "importance_df": importance_df,
        "chart_paths": chart_paths,
        "seg_summary": seg_summary,
        "pred_metrics": pred_metrics,
        "llm_insight": llm_insight,
        "llm_diagnosis": llm_diagnosis,
        "llm_plan": llm_plan,
        "output_dir": output_dir,
    }


def main():
    """CLI entry point."""
    main_pipeline()


if __name__ == "__main__":
    main()
