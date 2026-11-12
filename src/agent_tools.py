"""
Agent Tool Layer

Each analysis capability is exposed as a LangChain @tool with a typed
schema so the Executor (LLM) can select and parameterize them via
Function Calling. Tools operate on a shared Workspace singleton that
holds the DataFrame and intermediate results — the LLM only ever
passes scalar parameters, never data.

Every tool returns a compact, LLM-friendly string summary (or a
JSON-ish dict) and never raises: failures return an error string so
the Executor can feed it back into the loop.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from langchain_core.tools import tool

# Allow imports when run from any cwd
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import DataLoader
from data_cleaner import DataCleaner
from feature_engineering import FeatureEngineer
from user_segmentation import UserSegmentation
from behavior_prediction import BehaviorPredictor
from visualization import Visualizer
from report_generator import ReportGenerator


# ═══════════════════════════════════════════════════════════════
#  Shared Workspace (holds state tools operate on)
# ═══════════════════════════════════════════════════════════════

class Workspace:
    """Mutable bag holding the working data + intermediate results."""

    def __init__(self):
        self.df: pd.DataFrame = None          # raw loaded
        self.df_clean: pd.DataFrame = None    # after cleaning
        self.df_features: pd.DataFrame = None  # after engineering
        self.segment_labels: np.ndarray = None
        self.profiles: pd.DataFrame = None
        self.importance_df: pd.DataFrame = None
        self.pred_metrics: dict = {}
        self.predictor: BehaviorPredictor = None
        self.chart_paths: list = []
        self.output_dir: str = "output"
        self.data_path: str = ""

    def reset(self, data_path: str = None, output_dir: str = "output"):
        self.__init__()
        if data_path:
            self.data_path = data_path
        self.output_dir = output_dir


ws = Workspace()


# ═══════════════════════════════════════════════════════════════
#  Tools
# ═══════════════════════════════════════════════════════════════

@tool
def inspect_schema() -> str:
    """Inspect the loaded dataset: shape, dtypes, missing rates,
    and data-quality stats. Always call this first after loading."""
    if ws.df is None:
        return "ERROR: no data loaded. Call load_data first."
    df = ws.df
    rows, cols = df.shape
    lines = [f"shape: {rows} rows x {cols} cols"]
    for c in df.columns:
        n_null = df[c].isnull().sum()
        rate = round(n_null / max(rows, 1) * 100, 1)
        lines.append(f"  {c}: {df[c].dtype}, unique={df[c].nunique()}, "
                     f"missing={rate}%")
    constant = [c for c in df.columns if df[c].nunique() <= 1]
    if constant:
        lines.append(f"constant_columns: {constant}")
    return "\n".join(lines)


@tool
def query_data(sql: str = "", question: str = "") -> str:
    """Query the loaded data with SQL via an embedded DuckDB engine.
    Handles natural-language questions by translating them to SQL.

    Args:
        sql: explicit SELECT statement (validated, read-only, capped);
             ignored when question is also given
        question: natural-language question; translated to SQL with
             the LLM and then executed the same way
    """
    df = ws.df_clean if ws.df_clean is not None else ws.df
    if df is None:
        return "ERROR: no data to query. Run load_data first."
    try:
        import duckdb
    except ImportError:
        return "ERROR: duckdb is not installed (pip install duckdb)."

    # ── Translate natural language to SQL (LLM available?) ──
    if not sql and question:
        sql = _translate_question_to_sql(df, question)
        if sql is None:
            return ("ERROR: no API key — cannot translate the question "
                    "to SQL. Pass an explicit sql= statement instead.")

    if not sql:
        return "ERROR: nothing to run. Give sql= or question=."

    # ── Guardrails: read-only SELECT/WITH only ──
    stripped = sql.strip().rstrip(";").lower()
    if not (stripped.startswith("select") or stripped.startswith("with")):
        return f"ERROR: only read-only SELECT/WITH statements are allowed. Got: {sql[:80]}"
    if any(kw in stripped for kw in
           (" insert ", " update ", " delete ", " drop ", " create ",
            " alter ", " attach ", " copy ", " export ")):
        return f"ERROR: statement contains a forbidden keyword. Got: {sql[:80]}"

    try:
        con = duckdb.connect()
        con.register("data", df)
        # Cap output size so a careless query can't flood the context
        result = con.execute(f"SELECT * FROM (\n{sql}\n) LIMIT 500").fetchdf()
        con.close()
    except Exception as e:
        return f"ERROR: query failed: {e}"

    preview = result.head(10).to_string(index=False, max_colwidth=24)
    return (f"OK: {len(result)} rows x {len(result.columns)} cols.\n"
            f"Columns: {list(result.columns)[:20]}\nPreview:\n{preview}")


def _translate_question_to_sql(df: pd.DataFrame, question: str) -> str | None:
    """Ask the LLM to translate a question into a single SELECT.
    Returns None when no LLM is available so the caller degrades."""
    import os
    api_key = os.environ.get("DEEPSEEK_API_KEY") \
        or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url=os.environ.get("LLM_BASE_URL",
                                    "https://api.deepseek.com/v1"),
            api_key=api_key,
            model=os.environ.get("LLM_MODEL", "deepseek-chat"),
            temperature=0,
            max_tokens=512,
            timeout=60,
        )
    except Exception:
        return None

    cols = []
    for c in df.columns[:30]:
        cols.append(f"  {c}: {df[c].dtype}, unique={df[c].nunique()}")
    system = (
        "You translate business questions into DuckDB SQL over a table "
        "named 'data' (double-quote column names).\n"
        "Rules: output ONLY one SELECT statement, no explanation, no "
        "markdown; only SELECT/WITH; never modify data; keep the result "
        "aggregated unless detail rows are explicitly requested."
    )
    user = (f"Table 'data' schema:\n{chr(10).join(cols)}\n\n"
            f"Question: {question}\n\nSQL:")
    try:
        msg = llm.invoke([{"role": "system", "content": system},
                          {"role": "user", "content": user}])
        raw = str(msg.content).strip()
        # strip markdown fences if the model wraps them
        import re
        m = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", raw)
        return (m.group(1) if m else raw).strip().rstrip(";")
    except Exception:
        return None


@tool
def load_data(file_path: str = "") -> str:
    """Load a CSV/Excel file from disk. Returns shape and encoding info.

    Args:
        file_path: path to the file; '' = reuse the already-loaded
                   workspace data (no-op)
    """
    if not file_path:
        if ws.df is not None:
            return (f"OK: data already loaded ({len(ws.df)} rows x "
                    f"{len(ws.df.columns)} cols) — skipping re-load.")
        return "ERROR: no file_path given and no data loaded."
    try:
        loader = DataLoader(file_path)
        ws.df = loader.load()
        ws.data_path = file_path
        return (f"OK: loaded {len(ws.df)} rows x {len(ws.df.columns)} cols. "
                f"Columns: {list(ws.df.columns)[:20]}")
    except Exception as e:
        return f"ERROR: load failed: {e}"


@tool
def clean_data(fill_strategy: str = "auto", drop_outliers: bool = False,
               outlier_columns: str = "") -> str:
    """Clean the loaded data: dedupe, fill missing values, optionally
    remove outliers.

    Args:
        fill_strategy: 'auto' | 'median' | 'mean' | 'mode' | 'drop'
        drop_outliers: whether to apply IQR outlier removal
        outlier_columns: comma-separated column names, '' = all numeric
    """
    if ws.df is None:
        return "ERROR: no data loaded."
    try:
        cleaner = DataCleaner(ws.df)
        cleaner.drop_duplicates()
        cleaner.fill_missing(fill_strategy)
        if drop_outliers:
            cols = [c.strip() for c in outlier_columns.split(",") if c.strip()] \
                if outlier_columns else None
            cleaner.remove_outliers(cols)
        cleaner.correct_dtypes()
        ws.df_clean = cleaner.get_clean_data()
        # Quality report for the critic
        missing = int(ws.df_clean.isnull().sum().sum())
        rows, cols = ws.df_clean.shape
        return (f"OK: cleaned to {rows} rows x {cols} cols, "
                f"remaining nulls={missing}. "
                f"Strategy={fill_strategy}, outliers_dropped={drop_outliers}")
    except Exception as e:
        return f"ERROR: cleaning failed: {e}"


@tool
def engineer_features(id_column: str = "") -> str:
    """Auto-detect column types and build features: drop ID columns,
    extract time features, encode categoricals, scale numerics.

    Args:
        id_column: name of the user-ID column to drop, '' = auto-detect
    """
    df_in = ws.df_clean if ws.df_clean is not None else ws.df
    if df_in is None:
        return "ERROR: no data. Run clean_data first."
    try:
        eng = FeatureEngineer(df_in)
        eng.auto_process(id_column=id_column or None)
        ws.df_features = eng.get_features()
        return (f"OK: engineered to {ws.df_features.shape[1]} columns. "
                f"Columns: {list(ws.df_features.columns)[:25]}")
    except Exception as e:
        return f"ERROR: feature engineering failed: {e}"


@tool
def segment_users(n_clusters: int = 0, method: str = "auto",
                  reduce: str = "auto") -> str:
    """Segment users with automatic model selection.

    Args:
        n_clusters: number of segments; 0 = auto-select via elbow method
        method: 'auto' (default) picks the best of kmeans / gmm /
                hierarchical / dbscan / hdbscan by silhouette score;
                or name one of those five explicitly
        reduce: 'auto' (PCA when features > 8) | 'pca' | 'umap' | 'none'
    """
    df_in = ws.df_features if ws.df_features is not None else (
        ws.df_clean if ws.df_clean is not None else ws.df)
    if df_in is None:
        return "ERROR: no features. Run engineer_features first."
    try:
        seg = UserSegmentation(df_in)
        k = n_clusters if n_clusters and n_clusters > 0 else None
        ws.segment_labels = seg.auto_segment(n_clusters=k,
                                             method=method or "auto",
                                             reduce=reduce or "auto")
        ws.profiles = seg.get_segment_profiles()
        summary = seg.get_segment_summary()
        sil = seg.silhouette_
        return (f"OK: {len(summary)} segments. "
                f"Method={seg.method_}. "
                f"Distribution: {json.dumps(summary, ensure_ascii=False)}. "
                f"silhouette={None if sil is None else round(sil, 3)}")
    except Exception as e:
        return f"ERROR: segmentation failed: {e}"


@tool
def train_predictor(target_column: str = "", model: str = "auto") -> str:
    """Train a behavior-prediction model. 'auto' selects the best of
    LogisticRegression / RandomForest / GradientBoosting / SVC /
    XGBoost / LightGBM / CatBoost via stratified cross-validation.

    Args:
        target_column: name of the label column; '' = auto-resolve
                       (explicit column -> smallest segment as churn
                       -> auto-detect binary column)
        model: 'auto' or one of the seven model names above
    """
    df_in = ws.df_features if ws.df_features is not None else (
        ws.df_clean if ws.df_clean is not None else ws.df)
    if df_in is None:
        return "ERROR: no features to train on."
    try:
        pred = BehaviorPredictor(df_in)
        pred.auto_predict(
            target_column=target_column or None,
            segment_labels=ws.segment_labels,
            model=model or "auto",
        )
        ws.predictor = pred
        ws.importance_df = pred.get_feature_importance()
        ws.pred_metrics = pred.get_metrics()
        if not ws.pred_metrics:
            return "ERROR: no usable target found for prediction."
        cv_pool = ws.pred_metrics.get("model_cv") or {}
        return (f"OK: model={ws.pred_metrics.get('model')}, "
                f"target={ws.pred_metrics.get('target')}, "
                f"accuracy={ws.pred_metrics.get('accuracy')}, "
                f"cv={json.dumps(cv_pool, ensure_ascii=False)}")
    except Exception as e:
        return f"ERROR: prediction failed: {e}"


@tool
def plot_charts() -> str:
    """Generate all charts: segment distribution/profiles, correlation
    heatmap, feature importance, confusion matrix, prediction
    distribution, model performance card. Returns saved file paths."""
    df_in = ws.df_features if ws.df_features is not None else (
        ws.df_clean if ws.df_clean is not None else ws.df)
    if df_in is None:
        return "ERROR: no data to plot."
    try:
        viz = Visualizer(ws.output_dir)
        paths = []
        if ws.segment_labels is not None:
            p = viz.plot_segment_distribution(ws.segment_labels)
            if p:
                paths.append(p)
        if ws.profiles is not None and not ws.profiles.empty:
            p = viz.plot_segment_profiles(ws.profiles)
            if p:
                paths.append(p)
        p = viz.plot_correlation_heatmap(df_in)
        if p:
            paths.append(p)
        if ws.importance_df is not None and not ws.importance_df.empty:
            p = viz.plot_feature_importance(ws.importance_df)
            if p:
                paths.append(p)
        if ws.predictor is not None and ws.predictor.y_test_ is not None:
            p = viz.plot_confusion_matrix(
                ws.predictor.y_test_, ws.predictor.y_pred_)
            if p:
                paths.append(p)
            if ws.predictor.y_proba_ is not None:
                p = viz.plot_prediction_distribution(
                    ws.predictor.y_test_, ws.predictor.y_proba_,
                    target_name=ws.predictor._target_col)
                if p:
                    paths.append(p)
            p = viz.plot_model_performance(ws.predictor.get_metrics())
            if p:
                paths.append(p)
        ws.chart_paths = paths
        return f"OK: {len(paths)} charts saved to {ws.output_dir}/"
    except Exception as e:
        return f"ERROR: plotting failed: {e}"


@tool
def generate_report() -> str:
    """Generate Excel + HTML reports from current analysis results."""
    if ws.df is None:
        return "ERROR: no analysis to report."
    try:
        reporter = ReportGenerator(ws.output_dir)
        seg_summary = {}
        if ws.profiles is not None:
            for _, row in ws.profiles.iterrows():
                label = row.get("segment", row.name)
                seg_summary[label] = {
                    "count": int(row["count"]),
                    "pct": round(float(row["pct"]), 1),
                }
        report_summary = {
            "Data": f"{len(ws.df)} rows, {len(ws.df.columns)} cols",
            "Segments": seg_summary,
            "Model": ws.pred_metrics,
        }
        reporter.print_console_summary(report_summary)
        excel = reporter.generate_excel(
            segments_df=ws.profiles, profiles_df=ws.profiles,
            importance_df=ws.importance_df, metrics=ws.pred_metrics,
        )
        html = reporter.generate_html_summary(
            summary=report_summary, chart_paths=ws.chart_paths,
        )
        return f"OK: report written. Excel={excel}, HTML={html}"
    except Exception as e:
        return f"ERROR: report generation failed: {e}"


@tool
def quality_check() -> str:
    """Compute data-quality metrics of the current working DataFrame:
    missing rate, constant columns, high-cardinality columns,
    numeric column count. The critic uses this to decide retry."""
    df = ws.df_clean if ws.df_clean is not None else ws.df
    if df is None:
        return "ERROR: no data to check."
    rows = max(len(df), 1)
    total_missing = int(df.isnull().sum().sum())
    cells = rows * max(len(df.columns), 1)
    high_missing = [
        c for c in df.columns if df[c].isnull().mean() > 0.7]
    constant = [c for c in df.columns if df[c].nunique() <= 1]
    numeric = df.select_dtypes(include=[np.number]).shape[1]
    return json.dumps({
        "rows": int(rows),
        "columns": int(len(df.columns)),
        "missing_rate": round(total_missing / cells, 4),
        "high_missing_columns": high_missing,
        "constant_columns": constant,
        "numeric_columns": numeric,
    }, ensure_ascii=False)


# Registry used by the Executor to bind tools to the LLM
ALL_TOOLS = [
    load_data,
    inspect_schema,
    query_data,
    clean_data,
    engineer_features,
    segment_users,
    train_predictor,
    plot_charts,
    generate_report,
    quality_check,
]

TOOL_BY_NAME = {t.name: t for t in ALL_TOOLS}
