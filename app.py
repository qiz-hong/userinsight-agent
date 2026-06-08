"""
UserInsight Agent — Streamlit Web Interface

Run:  streamlit run app.py

Provides an interactive UI where users can:
- Upload a CSV file
- Configure analysis parameters
- See the agent's decision process in real time
- View charts, tables, and LLM-generated insights
- Download the full Excel report
"""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main import main_pipeline

# ── Page Config ───────────────────────────────────────────────

st.set_page_config(
    page_title="UserInsight Agent",
    page_icon="🧠",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 UserInsight Agent")
    st.caption("Intelligent User Profiling & Behavior Prediction")

    st.markdown("---")
    st.subheader("🔑 API Key")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        value=os.environ.get("DEEPSEEK_API_KEY", ""),
        help="Required for Agent decision layer. Get one at platform.deepseek.com",
        placeholder="sk-...",
    )
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    st.markdown("---")
    st.subheader("⚙️ Parameters")
    n_clusters = st.slider("Max User Segments", 2, 8, 4,
                           help="Agent may auto-adjust based on data")
    clean_strategy = st.selectbox(
        "Cleaning Strategy",
        ["auto", "median", "mean", "mode", "drop"],
        help="'auto' lets the Agent decide per column"
    )

    st.markdown("---")
    st.caption("Built with Streamlit + DeepSeek LLM")

# ── Main Area ────────────────────────────────────────────────

st.title("UserInsight Agent")
st.markdown(
    "Drop a CSV of user behavior data — the Agent will **diagnose**, "
    "**plan**, and **analyze** it automatically."
)

uploaded = st.file_uploader(
    "Upload user data (CSV)",
    type="csv",
    help="Recommended: user_id, demographics, activity dates, behavior counts, labels",
)

if uploaded:
    # Save to temp file
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".csv", mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(uploaded.getvalue().decode("utf-8"))
        tmp_path = tmp.name

    # Preview
    preview = pd.read_csv(tmp_path)
    st.subheader("📋 Data Preview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(preview))
    col2.metric("Columns", len(preview.columns))
    col3.metric("Missing", preview.isnull().sum().sum())
    st.dataframe(preview.head(5), use_container_width=True)

    # Run button
    if st.button("🚀 Run Agent Analysis", type="primary", use_container_width=True):
        with st.spinner("Agent is analyzing your data... This may take 1-2 minutes."):
            # Create a log capture area
            log_placeholder = st.empty()

            # Redirect prints to capture logs
            import io
            old_stdout = sys.stdout
            sys.stdout = log_stream = io.StringIO()

            try:
                result = main_pipeline(file_path=tmp_path, api_key=api_key)
            finally:
                sys.stdout = old_stdout

            logs = log_stream.getvalue()

        # ── Tabs for results ─────────────────────────────────
        if result.get("error"):
            st.error(f"Analysis failed: {result['error']}")
        else:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "🧠 Agent Insights", "👥 Segments", "🎯 Prediction",
                "📊 Charts", "🔍 Features", "📝 Logs"
            ])

            # Tab 1: Agent Insights
            with tab1:
                llm = result.get("llm_insight")
                if llm:
                    st.success(f"**{llm.get('headline', '')}**")
                    st.metric("Confidence", llm.get("confidence", "N/A"))

                    st.subheader("🔍 Top Findings")
                    for i, f in enumerate(llm.get("top_findings", []), 1):
                        st.info(f"**{i}.** {f}")

                    st.subheader("💡 Recommendations")
                    for i, r in enumerate(llm.get("recommendations", []), 1):
                        st.warning(f"**{i}.** {r}")

                    risk = llm.get("risk_alert")
                    if risk:
                        st.error(f"⚠️ Risk Alert: {risk}")
                else:
                    st.info(
                        "Agent insights require a DeepSeek API key. "
                        "Set it in the sidebar to unlock AI-powered analysis."
                    )
                    # Show raw numbers instead
                    st.subheader("Raw Summary")
                    st.json({
                        "segments": result.get("seg_summary", {}),
                        "model": result.get("pred_metrics", {}),
                    })

            # Tab 2: Segments
            with tab2:
                profiles = result.get("profiles")
                if profiles is not None and not profiles.empty:
                    st.subheader("Segment Profiles")
                    st.dataframe(profiles, use_container_width=True)

                    st.subheader("Segment Summary")
                    seg = result.get("seg_summary", {})
                    cols = st.columns(len(seg) if seg else 1)
                    for i, (k, v) in enumerate(seg.items()):
                        with cols[i % len(cols)]:
                            st.metric(
                                k,
                                f"{v.get('count', 0)} users",
                                f"{v.get('pct', 0)}%",
                            )
                else:
                    st.info("Segmentation not available.")

            # Tab 3: Prediction
            with tab3:
                pred_metrics = result.get("pred_metrics", {})
                if pred_metrics:
                    st.subheader("🎯 What is Being Predicted?")
                    target = pred_metrics.get("target", "Unknown")
                    classes = pred_metrics.get("classes", [])
                    st.markdown(
                        f"**Target variable:** `{target}` "
                        f"(predicting whether a user belongs to the at-risk segment)"
                    )

                    st.subheader("📈 Model Performance")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Model", pred_metrics.get("model", "N/A"))
                    col_b.metric("Accuracy", f"{pred_metrics.get('accuracy', 0):.2%}")
                    col_c.metric("CV Mean", f"{pred_metrics.get('cv_mean', 0):.2%}")

                    st.subheader("📊 Prediction Charts")
                    # Show prediction-specific charts
                    charts = result.get("chart_paths", [])
                    pred_chart_names = [
                        "confusion_matrix", "prediction_distribution",
                        "model_performance",
                    ]
                    pred_shown = False
                    for chart in charts:
                        for name in pred_chart_names:
                            if name in chart and os.path.exists(chart):
                                st.image(chart, use_container_width=True)
                                pred_shown = True
                                break
                    if not pred_shown:
                        st.info("Prediction charts are included in the Charts tab.")
                else:
                    st.info(
                        "No prediction was performed. The predictor needs a target column, "
                        "segment labels, or a binary column in the data."
                    )

            # Tab 4: Charts
            with tab4:
                charts = result.get("chart_paths", [])
                if charts:
                    for chart in charts:
                        if os.path.exists(chart):
                            st.image(chart, use_container_width=True)
                else:
                    st.info("No charts generated.")

            # Tab 5: Features
            with tab5:
                imp = result.get("importance_df")
                if imp is not None and not imp.empty:
                    st.subheader("Feature Importance")
                    st.dataframe(imp, use_container_width=True)
                    st.bar_chart(
                        imp.set_index("feature")["importance"],
                        use_container_width=True,
                    )
                else:
                    st.info("Feature importance not available.")

            # Tab 6: Logs
            with tab6:
                st.code(logs, language="text")

            # ── Download ──────────────────────────────────────
            report_path = Path(result.get("output_dir", "output")) / "analysis_report.xlsx"
            if report_path.exists():
                st.download_button(
                    "📥 Download Full Report (Excel)",
                    data=report_path.read_bytes(),
                    file_name="analysis_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

    # Cleanup
    try:
        os.unlink(tmp_path)
    except Exception:
        pass
