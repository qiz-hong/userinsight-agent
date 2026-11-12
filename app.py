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
    st.caption("自主规划式用户智能分析 Agent · LangGraph")

    st.markdown("---")
    st.subheader("🔑 API Key")
    api_key = st.text_input(
        "DeepSeek API Key",
        type="password",
        value=os.environ.get("DEEPSEEK_API_KEY", ""),
        help="Agent 决策层必需。无 Key 时自动降级为确定性流水线",
        placeholder="sk-...",
    )
    if api_key:
        os.environ["DEEPSEEK_API_KEY"] = api_key

    st.markdown("---")
    st.subheader("⚙️ 参数")
    max_iterations = st.slider(
        "最大迭代轮数 (护栏)", 5, 30, 15,
        help="限制 Planner→Executor⇄Critic 循环总轮数，防止失控",
    )
    st.caption("其余参数由 Agent 自主决定")

    st.markdown("---")
    st.caption("Built with LangGraph + DeepSeek LLM + ChromaDB")

# ── Main Area ────────────────────────────────────────────────

st.title("UserInsight Agent")
st.markdown(
    "基于 **LangGraph** 的自主规划式分析 Agent：Planner 分解任务 → "
    "Executor 自主调工具 → Critic 反思重试 → Summarizer 生成业务报告。"
)

uploaded = st.file_uploader(
    "上传用户数据 (CSV)",
    type="csv",
    help="推荐：user_id、画像字段、行为计数、日期、标签。脏数据（缺失/乱码/异常列名）可自动处理",
)

goal = st.text_input(
    "分析目标（自然语言，Agent 据此自主规划）",
    value="对用户数据完整分析：诊断清洗、特征构建、用户分群画像、流失预测、图表报告与业务洞察",
    help="也可以多轮追问，例如：重点找出高价值流失风险用户",
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
    st.subheader("📋 数据预览")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(preview))
    col2.metric("Columns", len(preview.columns))
    col3.metric("Missing", preview.isnull().sum().sum())
    st.dataframe(preview.head(5), use_container_width=True)

    # Run button
    if st.button("🚀 运行 Agent 分析", type="primary", use_container_width=True):
        with st.spinner("Agent 正在规划、执行、反思... 约 1-2 分钟"):
            import io
            old_stdout = sys.stdout
            sys.stdout = log_stream = io.StringIO()

            try:
                result = main_pipeline(
                    file_path=tmp_path,
                    api_key=api_key,
                    goal=goal,
                    max_iterations=max_iterations,
                )
            finally:
                sys.stdout = old_stdout

            logs = log_stream.getvalue()

        # ── Tabs for results ─────────────────────────────────
        if result.get("error"):
            st.error(f"分析失败: {result['error']}")
        else:
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "🧠 Agent 洞察", "📋 Agent 执行轨迹", "👥 分群",
                "🎯 预测", "📊 图表", "🔍 特征", "📝 日志",
            ])

            # Tab 1: Agent 洞察
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
                        "Agent 洞察需要 DeepSeek API Key。无 Key 时"
                        "自动降级为确定性流水线，下方展示原始结果。"
                    )
                    st.subheader("Raw Summary")
                    st.json({
                        "segments": result.get("seg_summary", {}),
                        "model": result.get("pred_metrics", {}),
                    })

            # Tab 2: Agent 执行轨迹
            with tab2:
                plan = result.get("plan", [])
                if plan:
                    st.subheader("📋 Agent 生成的任务计划")
                    plan_rows = []
                    for i, s in enumerate(plan, 1):
                        plan_rows.append({
                            "Step": i,
                            "Tool": s.get("tool", "?"),
                            "Args": str(s.get("args", {})),
                            "Criteria": s.get("criteria", ""),
                            "Status": s.get("status", "?"),
                        })
                    st.dataframe(pd.DataFrame(plan_rows),
                                 use_container_width=True)
                else:
                    st.info("无任务计划（可能执行了降级流水线）。")

                st.subheader("🛠️ 工具执行结果")
                results = result.get("tool_results", {})
                for name, text in results.items():
                    with st.expander(f"{name}"):
                        st.text(str(text)[:600])

                st.caption(
                    "Planner 分解任务 → Executor 自主调工具 → "
                    "Critic 评估不达标自动重试（最多 3 次/步）"
                )

            # Tab 3: 分群
            with tab3:
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

            # Tab 4: 预测
            with tab4:
                pred_metrics = result.get("pred_metrics", {})
                if pred_metrics:
                    st.subheader("🎯 预测目标")
                    target = pred_metrics.get("target", "Unknown")
                    st.markdown(
                        f"**目标变量:** `{target}` "
                        f"（预测用户是否属于风险流失分群）"
                    )

                    st.subheader("📈 模型性能")
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Model", pred_metrics.get("model", "N/A"))
                    col_b.metric("Accuracy", f"{pred_metrics.get('accuracy', 0):.2%}")
                    col_c.metric("CV Mean", f"{pred_metrics.get('cv_mean', 0):.2%}")

                    st.subheader("📊 预测图表")
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
                        st.info("预测图表见「图表」Tab。")
                else:
                    st.info(
                        "未执行预测。需要目标列、分群标签或数据中的二值列。"
                    )

            # Tab 5: 图表
            with tab5:
                charts = result.get("chart_paths", [])
                if charts:
                    for chart in charts:
                        if os.path.exists(chart):
                            st.image(chart, use_container_width=True)
                else:
                    st.info("No charts generated.")

            # Tab 6: 特征
            with tab6:
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

            # Tab 7: 日志
            with tab7:
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
