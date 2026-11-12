"""
UserInsight Agent — Main Entry Point

LangGraph-based autonomous analysis agent (Plan-and-Execute with a
reflection loop):

    Planner → Executor ⇄ Critic → Summarizer

- Planner decomposes the natural-language goal into a tool plan,
  enriched by long-term memory (ChromaDB).
- Executor picks tools/args via Function Calling and runs them.
- Critic evaluates each result; failures trigger retries (the loop).
- Summarizer produces the final business report and writes memory.

Every LLM node has a deterministic fallback, so the graph also runs
without an API key (static plan + rule-based critic).
"""

import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agent_graph import run_agent
from agent_tools import ws


# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        print(f"Config not found: {config_path}, using defaults.")
        return _default_config()
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _default_config() -> dict:
    """Return a minimal default config."""
    return {
        "data": {"file_path": "data/sample_user_data.csv"},
        "output": {"dir": "output"},
        "llm": {"enabled": True},
        "agent": {"max_iterations": 15},
    }


# ═══════════════════════════════════════════════════════════════
#  Main pipeline (also importable as main_pipeline())
# ═══════════════════════════════════════════════════════════════

def main_pipeline(
    file_path: str = None,
    api_key: str = None,
    goal: str = None,
    max_iterations: int = None,
) -> dict:
    """Run the agent graph on a dataset and return structured results.

    Args:
        file_path : CSV/Excel path; defaults to config data.file_path
        api_key   : LLM API key; defaults to DEEPSEEK_API_KEY env
        goal      : natural-language analysis goal; defaults to the
                    standard profiling + churn-prediction goal
        max_iterations : cap on planner/executor/critic loop turns

    Returns a dict with the same keys as the legacy pipeline:
        profiles, segment_labels, importance_df, chart_paths,
        pred_metrics, llm_insight, output_dir — plus agent extras:
        plan, tool_results, error, done
    """
    config = load_config()
    output_dir = config.get("output", {}).get("dir", "output")
    llm_enabled = config.get("llm", {}).get("enabled", True)
    data_path = file_path or config.get("data", {}).get(
        "file_path", "data/sample_user_data.csv")
    if max_iterations is None:
        max_iterations = config.get("agent", {}).get("max_iterations", 15)

    if goal is None:
        goal = ("对这批用户数据进行完整分析：诊断数据质量并清洗，"
                "构建特征，进行用户分群画像，训练流失/行为预测模型，"
                "输出图表、报告与业务洞察。")

    print("=" * 60)
    print("  UserInsight Agent — LangGraph 自主分析 Agent")
    print(f"  Goal: {goal[:40]}...")
    print(f"  Data: {data_path}")
    print(f"  Agent mode: {'LLM 决策' if llm_enabled else '确定性降级'}")
    print("=" * 60)

    if not llm_enabled:
        api_key = None

    # Reset the shared tool workspace for a clean run
    ws.reset(data_path=data_path, output_dir=output_dir)

    result = None
    try:
        result = run_agent(
            goal=goal,
            data_path=data_path,
            api_key=api_key,
            max_iterations=max_iterations,
        )
    except Exception as e:
        print(f"  ERROR in agent run: {e}")
        return {"error": str(e), "done": False}

    if result.get("error"):
        return result

    # ── Normalize to the legacy result shape ──
    report = result.get("report") or {}
    seg_summary = {}
    profiles = report.get("profiles")
    if profiles is not None and not profiles.empty:
        for _, row in profiles.iterrows():
            label = row.get("segment", row.name)
            seg_summary[label] = {
                "count": int(row["count"]),
                "pct": round(float(row["pct"]), 1),
            }
    out = {
        "profiles": profiles,
        "segment_labels": report.get("segment_labels"),
        "importance_df": report.get("importance_df"),
        "chart_paths": report.get("chart_paths", []),
        "pred_metrics": report.get("pred_metrics", {}),
        "seg_summary": seg_summary,
        "llm_insight": report.get("llm_insight"),
        "output_dir": output_dir,
        # Agent extras
        "plan": result.get("plan", []),
        "tool_results": result.get("tool_results", {}),
        "done": result.get("done", False),
        "error": result.get("error"),
    }

    print("\n" + "=" * 60)
    print("  Agent 分析完成")
    if out["chart_paths"]:
        print(f"  图表: {len(out['chart_paths'])} 张 → {output_dir}/")
    if out["llm_insight"]:
        print(f"  Agent 洞察: {out['llm_insight'].get('headline', '')[:60]}")
    print("=" * 60)
    return out


def main():
    """CLI entry point."""
    result = main_pipeline()
    if result.get("error"):
        print(f"\n  FAILED: {result['error']}")


if __name__ == "__main__":
    main()
