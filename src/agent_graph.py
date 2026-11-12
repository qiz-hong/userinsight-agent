"""
Agent Graph (LangGraph)

Plan-and-Execute agent with a reflection loop:

    START → planner → executor ⇄ critic → summarize → END

- planner    : LLM decomposes the natural-language goal into a step plan
               (tool, args, success criteria), enriched by long-term memory.
- executor   : LLM selects tools/args via Function Calling and runs them;
               results are recorded in shared state.
- critic     : LLM evaluates each step result against its criteria plus
               data-quality metrics; verdict = pass | retry | adjust.
- summarize  : generates the final business report, writes memory.

Guardrails: max_iterations caps total loop turns; retry budget caps
per-step retries; every LLM node has a deterministic fallback so the
graph terminates even if the LLM misbehaves.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from langgraph.graph import StateGraph, START, END

from graph_state import AgentState
import agent_tools as tools
from agent_tools import ws, TOOL_BY_NAME, ALL_TOOLS
from memory_store import get_memory_store


# ── LLM factory (lazy, shared) ─────────────────────────────────

_llm = None
_llm_api_key = None
_no_key_warned = False


def _get_llm(api_key: str = None, model: str = None, temperature: float = 0.3):
    """Create/reuse a ChatOpenAI instance pointed at DeepSeek (or any
    OpenAI-compatible endpoint via LLM_BASE_URL).

    Returns None when no API key is available so callers degrade to
    deterministic fallbacks.
    """
    global _llm, _llm_api_key, _no_key_warned
    if _llm is not None and api_key in (None, _llm_api_key):
        return _llm
    try:
        from langchain_openai import ChatOpenAI
        import os
        _llm_api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") \
            or os.environ.get("OPENAI_API_KEY")
        if not _llm_api_key:
            if not _no_key_warned:
                _no_key_warned = True
                print("  [Agent] No API key — LLM nodes degrade to "
                      "deterministic fallback.")
            _llm = None
            return None
        _llm = ChatOpenAI(
            base_url=os.environ.get("LLM_BASE_URL",
                                    "https://api.deepseek.com/v1"),
            api_key=_llm_api_key,
            model=model or os.environ.get("LLM_MODEL", "deepseek-chat"),
            temperature=temperature,
            max_tokens=1024,
            timeout=60,
        )
        return _llm
    except ImportError:
        _llm = None
        return None
    except Exception as e:
        print(f"  [Agent] LLM init failed ({e}); deterministic fallback.")
        _llm = None
        return None


def _llm_json(llm, system: str, user: str) -> dict | None:
    """Call LLM and extract a JSON object from its reply."""
    if llm is None:
        return None
    try:
        msg = llm.invoke(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}]
        )
        raw = msg.content
        return _extract_json(raw)
    except Exception as e:
        print(f"  [Agent] LLM call failed: {e}")
        return None


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    print(f"  [Agent] Could not parse JSON from: {raw[:200]}...")
    return None


# ═══════════════════════════════════════════════════════════════
#  Fallback plan (used when LLM unavailable)
# ═══════════════════════════════════════════════════════════════

STATIC_PLAN = [
    {"tool": "load_data", "args": {}, "criteria": "data loaded",
     "status": "pending"},
    {"tool": "clean_data", "args": {"fill_strategy": "auto"},
     "criteria": "no remaining nulls", "status": "pending"},
    {"tool": "engineer_features", "args": {},
     "criteria": "features built", "status": "pending"},
    {"tool": "segment_users", "args": {"n_clusters": 0},
     "criteria": "segments formed", "status": "pending"},
    {"tool": "train_predictor", "args": {},
     "criteria": "model trained", "status": "pending"},
    {"tool": "plot_charts", "args": {},
     "criteria": "charts saved", "status": "pending"},
    {"tool": "generate_report", "args": {},
     "criteria": "report written", "status": "pending"},
]


# ═══════════════════════════════════════════════════════════════
#  Nodes
# ═══════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """You are the Planner of a data-analysis agent.
Given the user's goal and the dataset schema, produce a JSON plan.
Rules:
- Output ONLY JSON: {"plan": [{"tool": str, "args": dict, "criteria": str}]}
- Available tools and their args:
  load_data(file_path), inspect_schema(), clean_data(fill_strategy, drop_outliers, outlier_columns),
  engineer_features(id_column), segment_users(n_clusters), train_predictor(target_column),
  plot_charts(), generate_report(), quality_check()
- Plan only the tools needed to satisfy the goal, in dependency order.
- clean_data should come after load_data; engineer_features after clean_data;
  segment_users/train_predictor after engineer_features;
  plot_charts/generate_report near the end.
- criteria: one measurable sentence describing success for that step."""


def planner_node(state: AgentState) -> dict:
    print("\n  [Planner] Decomposing goal into execution plan...")
    goal = state.get("goal", "")
    schema = state.get("schema_summary", "")
    llm = _get_llm(state.get("_api_key"))

    memory = get_memory_store().context_for(
        f"{goal} {schema}", kind="strategy", k=2)

    plan = None
    if llm is not None:
        user = (f"Goal: {goal}\n\nDataset schema:\n{schema}\n\n"
                + (f"Relevant memory from past analyses:\n{memory}\n\n"
                   if memory else "")
                + "Create the plan now (JSON only).")
        data = _llm_json(llm, PLANNER_SYSTEM, user)
        if data:
            plan = data.get("plan") or None

    if not plan:
        plan = [dict(s) for s in STATIC_PLAN]
        # load_data already happened; drop it and start at clean
        if state.get("df_loaded"):
            plan = plan[1:]
        print("  [Planner] LLM unavailable — using static fallback plan.")

    for step in plan:
        step["status"] = "pending"
        if "tool" not in step or step["tool"] not in TOOL_BY_NAME:
            step["status"] = "skipped"

    print(f"  [Planner] Plan: {' -> '.join(s['tool'] for s in plan)}")
    return {"plan": plan, "current_step": 0, "retry_count": 0}


EXECUTOR_SYSTEM = """You are the Executor of a data-analysis agent.
Pick ONE tool and its arguments to accomplish the current step.
Answer ONLY with a tool call. If the step is already satisfied,
call quality_check() to confirm instead."""


def executor_node(state: AgentState) -> dict:
    plan = state.get("plan", [])
    i = state.get("current_step", 0)
    step = plan[i] if i < len(plan) else None
    if step is None:
        return {}

    tool_name = step["tool"]
    print(f"\n  [Executor] Step {i+1}/{len(plan)}: {tool_name}")

    # Step args: plan defaults, optionally corrected by the Critic
    # after a failed attempt (reflection feedback loop).
    step_args = dict(step.get("args", {}) or {})
    if state.get("critic_verdict") in ("retry", "adjust"):
        step_args.update(state.get("critic_new_args") or {})
        if step_args != step.get("args"):
            print(f"  [Executor] Retry with critic-corrected args: {step_args}")

    llm = _get_llm(state.get("_api_key"))
    result_text = None

    # ── LLM function-calling path ──
    if llm is not None:
        try:
            llm_tools = llm.bind_tools(ALL_TOOLS)
            goal = state.get("goal", "")
            history = state.get("messages", [])
            msg = llm_tools.invoke(
                [{"role": "system", "content": EXECUTOR_SYSTEM},
                 {"role": "user",
                  "content": (f"Goal: {goal}\nCurrent step: {tool_name}\n"
                              f"Step criteria: {step.get('criteria','')}\n"
                              f"Previous results: {json.dumps(state.get('tool_results', {}), ensure_ascii=False)[:1500]}")}]
                + list(history[-4:])
            )
            if getattr(msg, "tool_calls", None):
                for call in msg.tool_calls[:1]:
                    fn = TOOL_BY_NAME.get(call["name"])
                    if fn is None:
                        continue
                    args = call.get("args", {})
                    print(f"  [Executor] LLM chose {call['name']}{args}")
                    result_text = _safe_run(fn, args)
            if result_text is None:
                # LLM returned no tool call — run the planned tool directly
                result_text = _safe_run(TOOL_BY_NAME[tool_name], step_args)
        except Exception as e:
            print(f"  [Executor] LLM dispatch failed ({e}); deterministic run")
            result_text = _safe_run(TOOL_BY_NAME[tool_name], step_args)

    # ── Deterministic path (no LLM) ──
    if result_text is None:
        result_text = _safe_run(TOOL_BY_NAME[tool_name], step_args)

    print(f"  [Executor] Result: {str(result_text)[:200]}")
    step["status"] = "done" if not str(result_text).startswith("ERROR") else "failed"
    results = dict(state.get("tool_results", {}))
    results[tool_name] = result_text

    return {
        "tool_results": results,
        "plan": plan,
        "iteration": state.get("iteration", 0) + 1,
    }


def _safe_run(fn, args: dict) -> str:
    """Run a tool with argument validation; never raises."""
    try:
        return fn.invoke(args or {})
    except TypeError:
        # LLM passed bad args — retry with empty args
        try:
            return fn.invoke({})
        except Exception as e:
            return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


CRITIC_SYSTEM = """You are the Critic of a data-analysis agent.
Evaluate the latest tool result against its success criteria.
Output ONLY JSON: {"verdict": "pass"|"retry"|"adjust",
"reason": str, "new_args": dict}.
- pass: result satisfies criteria
- retry: same tool should be retried (include corrected args in new_args)
- adjust: tool is wrong, but never return this if pass/retry suffice."""


def critic_node(state: AgentState) -> dict:
    plan = state.get("plan", [])
    i = state.get("current_step", 0)
    step = plan[i] if i < len(plan) else None
    if step is None:
        return {"critic_verdict": "pass"}

    results = state.get("tool_results", {})
    result_text = results.get(step["tool"], "")
    quality = ""
    try:
        quality = tools.quality_check.invoke({})
    except Exception:
        pass

    # Budget exhausted → force pass to guarantee termination
    if state.get("iteration", 0) >= state.get("max_iterations", 12):
        print("  [Critic] Budget exhausted — forcing pass.")
        return {"critic_verdict": "pass"}

    # Hard failure without LLM → one deterministic retry then pass
    if str(result_text).startswith("ERROR"):
        retries = state.get("retry_count", 0)
        if retries >= 3:
            print("  [Critic] Step failed repeatedly — skipping.")
            return {"critic_verdict": "pass"}
        print(f"  [Critic] Step failed (retry {retries+1}/3).")
        return {"critic_verdict": "retry",
                "retry_count": retries + 1}

    llm = _get_llm(state.get("_api_key"))
    verdict = None
    new_args = {}
    if llm is not None:
        user = (f"Goal: {state.get('goal','')}\n"
                f"Step: {step['tool']}\nCriteria: {step.get('criteria','')}\n"
                f"Result: {str(result_text)[:800]}\n"
                f"Data quality: {quality}\n"
                "Verdict (JSON only):")
        data = _llm_json(llm, CRITIC_SYSTEM, user)
        if data:
            verdict = data.get("verdict")
            new_args = data.get("new_args") or {}

    if verdict not in ("pass", "retry", "adjust"):
        verdict = "pass"

    if verdict in ("retry", "adjust") and state.get("retry_count", 0) >= 3:
        verdict = "pass"  # retry budget guardrail

    if verdict == "pass":
        print("  [Critic] PASS — step accepted.")
        return {"critic_verdict": "pass"}
    retries = state.get("retry_count", 0) + 1
    print(f"  [Critic] {verdict.upper()} — retry {retries}/3"
          + (f" with corrected args {new_args}" if new_args else ""))
    return {"critic_verdict": verdict, "retry_count": retries,
            "critic_reason": "", "critic_new_args": new_args}


# ── Routers ────────────────────────────────────────────────────

def route_after_critic(state: AgentState) -> str:
    plan = state.get("plan", [])
    i = state.get("current_step", 0)

    if state.get("critic_verdict") in ("retry", "adjust"):
        return "executor"

    # Step accepted — advance
    next_i = i + 1
    if next_i >= len(plan):
        return "summarize"
    return "advance"


def advance_step(state: AgentState) -> dict:
    """Plain node: move to the next plan step."""
    return {"current_step": state.get("current_step", 0) + 1,
            "retry_count": 0, "critic_new_args": {}}


SUMMARIZER_SYSTEM = """You are the business-insight writer of a data-
analysis agent. Given the tool results, write a concise business
summary. Output ONLY JSON:
{"headline": str, "top_findings": [str x3], "recommendations": [str x2],
 "risk_alert": str, "confidence": float}
Write in the language of the goal if possible."""


def summarize_node(state: AgentState) -> dict:
    print("\n  [Summarizer] Compiling final results...")
    results = state.get("tool_results", {})
    goal = state.get("goal", "")

    llm = _get_llm(state.get("_api_key"))
    insight = None
    if llm is not None:
        data = _llm_json(
            llm, SUMMARIZER_SYSTEM,
            f"Goal: {goal}\n\nTool results:\n{json.dumps(results, ensure_ascii=False)[:3000]}\n\n"
            "Summarize now (JSON only).")
        if data:
            insight = data
            print(f"  [Summarizer] {data.get('headline', '')[:100]}")

    # Produce the report file unless the plan already generated it
    if "generate_report" in results:
        report_msg = "Report already generated during execution."
    else:
        report_msg = tools.generate_report.invoke({})

    # ── Write long-term memory ──
    mem = get_memory_store()
    try:
        mem.remember("schema",
                     f"Dataset {ws.data_path or 'uploaded'}: "
                     f"{tools.inspect_schema.invoke({})}")
        if results.get("clean_data"):
            mem.remember("strategy",
                         f"Goal '{goal[:80]}': cleaning result "
                         f"{results['clean_data'][:200]}")
        if results.get("train_predictor"):
            mem.remember("strategy",
                         f"Goal '{goal[:80]}': prediction result "
                         f"{results['train_predictor'][:200]}")
        if insight:
            mem.remember("insight",
                         f"Goal '{goal[:80]}': {insight.get('headline','')}")
        print("  [Memory] Session memories stored.")
    except Exception as e:
        print(f"  [Memory] write failed: {e}")

    return {
        "done": True,
        "critic_verdict": "pass",
        "report": {
            "profiles": ws.profiles,
            "segment_labels": ws.segment_labels,
            "importance_df": ws.importance_df,
            "chart_paths": ws.chart_paths,
            "pred_metrics": ws.pred_metrics,
            "llm_insight": insight,
            "tool_results": results,
            "report_message": report_msg,
        },
    }


# ═══════════════════════════════════════════════════════════════
#  Graph assembly
# ═══════════════════════════════════════════════════════════════

def build_graph() -> "StateGraph":
    """Assemble the agent graph."""
    g = StateGraph(AgentState)

    g.add_node("planner", planner_node)
    g.add_node("executor", executor_node)
    g.add_node("critic", critic_node)
    g.add_node("advance", advance_step)
    g.add_node("summarize", summarize_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "executor")
    g.add_edge("executor", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "executor": "executor",   # retry loop (reflection)
            "summarize": "summarize",  # all steps done
            "advance": "advance",      # next step
        },
    )
    g.add_edge("advance", "executor")
    g.add_edge("summarize", END)

    return g.compile()


def run_agent(goal: str, data_path: str, api_key: str = None,
              max_iterations: int = 15) -> dict:
    """Run the full agent graph on a dataset.

    Loads the data into the tool workspace, builds the schema
    summary, then executes planner → executor ⇄ critic → summarize.
    """
    # ── Pre-load data so planner can see the schema ──
    print("  [Agent] Loading data into workspace...")
    load_msg = tools.load_data.invoke({"file_path": data_path})
    print(f"  [Agent] {load_msg}")
    if load_msg.startswith("ERROR"):
        return {"error": load_msg, "done": False}

    schema = tools.inspect_schema.invoke({})

    graph = build_graph()
    initial: AgentState = {
        "goal": goal,
        "data_path": data_path,
        "schema_summary": str(schema),
        "messages": [],
        "tool_results": {"load_data": load_msg,
                         "inspect_schema": schema},
        "plan": [],
        "current_step": 0,
        "retry_count": 0,
        "iteration": 0,
        "max_iterations": max_iterations,
        "done": False,
        "df_loaded": True,
        "_api_key": api_key,
    }
    final = graph.invoke(initial)
    return final
