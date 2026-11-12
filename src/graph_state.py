"""
Agent State Definition

Shared state object passed between LangGraph nodes.
All values are optional — nodes must handle missing keys gracefully.
"""

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """State flowing through the agent graph.

    Keys:
      messages        : conversation history (LLM messages)
      goal            : user's natural-language analysis goal
      data_path       : input data file path
      schema_summary  : text summary of the loaded DataFrame
      df              : current working DataFrame (mutated by tools)
      plan            : list of steps, each {tool, args, criteria, status}
      current_step    : index of the step being executed
      tool_results    : dict of tool name -> result summary
      quality_report  : latest quality metrics computed by tools
      critic_verdict  : last critic decision (pass / retry / adjust)
      retry_count     : number of retries for the current step
      iteration       : global iteration counter (guardrail)
      max_iterations  : hard cap on total graph iterations
      error           : fatal error message, if any
      done            : set when the final report is produced
      report          : final structured results (for UI / files)
      _api_key        : LLM API key override for this run
      df_loaded       : True once data is loaded into the workspace
      critic_reason   : reason text from the last critic verdict
      critic_new_args : corrected args proposed by the critic on retry
    """

    messages: Annotated[list, add_messages]
    goal: str
    data_path: str
    schema_summary: str
    df: object
    plan: list
    current_step: int
    tool_results: dict
    quality_report: dict
    critic_verdict: str
    retry_count: int
    iteration: int
    max_iterations: int
    error: str
    done: bool
    report: dict
    _api_key: str
    df_loaded: bool
    critic_reason: str
    critic_new_args: dict
