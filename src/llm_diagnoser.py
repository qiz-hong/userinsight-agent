"""
LLM Data Diagnoser — Intelligent data quality assessment.

Given a data summary (column types, null counts, sample values),
uses LLM to suggest optimal cleaning strategies.

Part of the Agent layer: replaces hard-coded "auto" strategies
with context-aware LLM decisions.
"""

import json
from llm_client import get_client


SYSTEM_PROMPT = (
    "You are a senior data analyst. Given a structured summary of a dataset, "
    "diagnose data quality issues and output a JSON cleaning plan. "
    "Be concise. Only output valid JSON, no explanation."
)

USER_PROMPT_TEMPLATE = """Analyze this dataset summary and output a cleaning plan:

=== Column Summary ===
{column_summary}

=== Null Report ===
{null_report}

=== Sample Rows (first 5) ===
{sample_rows}

Output a JSON object with exactly these keys:
{{
  "issues_found": ["brief issue 1", "brief issue 2"],
  "fill_strategy": {{"column_name": "median"|"mean"|"mode"|"drop"}},
  "outlier_columns": ["col1", "col2"],
  "id_columns": ["col1", "col2"],
  "overall_quality": "good"|"fair"|"poor"
}}

Rules:
- fill_strategy: only include columns with nulls. Use "median" for numeric, "mode" for categorical, "drop" if >50% null
- outlier_columns: columns likely to have extreme outliers
- id_columns: columns that look like user IDs (unique per row, not useful for modeling)
"""


class LLMDiagnoser:
    """LLM-powered data quality diagnosis."""

    def __init__(self, api_key: str = None):
        self.client = get_client(api_key)

    def diagnose(self, df) -> dict | None:
        """Analyze a DataFrame and return a cleaning plan dict."""
        if not self.client.available:
            return None

        summary = self._build_column_summary(df)
        nulls = self._build_null_report(df)
        sample = df.head(5).to_string()

        prompt = USER_PROMPT_TEMPLATE.format(
            column_summary=summary,
            null_report=nulls,
            sample_rows=sample,
        )
        # Truncate if too long (DeepSeek has ~64K context, but keep it reasonable)
        if len(prompt) > 3000:
            prompt = prompt[:3000] + "\n... (truncated)"

        return self.client.chat_json(SYSTEM_PROMPT, prompt)

    # ── helpers ───────────────────────────────────────────────

    def _build_column_summary(self, df) -> str:
        lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_unique = df[col].nunique()
            n_null = df[col].isnull().sum()
            sample_vals = df[col].dropna().head(3).tolist()
            lines.append(
                f"  {col}: dtype={dtype}, unique={n_unique}, "
                f"nulls={n_null}, sample={sample_vals}"
            )
        return "\n".join(lines)

    def _build_null_report(self, df) -> str:
        null_counts = df.isnull().sum()
        null_pct = (null_counts / len(df) * 100).round(1)
        lines = []
        for col in df.columns:
            if null_counts[col] > 0:
                lines.append(f"  {col}: {null_counts[col]} nulls ({null_pct[col]}%)")
        return "\n".join(lines) if lines else "  No null values found."
