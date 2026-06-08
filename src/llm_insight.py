"""
LLM Insight Generator — Translates numbers into business language.

Takes the final analysis results (segment profiles, model metrics,
feature importance) and uses LLM to generate:
- Top 3 discoveries
- Actionable business recommendations
- Human-readable summary

This is the "agent speaks" module — the part that turns raw
analytics output into something a business stakeholder can read.
"""

from llm_client import get_client


SYSTEM_PROMPT = (
    "You are a senior business analyst translating data analysis results "
    "into actionable insights for a product/operations team. "
    "Write in professional but accessible language. "
    "Focus on WHAT to do, not HOW the model works."
)

USER_PROMPT_TEMPLATE = """Generate a business insight report from these analysis results:

=== User Segments ===
{segments}

=== Prediction Model ===
{model_info}

=== Top Predictive Features ===
{feature_importance}

=== Data Summary ===
{data_summary}

Output a JSON object with exactly these keys:
{{
  "headline": "One-sentence executive summary",
  "top_findings": [
    "Finding 1 with specific numbers",
    "Finding 2 with specific numbers",
    "Finding 3 with specific numbers"
  ],
  "recommendations": [
    "Actionable recommendation 1",
    "Actionable recommendation 2"
  ],
  "risk_alert": "Brief note about any risk segment" or null,
  "confidence": "high" or "medium"
}}

Rules:
- Use specific numbers from the data (e.g., "32% of users in the high-value segment account for 65% of revenue")
- Recommendations must be actionable (e.g., "Send re-engagement push to Segment_2 within 7 days")
- Keep each finding under 2 sentences
"""


class LLMInsight:
    """LLM-powered business insight generator."""

    def __init__(self, api_key: str = None):
        self.client = get_client(api_key)

    def generate(
        self,
        segments_summary: dict,
        model_metrics: dict,
        feature_importance: str = "",
        data_summary: str = "",
    ) -> dict | None:
        """Generate business insights from analysis results."""
        if not self.client.available:
            return None

        prompt = USER_PROMPT_TEMPLATE.format(
            segments=segments_summary,
            model_info=model_metrics,
            feature_importance=feature_importance or "Not available",
            data_summary=data_summary or "Not available",
        )

        return self.client.chat_json(SYSTEM_PROMPT, prompt)
