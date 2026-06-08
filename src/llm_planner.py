"""
LLM Planner — Intelligent analysis strategy planning.

Given engineered feature metadata, uses LLM to decide:
- Optimal number of user segments (k)
- Which features drive segmentation
- Prediction target and model preference
- Key features for prediction

Part of the Agent layer: replaces static config decisions
with context-aware LLM reasoning.
"""

import json
from llm_client import get_client


SYSTEM_PROMPT = (
    "You are a data science strategist. Given feature metadata from a user "
    "behavior dataset, decide the optimal analysis strategy. "
    "Only output valid JSON, no explanation."
)

USER_PROMPT_TEMPLATE = """Based on this engineered feature summary, plan the analysis:

=== Feature Columns ({n_cols} total) ===
{feature_list}

=== Descriptive Stats ===
{stats}

Output a JSON object:
{{
  "segmentation": {{
    "n_clusters": 3,
    "recommended_features": ["feat1", "feat2", "feat3"],
    "reasoning": "one sentence why"
  }},
  "prediction": {{
    "recommended_target": "churn" or null,
    "model_preference": "rf" or "lr",
    "reasoning": "one sentence why"
  }},
  "key_insight_directions": ["direction 1", "direction 2"]
}}

Rules:
- n_clusters: 3-5 based on feature count (more features → more clusters)
- recommended_features: top 3-5 features most useful for segmentation
- model_preference: "rf" for non-linear patterns, "lr" for linear data
"""


class LLMPlanner:
    """LLM-powered analysis strategy planner."""

    def __init__(self, api_key: str = None):
        self.client = get_client(api_key)

    def plan(self, df_features) -> dict | None:
        """Analyze feature metadata and return a strategy dict."""
        if not self.client.available:
            return None

        feature_list = self._describe_features(df_features)
        stats = self._quick_stats(df_features)

        prompt = USER_PROMPT_TEMPLATE.format(
            n_cols=len(df_features.columns),
            feature_list=feature_list,
            stats=stats,
        )
        if len(prompt) > 3000:
            prompt = prompt[:3000] + "\n... (truncated)"

        return self.client.chat_json(SYSTEM_PROMPT, prompt)

    # ── helpers ───────────────────────────────────────────────

    def _describe_features(self, df) -> str:
        lines = []
        for col in df.columns:
            s = df[col]
            lines.append(
                f"  {col}: mean={s.mean():.2f}, std={s.std():.2f}, "
                f"min={s.min():.2f}, max={s.max():.2f}"
            )
        return "\n".join(lines)

    def _quick_stats(self, df) -> str:
        import numpy as np
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return "  Not enough numeric features for correlation."

        corr = numeric.corr()
        # Find top 3 strongest correlations (excluding self)
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                pairs.append((corr.columns[i], corr.columns[j],
                              abs(corr.iloc[i, j])))
        pairs.sort(key=lambda x: x[2], reverse=True)
        top = pairs[:5]
        lines = [f"  {a} ↔ {b}: r={v:.2f}" for a, b, v in top]
        return "\n".join(lines)
