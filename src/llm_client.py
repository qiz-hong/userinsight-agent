"""
LLM Client — Shared DeepSeek API wrapper.

Usage:
    from llm_client import LLMClient
    client = LLMClient()               # reads DEEPSEEK_API_KEY from env
    reply = client.chat("你好")
    reply = client.chat_json("输出JSON...")  # auto-parses JSON

Supports OPENAI_API_KEY / DEEPSEEK_API_KEY env vars.
Falls back gracefully if no key is set → returns None.
"""

import json
import os
import re
import sys


class LLMClient:
    """Thin wrapper around DeepSeek's OpenAI-compatible chat API."""

    # Defaults — override via env vars or constructor
    BASE_URL = os.environ.get(
        "LLM_BASE_URL", "https://api.deepseek.com/v1"
    )
    MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
    MAX_TOKENS = 1024
    TEMPERATURE = 0.3  # low temp for deterministic analytical output

    def __init__(self, api_key: str = None):
        self.api_key = api_key or self._read_key_from_env()
        self._available = bool(self.api_key)
        if not self._available:
            print("  [LLM] No API key found — LLM features disabled. "
                  "Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")

    # ── Public API ────────────────────────────────────────────

    def chat(self, system_prompt: str, user_prompt: str) -> str | None:
        """Send a chat request, return text reply or None on failure."""
        if not self._available:
            return None
        try:
            return self._call(system_prompt, user_prompt, json_mode=False)
        except Exception as e:
            print(f"  [LLM] Request failed: {e}")
            return None

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict | None:
        """Send a chat request, parse JSON from the reply."""
        raw = self.chat(system_prompt, user_prompt)
        if raw is None:
            return None
        return self._extract_json(raw)

    @property
    def available(self) -> bool:
        return self._available

    # ── Internal ──────────────────────────────────────────────

    @staticmethod
    def _read_key_from_env() -> str | None:
        for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
            val = os.environ.get(var, "").strip()
            if val:
                return val
        return None

    def _call(self, system: str, user: str, json_mode: bool = False) -> str:
        import urllib.request
        import urllib.error

        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": self.MAX_TOKENS,
            "temperature": self.TEMPERATURE,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.BASE_URL}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {e.code}: {err_body[:300]}")

    def _extract_json(self, raw: str) -> dict | None:
        """Try to extract JSON from LLM output (handles markdown fences)."""
        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try ```json ... ``` fences
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        # Try to find JSON object { ... }
        match = re.search(r"\{[\s\S]*\}", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        print(f"  [LLM] Could not parse JSON from: {raw[:200]}...")
        return None


# ── Convenience singleton ─────────────────────────────────────

_client: LLMClient | None = None


def get_client(api_key: str = None) -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(api_key)
    return _client
