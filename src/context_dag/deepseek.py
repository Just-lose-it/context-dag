"""Optional DeepSeek Chat Completions adapter; no third-party dependency."""

from __future__ import annotations

import json
import os
import urllib.request


class DeepSeekClient:
    def __init__(self, *, model: str = "deepseek-flash", api_key: str | None = None,
                 base_url: str = "https://api.deepseek.com", timeout: int = 120,
                 max_tokens: int = 600, total_token_budget: int | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY") or self._windows_user_key()
        if not self.api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY or pass api_key")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.total_token_budget = total_token_budget
        self.usages: list[dict] = []

    @staticmethod
    def _windows_user_key() -> str | None:
        if os.name != "nt":
            return None
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                return winreg.QueryValueEx(key, "DEEPSEEK_API_KEY")[0]
        except FileNotFoundError:
            return None

    def __call__(self, messages: list[dict[str, str]]) -> str:
        if self.total_token_budget is not None and sum(
            usage.get("total_tokens", 0) for usage in self.usages
        ) >= self.total_token_budget:
            raise RuntimeError("total token budget exhausted")
        body = json.dumps({"model": self.model, "messages": messages,
                           "thinking": {"type": "disabled"}, "temperature": 0,
                           "max_tokens": self.max_tokens,
                           "response_format": {"type": "json_object"}}).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/chat/completions", data=body, method="POST",
            headers={"Authorization": "Bearer " + self.api_key,
                     "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.load(response)
        self.usages.append(data.get("usage", {}))
        if self.total_token_budget is not None and sum(
            usage.get("total_tokens", 0) for usage in self.usages
        ) > self.total_token_budget:
            raise RuntimeError("total token budget exceeded")
        content = data["choices"][0]["message"]["content"]
        if not content or not content.strip():
            raise ValueError("DeepSeek returned empty JSON content")
        return content
