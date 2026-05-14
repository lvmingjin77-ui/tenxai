from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str
    model: str


class OpenAICompatibleClient:
    ENV_FILES = [Path(__file__).resolve().parent.parent / ".env"]

    def __init__(self) -> None:
        env_values = self._load_env_files()
        api_key = (
            os.environ.get("DEEPSEEK_API_KEY")
            or env_values.get("DEEPSEEK_API_KEY")
            or os.environ.get("Q1_OPENAI_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        base_url = (
            os.environ.get("DEEPSEEK_BASE_URL")
            or env_values.get("DEEPSEEK_BASE_URL")
            or os.environ.get("Q1_OPENAI_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        model = (
            os.environ.get("DEEPSEEK_MODEL")
            or env_values.get("DEEPSEEK_MODEL")
            or os.environ.get("Q1_OPENAI_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "deepseek-chat"
        )
        self.settings = LLMSettings(
            api_key=api_key.strip(),
            base_url=base_url.rstrip("/"),
            model=model.strip(),
        )

    @property
    def configured(self) -> bool:
        return bool(self.settings.api_key and self.settings.base_url and self.settings.model)

    def connectivity_check(self) -> Dict[str, Any]:
        return {
            "configured": self.configured,
            "base_url": self.settings.base_url,
            "model": self.settings.model,
        }

    def _load_env_files(self) -> Dict[str, str]:
        values: Dict[str, str] = {}
        for path in self.ENV_FILES:
            if not path.exists():
                continue
            for raw_line in path.read_text().splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in values:
                    values[key] = value
        return values

    def create_chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2200,
    ) -> Dict[str, Any]:
        if not self.configured:
            raise RuntimeError(
                "LLM client is not configured. Set DEEPSEEK_API_KEY in backend/.env or export it in the shell."
            )

        payload = {
            "model": self.settings.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        request = urllib.request.Request(
            url=f"{self.settings.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc

        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError("LLM response did not include any choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
            content = "".join(text_parts)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response content is empty.")
        return {"content": content}
