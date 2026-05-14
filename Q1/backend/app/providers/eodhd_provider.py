from __future__ import annotations

from typing import Any, Dict

import requests
from requests import RequestException

from ..config import Settings


class EodhdProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.eodhd_api_key
        self.base_url = settings.eodhd_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def connectivity_check(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._not_configured("未配置 EODHD_API_KEY。")
        try:
            payload = self._get_json("fundamentals/AAPL.US")
            available_sections = sorted(payload.keys())[:8] if isinstance(payload, dict) else []
            return {
                "enabled": True,
                "base_url": self.base_url,
                "status": "ok",
                "detail": f"fundamentals 可访问，返回 section 示例：{available_sections}",
            }
        except ValueError as exc:
            return self._error(str(exc))

    def fetch_analyst_revisions(self, symbol: str) -> Dict[str, Any]:
        if not self.enabled:
            raise ValueError("未配置 EODHD_API_KEY。")
        normalized_symbol = self._normalize_symbol(symbol)
        payload = self._get_json(f"fundamentals/{normalized_symbol}")
        earnings = payload.get("Earnings") if isinstance(payload, dict) else None
        analyst_ratings = payload.get("AnalystRatings") if isinstance(payload, dict) else None
        highlights = payload.get("Highlights") if isinstance(payload, dict) else None
        return {
            "source": "eodhd",
            "symbol": normalized_symbol,
            "earnings": earnings,
            "analyst_ratings": analyst_ratings,
            "highlights": highlights,
        }

    def _get_json(self, path: str) -> Dict[str, Any]:
        try:
            response = requests.get(
                f"{self.base_url}/{path}",
                params={"api_token": self.api_key, "fmt": "json"},
                timeout=30,
            )
            response.raise_for_status()
        except RequestException as exc:
            raise ValueError(f"EODHD 请求失败：{exc}") from exc
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise ValueError(f"EODHD 返回错误：{payload['error']}")
        return payload

    def _normalize_symbol(self, symbol: str) -> str:
        normalized = symbol.strip().upper()
        return normalized if "." in normalized else f"{normalized}.US"

    def _not_configured(self, detail: str) -> Dict[str, Any]:
        return {"enabled": False, "base_url": self.base_url, "status": "not_configured", "detail": detail}

    def _error(self, detail: str) -> Dict[str, Any]:
        return {"enabled": True, "base_url": self.base_url, "status": "error", "detail": detail}
