from __future__ import annotations

from datetime import date
from typing import Any, Dict

import requests
from requests import RequestException

from ..config import Settings


class FinnhubProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.finnhub_api_key
        self.base_url = settings.finnhub_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def connectivity_check(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._not_configured("未配置 FINNHUB_API_KEY。")
        try:
            payload = self._get_json(
                "calendar/earnings",
                {"symbol": "AAPL", "from": "2025-01-01", "to": "2025-01-31"},
            )
            earnings_count = len(payload.get("earningsCalendar") or []) if isinstance(payload, dict) else 0
            return {
                "enabled": True,
                "base_url": self.base_url,
                "status": "ok",
                "detail": f"earnings calendar 可访问，返回 {earnings_count} 条记录。",
            }
        except ValueError as exc:
            return self._error(str(exc))

    def fetch_earnings_calendar(
        self,
        *,
        symbol: str | None,
        date_from: date,
        date_to: date,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ValueError("未配置 FINNHUB_API_KEY。")
        params: Dict[str, Any] = {
            "from": date_from.isoformat(),
            "to": date_to.isoformat(),
        }
        if symbol:
            params["symbol"] = symbol.upper()
        payload = self._get_json("calendar/earnings", params)
        return {
            "source": "finnhub",
            "symbol": symbol.upper() if symbol else None,
            "from": date_from.isoformat(),
            "to": date_to.isoformat(),
            "earnings_calendar": payload.get("earningsCalendar") if isinstance(payload, dict) else [],
        }

    def _get_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = requests.get(
                f"{self.base_url}/{path}",
                params={**params, "token": self.api_key},
                timeout=30,
            )
            response.raise_for_status()
        except RequestException as exc:
            raise ValueError(f"Finnhub 请求失败：{exc}") from exc
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise ValueError(f"Finnhub 返回错误：{payload['error']}")
        return payload

    def _not_configured(self, detail: str) -> Dict[str, Any]:
        return {"enabled": False, "base_url": self.base_url, "status": "not_configured", "detail": detail}

    def _error(self, detail: str) -> Dict[str, Any]:
        return {"enabled": True, "base_url": self.base_url, "status": "error", "detail": detail}
