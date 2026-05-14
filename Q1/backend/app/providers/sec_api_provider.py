from __future__ import annotations

from typing import Any, Dict

import requests
from requests import RequestException

from ..config import Settings


class SecApiProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.sec_api_io_api_key
        self.base_url = settings.sec_api_io_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def connectivity_check(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._not_configured("未配置 SEC_API_IO_API_KEY。")
        try:
            payload = self.fetch_historical_shares_float("AAPL")
            record_count = len(payload.get("records") or [])
            return {
                "enabled": True,
                "base_url": self.base_url,
                "status": "ok",
                "detail": f"historical shares/float 可访问，返回 {record_count} 条记录。",
            }
        except ValueError as exc:
            return self._error(str(exc))

    def fetch_historical_shares_float(self, symbol: str) -> Dict[str, Any]:
        if not self.enabled:
            raise ValueError("未配置 SEC_API_IO_API_KEY。")
        normalized_symbol = symbol.strip().upper()
        try:
            response = requests.get(
                f"{self.base_url}/float",
                params={"ticker": normalized_symbol, "token": self.api_key},
                timeout=30,
            )
            response.raise_for_status()
        except RequestException as exc:
            raise ValueError(f"SEC-API 请求失败：{exc}") from exc
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error"):
            raise ValueError(f"SEC-API 返回错误：{payload['error']}")
        records = payload if isinstance(payload, list) else payload.get("data") or payload.get("records") or []
        return {"source": "sec_api_io", "symbol": normalized_symbol, "records": records}

    def _not_configured(self, detail: str) -> Dict[str, Any]:
        return {"enabled": False, "base_url": self.base_url, "status": "not_configured", "detail": detail}

    def _error(self, detail: str) -> Dict[str, Any]:
        return {"enabled": True, "base_url": self.base_url, "status": "error", "detail": detail}
