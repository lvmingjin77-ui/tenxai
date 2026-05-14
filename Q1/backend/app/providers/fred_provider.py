from __future__ import annotations

from datetime import date
from typing import Any, Dict, Iterable

import requests
from requests import RequestException

from ..config import Settings


DEFAULT_REGIME_SERIES = ["VIXCLS", "DGS2", "DGS10", "T10Y2Y", "FEDFUNDS"]


class FredProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.fred_api_key
        self.base_url = settings.fred_base_url.rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def connectivity_check(self) -> Dict[str, Any]:
        if not self.enabled:
            return self._not_configured("未配置 FRED_API_KEY。")
        try:
            payload = self._get_json(
                "series/observations",
                {
                    "series_id": "VIXCLS",
                    "observation_start": "2025-01-01",
                    "observation_end": "2025-01-15",
                },
            )
            observation_count = len(payload.get("observations") or []) if isinstance(payload, dict) else 0
            return {
                "enabled": True,
                "base_url": self.base_url,
                "status": "ok",
                "detail": f"FRED observations 可访问，返回 {observation_count} 条记录。",
            }
        except ValueError as exc:
            return self._error(str(exc))

    def fetch_macro_regime(
        self,
        *,
        series_ids: Iterable[str] | None,
        date_from: date,
        date_to: date,
    ) -> Dict[str, Any]:
        if not self.enabled:
            raise ValueError("未配置 FRED_API_KEY。")
        selected_series = [item.strip().upper() for item in (series_ids or DEFAULT_REGIME_SERIES) if item.strip()]
        if not selected_series:
            selected_series = list(DEFAULT_REGIME_SERIES)
        series_payloads: Dict[str, Any] = {}
        for series_id in selected_series:
            payload = self._get_json(
                "series/observations",
                {
                    "series_id": series_id,
                    "observation_start": date_from.isoformat(),
                    "observation_end": date_to.isoformat(),
                },
            )
            series_payloads[series_id] = payload.get("observations") if isinstance(payload, dict) else []
        return {
            "source": "fred",
            "from": date_from.isoformat(),
            "to": date_to.isoformat(),
            "series": series_payloads,
        }

    def _get_json(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = requests.get(
                f"{self.base_url}/{path}",
                params={
                    **params,
                    "api_key": self.api_key,
                    "file_type": "json",
                },
                timeout=30,
            )
            response.raise_for_status()
        except RequestException as exc:
            raise ValueError(f"FRED 请求失败：{exc}") from exc
        payload = response.json()
        if isinstance(payload, dict) and payload.get("error_message"):
            raise ValueError(f"FRED 返回错误：{payload['error_message']}")
        return payload

    def _not_configured(self, detail: str) -> Dict[str, Any]:
        return {"enabled": False, "base_url": self.base_url, "status": "not_configured", "detail": detail}

    def _error(self, detail: str) -> Dict[str, Any]:
        return {"enabled": True, "base_url": self.base_url, "status": "error", "detail": detail}
