from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BACKEND_DIR / ".env"


def _load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    alpaca_api_key: str | None
    alpaca_secret_key: str | None
    alpaca_data_base_url: str
    alpaca_trading_base_url: str
    alpaca_data_feed: str
    fmp_api_key: str | None
    fmp_base_url: str
    finnhub_api_key: str | None
    finnhub_base_url: str
    eodhd_api_key: str | None
    eodhd_base_url: str
    sec_api_io_api_key: str | None
    sec_api_io_base_url: str
    fred_api_key: str | None
    fred_base_url: str
    local_research_cache_dir: Path | None
    sec_user_agent: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    env_values = _load_env_file(ENV_PATH)

    def read_env(key: str, default: str | None = None) -> str | None:
        return os.environ.get(key) or env_values.get(key) or default

    local_cache_dir = read_env("LOCAL_RESEARCH_CACHE_DIR")
    return Settings(
        alpaca_api_key=read_env("ALPACA_API_KEY"),
        alpaca_secret_key=read_env("ALPACA_SECRET_KEY"),
        alpaca_data_base_url=read_env("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets/v2") or "",
        alpaca_trading_base_url=read_env("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets") or "",
        alpaca_data_feed=read_env("ALPACA_DATA_FEED", "iex") or "iex",
        fmp_api_key=read_env("FMP_API_KEY"),
        fmp_base_url=read_env("FMP_BASE_URL", "https://financialmodelingprep.com/stable") or "",
        finnhub_api_key=read_env("FINNHUB_API_KEY"),
        finnhub_base_url=read_env("FINNHUB_BASE_URL", "https://finnhub.io/api/v1") or "",
        eodhd_api_key=read_env("EODHD_API_KEY"),
        eodhd_base_url=read_env("EODHD_BASE_URL", "https://eodhd.com/api") or "",
        sec_api_io_api_key=read_env("SEC_API_IO_API_KEY"),
        sec_api_io_base_url=read_env("SEC_API_IO_BASE_URL", "https://api.sec-api.io") or "",
        fred_api_key=read_env("FRED_API_KEY"),
        fred_base_url=read_env("FRED_BASE_URL", "https://api.stlouisfed.org/fred") or "",
        local_research_cache_dir=Path(local_cache_dir).expanduser() if local_cache_dir else None,
        sec_user_agent=read_env("SEC_USER_AGENT", "Q1ResearchPlatform/1.0 contact@example.com") or "",
    )
