from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from app.config import get_settings
from app.dataset_import.base import ImportedDataset
from app.schemas import DataImportRequest, MarketId


BACKEND_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BACKEND_DIR / "data" / "q1_market.db"
TARGET_DATASET_ID = "hetero_trade_mas_cache_us_v1"
DATA_TABLES = (
    "instruments",
    "trade_calendar",
    "daily_bars",
    "daily_basic",
    "financial_indicators",
    "dataset_events",
    "index_bars",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import one US-stock dataset into the Q1 local database.")
    parser.add_argument(
        "--source",
        default="local_cache",
        choices=["local_cache", "alpaca", "fmp", "yfinance"],
        help="导入源。会覆盖 Q1 当前默认数据集，前端无需改动。",
    )
    parser.add_argument("--start-date", default="2022-10-05")
    parser.add_argument("--end-date", default="2023-06-10")
    parser.add_argument("--symbols", default="AAPL,MSFT,NVDA,TSLA")
    parser.add_argument("--benchmark-symbol", default="SPY")
    parser.add_argument("--name", default=None)
    parser.add_argument("--symbol-limit", type=int, default=20)
    return parser.parse_args()


def parse_symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def build_request(args: argparse.Namespace) -> DataImportRequest:
    return DataImportRequest(
        market_id=MarketId.US_STOCK,
        name=args.name,
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        symbols=parse_symbols(args.symbols),
        symbol_limit=max(1, int(args.symbol_limit)),
        benchmark_symbol=args.benchmark_symbol.upper(),
    )


def build_provider(source: str):
    settings = get_settings()
    if source == "local_cache":
        from app.dataset_import.local_cache_provider import LocalResearchCacheProvider

        provider = LocalResearchCacheProvider(settings)
        if not provider.enabled:
            raise RuntimeError("Local Research Cache 不可用，请在 Q1 的 backend/.env 中配置 LOCAL_RESEARCH_CACHE_DIR。")
        return provider
    if source == "alpaca":
        from app.dataset_import.alpaca_provider import AlpacaProvider

        return AlpacaProvider(settings)
    if source == "fmp":
        from app.dataset_import.fmp_provider import FmpProvider

        return FmpProvider(settings)
    if source == "yfinance":
        from app.dataset_import.yfinance_provider import YFinanceProvider

        return YFinanceProvider()
    raise ValueError(f"未知导入源：{source}")


def build_dataset(args: argparse.Namespace) -> ImportedDataset:
    provider = build_provider(args.source)
    request = build_request(args)
    return provider.fetch_dataset(request)


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_versions (
            dataset_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            source_type TEXT NOT NULL,
            status TEXT NOT NULL,
            benchmark_symbol TEXT NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            symbol_count INTEGER NOT NULL,
            notes_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            market_id TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS instruments (
            dataset_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            industry TEXT NOT NULL,
            listed_date TEXT NOT NULL,
            delisted_date TEXT,
            market TEXT NOT NULL,
            market_id TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_calendar (
            dataset_id TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            is_open INTEGER NOT NULL,
            pretrade_date TEXT,
            market_id TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_bars (
            dataset_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            turnover REAL NOT NULL,
            adj_factor REAL NOT NULL,
            is_suspended INTEGER NOT NULL,
            is_st INTEGER NOT NULL,
            market_id TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_basic (
            dataset_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            turnover_rate REAL,
            pe_ttm REAL,
            pb REAL,
            total_mv REAL,
            circ_mv REAL,
            market_id TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS financial_indicators (
            dataset_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            ann_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            roe REAL,
            roa REAL,
            grossprofit_margin REAL,
            netprofit_yoy REAL,
            revenue_yoy REAL,
            ocf_to_or REAL,
            market_id TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_events (
            dataset_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            event_date TEXT NOT NULL,
            event_type TEXT NOT NULL,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            sentiment TEXT NOT NULL,
            score REAL NOT NULL,
            value REAL,
            market_id TEXT NOT NULL,
            published_at TEXT,
            summary TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS index_bars (
            dataset_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            turnover REAL NOT NULL,
            market_id TEXT NOT NULL
        )
        """
    )


def normalize_dataset(dataset: ImportedDataset, dataset_id: str = TARGET_DATASET_ID) -> ImportedDataset:
    return replace(dataset, dataset_id=dataset_id)


def iso_or_none(value) -> str | None:
    return value.isoformat() if value is not None else None


def delete_existing_dataset(conn: sqlite3.Connection, dataset_id: str) -> None:
    conn.execute("DELETE FROM dataset_versions WHERE dataset_id = ?", (dataset_id,))
    for table in DATA_TABLES:
        conn.execute(f"DELETE FROM {table} WHERE dataset_id = ?", (dataset_id,))


def insert_many(conn: sqlite3.Connection, sql: str, rows: Sequence[tuple]) -> None:
    if rows:
        conn.executemany(sql, rows)


def dataset_version_row(dataset: ImportedDataset) -> tuple:
    start_date = min(bar.trade_date for bar in dataset.daily_bars)
    end_date = max(bar.trade_date for bar in dataset.daily_bars)
    return (
        dataset.dataset_id,
        dataset.name,
        dataset.source_type,
        "ready",
        dataset.benchmark_symbol,
        start_date.isoformat(),
        end_date.isoformat(),
        len({item.symbol for item in dataset.instruments}),
        json.dumps(dataset.notes, ensure_ascii=False),
        datetime.utcnow().isoformat(),
        dataset.market_id.value,
    )


def instrument_rows(dataset: ImportedDataset) -> list[tuple]:
    return [
        (
            dataset.dataset_id,
            item.symbol,
            item.name,
            item.industry,
            item.listed_date.isoformat(),
            iso_or_none(item.delisted_date),
            item.market,
            dataset.market_id.value,
        )
        for item in dataset.instruments
    ]


def trade_calendar_rows(dataset: ImportedDataset) -> list[tuple]:
    return [
        (
            dataset.dataset_id,
            item.trade_date.isoformat(),
            1 if item.is_open else 0,
            iso_or_none(item.pretrade_date),
            dataset.market_id.value,
        )
        for item in dataset.trade_calendar
    ]


def daily_bar_rows(dataset: ImportedDataset) -> list[tuple]:
    return [
        (
            dataset.dataset_id,
            item.symbol,
            item.trade_date.isoformat(),
            item.open,
            item.high,
            item.low,
            item.close,
            item.volume,
            item.turnover,
            item.adj_factor,
            1 if item.is_suspended else 0,
            1 if item.is_st else 0,
            dataset.market_id.value,
        )
        for item in dataset.daily_bars
    ]


def daily_basic_rows(dataset: ImportedDataset) -> list[tuple]:
    return [
        (
            dataset.dataset_id,
            item.symbol,
            item.trade_date.isoformat(),
            item.turnover_rate,
            item.pe_ttm,
            item.pb,
            item.total_mv,
            item.circ_mv,
            dataset.market_id.value,
        )
        for item in dataset.daily_basics
    ]


def financial_rows(dataset: ImportedDataset) -> list[tuple]:
    return [
        (
            dataset.dataset_id,
            item.symbol,
            item.ann_date.isoformat(),
            item.end_date.isoformat(),
            item.roe,
            item.roa,
            item.grossprofit_margin,
            item.netprofit_yoy,
            item.revenue_yoy,
            item.ocf_to_or,
            dataset.market_id.value,
        )
        for item in dataset.financial_indicators
    ]


def event_rows(dataset: ImportedDataset) -> list[tuple]:
    return [
        (
            dataset.dataset_id,
            item.symbol,
            item.event_date.isoformat(),
            item.event_type,
            item.source,
            item.title,
            item.sentiment,
            item.score,
            item.value,
            dataset.market_id.value,
            iso_or_none(item.published_at),
            item.summary,
            json.dumps(item.metadata or {}, ensure_ascii=False),
        )
        for item in dataset.events
    ]


def index_bar_rows(dataset: ImportedDataset) -> list[tuple]:
    return [
        (
            dataset.dataset_id,
            item.symbol,
            item.trade_date.isoformat(),
            item.open,
            item.high,
            item.low,
            item.close,
            item.volume,
            item.turnover,
            dataset.market_id.value,
        )
        for item in dataset.index_bars
    ]


def write_dataset(dataset: ImportedDataset) -> ImportedDataset:
    normalized = normalize_dataset(dataset)
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DATABASE_PATH)) as conn:
        ensure_schema(conn)
        delete_existing_dataset(conn, normalized.dataset_id)
        conn.execute(
            """
            INSERT INTO dataset_versions (
                dataset_id, name, source_type, status, benchmark_symbol,
                start_date, end_date, symbol_count, notes_json, created_at, market_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            dataset_version_row(normalized),
        )
        insert_many(
            conn,
            """
            INSERT INTO instruments (
                dataset_id, symbol, name, industry, listed_date, delisted_date, market, market_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            instrument_rows(normalized),
        )
        insert_many(
            conn,
            """
            INSERT INTO trade_calendar (dataset_id, trade_date, is_open, pretrade_date, market_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            trade_calendar_rows(normalized),
        )
        insert_many(
            conn,
            """
            INSERT INTO daily_bars (
                dataset_id, symbol, trade_date, open, high, low, close,
                volume, turnover, adj_factor, is_suspended, is_st, market_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            daily_bar_rows(normalized),
        )
        insert_many(
            conn,
            """
            INSERT INTO daily_basic (
                dataset_id, symbol, trade_date, turnover_rate, pe_ttm, pb, total_mv, circ_mv, market_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            daily_basic_rows(normalized),
        )
        insert_many(
            conn,
            """
            INSERT INTO financial_indicators (
                dataset_id, symbol, ann_date, end_date, roe, roa, grossprofit_margin,
                netprofit_yoy, revenue_yoy, ocf_to_or, market_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            financial_rows(normalized),
        )
        insert_many(
            conn,
            """
            INSERT INTO dataset_events (
                dataset_id, symbol, event_date, event_type, source, title, sentiment, score, value,
                market_id, published_at, summary, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            event_rows(normalized),
        )
        insert_many(
            conn,
            """
            INSERT INTO index_bars (
                dataset_id, symbol, trade_date, open, high, low, close, volume, turnover, market_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            index_bar_rows(normalized),
        )
        conn.commit()
    return normalized


def build_summary(dataset: ImportedDataset) -> dict:
    return {
        "dataset_id": dataset.dataset_id,
        "name": dataset.name,
        "source_type": dataset.source_type,
        "symbols": sorted({item.symbol for item in dataset.instruments}),
        "benchmark_symbol": dataset.benchmark_symbol,
        "bar_count": len(dataset.daily_bars),
        "event_count": len(dataset.events),
        "database": str(DATABASE_PATH),
    }


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    normalized = write_dataset(dataset)
    print(json.dumps(build_summary(normalized), ensure_ascii=False))


if __name__ == "__main__":
    main()
