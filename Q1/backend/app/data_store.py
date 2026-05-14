from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from .schemas import (
    Bar,
    DailyBasicRecord,
    DatasetBundle,
    DatasetEventRecord,
    DatasetSummary,
    FinancialIndicatorRecord,
    Instrument,
)


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "q1_market.db"


class SQLiteMarketStore:
    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"market db not found: {self.db_path}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def list_datasets(self) -> List[DatasetSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select dataset_id, name, source_type, benchmark_symbol,
                       start_date, end_date, symbol_count, notes_json
                from dataset_versions
                order by created_at desc
                """
            ).fetchall()
        return [
            DatasetSummary(
                dataset_id=row["dataset_id"],
                name=row["name"],
                source_type=row["source_type"],
                benchmark_symbol=row["benchmark_symbol"],
                start_date=date.fromisoformat(row["start_date"]),
                end_date=date.fromisoformat(row["end_date"]),
                symbol_count=row["symbol_count"],
                notes=json.loads(row["notes_json"] or "[]"),
            )
            for row in rows
        ]

    def load_dataset_bundle(
        self,
        dataset_id: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> DatasetBundle:
        summary = next((item for item in self.list_datasets() if item.dataset_id == dataset_id), None)
        if summary is None:
            raise ValueError(f"dataset not found: {dataset_id}")

        query_params: List[object] = [dataset_id]
        date_clause = ""
        if start_date:
            date_clause += " and trade_date >= ?"
            query_params.append(start_date.isoformat())
        if end_date:
            date_clause += " and trade_date <= ?"
            query_params.append(end_date.isoformat())

        with self._connect() as conn:
            instrument_rows = conn.execute(
                """
                select symbol, name, industry, listed_date, delisted_date, market
                from instruments
                where dataset_id = ?
                order by symbol
                """,
                (dataset_id,),
            ).fetchall()
            bar_rows = conn.execute(
                f"""
                select symbol, trade_date, open, high, low, close, volume, turnover, adj_factor, is_suspended
                from daily_bars
                where dataset_id = ? {date_clause}
                order by symbol, trade_date
                """,
                query_params,
            ).fetchall()
            index_rows = conn.execute(
                f"""
                select symbol, trade_date, open, high, low, close, volume, turnover
                from index_bars
                where dataset_id = ? and symbol = ? {date_clause}
                order by trade_date
                """,
                [dataset_id, summary.benchmark_symbol, *query_params[1:]],
            ).fetchall()
            basic_rows = conn.execute(
                f"""
                select symbol, trade_date, turnover_rate, pe_ttm, pb, total_mv, circ_mv
                from daily_basic
                where dataset_id = ? {date_clause}
                order by symbol, trade_date
                """,
                query_params,
            ).fetchall()
            financial_rows = conn.execute(
                """
                select symbol, ann_date, end_date, roe, roa, grossprofit_margin, netprofit_yoy, revenue_yoy, ocf_to_or
                from financial_indicators
                where dataset_id = ?
                order by symbol, ann_date
                """,
                (dataset_id,),
            ).fetchall()
            event_rows = conn.execute(
                """
                select symbol, event_date, event_type, source, title, published_at, summary, sentiment, score
                from dataset_events
                where dataset_id = ?
                order by symbol, coalesce(published_at, event_date), event_type, title
                """,
                (dataset_id,),
            ).fetchall()
            calendar_rows = conn.execute(
                f"""
                select trade_date
                from trade_calendar
                where dataset_id = ? and is_open = 1 {date_clause}
                order by trade_date
                """,
                query_params,
            ).fetchall()

        instruments = {
            row["symbol"]: Instrument(
                symbol=row["symbol"],
                name=row["name"],
                industry=row["industry"],
                listed_date=date.fromisoformat(row["listed_date"]),
                delisted_date=date.fromisoformat(row["delisted_date"]) if row["delisted_date"] else None,
                market=row["market"],
            )
            for row in instrument_rows
        }
        price_history: Dict[str, List[Bar]] = {}
        for row in bar_rows:
            price_history.setdefault(row["symbol"], []).append(
                Bar(
                    trade_date=date.fromisoformat(row["trade_date"]),
                    symbol=row["symbol"],
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    turnover=row["turnover"],
                    adj_factor=row["adj_factor"],
                    is_suspended=bool(row["is_suspended"]),
                )
            )
        benchmark_history = [
            Bar(
                trade_date=date.fromisoformat(row["trade_date"]),
                symbol=row["symbol"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
                turnover=row["turnover"],
            )
            for row in index_rows
        ]
        daily_basic_history: Dict[str, Dict[date, DailyBasicRecord]] = {}
        for row in basic_rows:
            trade_day = date.fromisoformat(row["trade_date"])
            daily_basic_history.setdefault(row["symbol"], {})[trade_day] = DailyBasicRecord(
                trade_date=trade_day,
                symbol=row["symbol"],
                turnover_rate=row["turnover_rate"],
                pe_ttm=row["pe_ttm"],
                pb=row["pb"],
                total_mv=row["total_mv"],
                circ_mv=row["circ_mv"],
            )
        financial_history: Dict[str, List[FinancialIndicatorRecord]] = {}
        for row in financial_rows:
            financial_history.setdefault(row["symbol"], []).append(
                FinancialIndicatorRecord(
                    symbol=row["symbol"],
                    ann_date=date.fromisoformat(row["ann_date"]),
                    end_date=date.fromisoformat(row["end_date"]),
                    roe=row["roe"],
                    roa=row["roa"],
                    grossprofit_margin=row["grossprofit_margin"],
                    netprofit_yoy=row["netprofit_yoy"],
                    revenue_yoy=row["revenue_yoy"],
                    ocf_to_or=row["ocf_to_or"],
                )
            )
        event_history: Dict[str, Dict[date, List[DatasetEventRecord]]] = {}
        for row in event_rows:
            source_event_date = date.fromisoformat(row["event_date"])
            published_at = datetime.fromisoformat(row["published_at"]) if row["published_at"] else None
            available_date = self._event_available_date(source_event_date, published_at)
            event_history.setdefault(row["symbol"], {}).setdefault(available_date, []).append(
                DatasetEventRecord(
                    symbol=row["symbol"],
                    event_date=source_event_date,
                    event_type=row["event_type"],
                    source=row["source"],
                    title=row["title"],
                    published_at=published_at,
                    summary=row["summary"] or "",
                    sentiment=row["sentiment"],
                    score=row["score"],
                )
            )

        return DatasetBundle(
            dataset_id=summary.dataset_id,
            name=summary.name,
            benchmark_symbol=summary.benchmark_symbol,
            trade_dates=[date.fromisoformat(row["trade_date"]) for row in calendar_rows],
            benchmark_history=benchmark_history,
            instruments=instruments,
            price_history=price_history,
            daily_basic_history=daily_basic_history,
            financial_history=financial_history,
            event_history=event_history,
        )

    def _event_available_date(self, source_event_date: date, published_at: datetime | None) -> date:
        if published_at is None:
            return source_event_date
        published_date = published_at.date()
        return published_date if published_date > source_event_date else source_event_date

    def get_dataset_detail(self, dataset_id: str) -> Dict[str, object]:
        summary = next((item for item in self.list_datasets() if item.dataset_id == dataset_id), None)
        if summary is None:
            raise ValueError(f"dataset not found: {dataset_id}")

        with self._connect() as conn:
            bar_count = conn.execute(
                "select count(*) as cnt from daily_bars where dataset_id = ?",
                (dataset_id,),
            ).fetchone()["cnt"]
            basic_count = conn.execute(
                "select count(*) as cnt from daily_basic where dataset_id = ?",
                (dataset_id,),
            ).fetchone()["cnt"]
            financial_count = conn.execute(
                "select count(*) as cnt from financial_indicators where dataset_id = ?",
                (dataset_id,),
            ).fetchone()["cnt"]
            event_count = conn.execute(
                "select count(*) as cnt from dataset_events where dataset_id = ?",
                (dataset_id,),
            ).fetchone()["cnt"]
            news_count = conn.execute(
                "select count(*) as cnt from dataset_events where dataset_id = ? and event_type = 'news'",
                (dataset_id,),
            ).fetchone()["cnt"]
            earnings_count = conn.execute(
                "select count(*) as cnt from dataset_events where dataset_id = ? and event_type = 'earnings'",
                (dataset_id,),
            ).fetchone()["cnt"]
            symbols = conn.execute(
                """
                select symbol, name, industry, market
                from instruments
                where dataset_id = ?
                order by symbol
                """,
                (dataset_id,),
            ).fetchall()
        return {
            "summary": {
                "dataset_id": summary.dataset_id,
                "name": summary.name,
                "source_type": summary.source_type,
                "benchmark_symbol": summary.benchmark_symbol,
                "start_date": str(summary.start_date),
                "end_date": str(summary.end_date),
                "symbol_count": summary.symbol_count,
                "notes": summary.notes,
            },
            "coverage": {
                "bar_count": bar_count,
                "daily_basic_count": basic_count,
                "financial_count": financial_count,
                "event_count": event_count,
                "news_event_count": news_count,
                "earnings_event_count": earnings_count,
            },
            "symbols": [
                {
                    "symbol": row["symbol"],
                    "name": row["name"],
                    "industry": row["industry"],
                    "market": row["market"],
                }
                for row in symbols
            ],
        }

    def get_data_sources(self) -> List[Dict[str, object]]:
        return [
            {
                "source_type": "local_research_cache",
                "label": "Local Research Cache",
                "enabled": True,
                "mode": "terminal_import",
                "description": "从本地缓存目录导入价格、news、SEC、macro 和事件数据。",
                "datasets": ["hetero_trade_mas_cache_us_v1"],
            }
        ]
