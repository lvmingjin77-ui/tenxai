from __future__ import annotations

import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

from ..config import Settings
from ..schemas import (
    DataImportRequest,
    DatasetEventRecord,
    DailyBar,
    DailyBasicRecord,
    InstrumentRecord,
    MarketId,
    TradeCalendarDay,
)
from .base import ImportedDataset


class LocalResearchCacheProvider:
    DATASET_ID = "hetero_trade_mas_cache_us_v1"
    SYMBOL_METADATA: Mapping[str, Dict[str, str]] = {
        "AAPL": {"name": "Apple Inc.", "industry": "Consumer Electronics", "market": "NASDAQ"},
        "MSFT": {"name": "Microsoft Corp.", "industry": "Software", "market": "NASDAQ"},
        "NVDA": {"name": "NVIDIA Corp.", "industry": "Semiconductors", "market": "NASDAQ"},
        "TSLA": {"name": "Tesla Inc.", "industry": "Automobiles", "market": "NASDAQ"},
        "SPY": {"name": "SPDR S&P 500 ETF Trust", "industry": "ETF", "market": "NYSE"},
    }

    def __init__(self, settings: Settings) -> None:
        self.cache_dir = settings.local_research_cache_dir
        self._macro_frame: pd.DataFrame | None = None
        self._global_news_frame: pd.DataFrame | None = None

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None and self.cache_dir.exists()

    def load_symbol_events(
        self,
        *,
        symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, Dict[date, List[DatasetEventRecord]]]:
        if not self.enabled:
            return {}
        events: Dict[str, Dict[date, List[DatasetEventRecord]]] = {}
        for symbol in symbols:
            symbol_events: List[DatasetEventRecord] = []
            symbol_events.extend(self._load_news_events(symbol, start_date, end_date))
            symbol_events.extend(self._load_sec_events(symbol, start_date, end_date))
            symbol_events.extend(self._load_insider_events(symbol, start_date, end_date))
            if not symbol_events:
                continue
            for event in symbol_events:
                availability_date = event.published_at.date() if event.published_at else event.event_date
                events.setdefault(symbol, {}).setdefault(availability_date, []).append(event)
        return events

    def fetch_dataset(self, request: DataImportRequest) -> ImportedDataset:
        if not self.enabled:
            raise ValueError("未配置本地研究缓存目录。")
        benchmark_symbol = (request.benchmark_symbol or "SPY").upper()
        available_symbols = self.list_available_price_symbols()
        if benchmark_symbol not in available_symbols:
            raise ValueError(f"本地缓存缺少基准价格：{benchmark_symbol}")

        requested_symbols = [symbol.upper() for symbol in request.symbols if symbol.upper() != benchmark_symbol]
        symbol_universe = requested_symbols or [symbol for symbol in available_symbols if symbol != benchmark_symbol]
        symbols = [symbol for symbol in symbol_universe if symbol in available_symbols]
        if not symbols:
            raise ValueError("本地缓存中没有可导入的股票价格文件。")

        price_frames = {
            symbol: self._load_price_frame(symbol, request.start_date, request.end_date)
            for symbol in [*symbols, benchmark_symbol]
        }
        for symbol, frame in price_frames.items():
            if frame is None or frame.empty:
                raise ValueError(f"本地缓存价格为空：{symbol}")

        trade_dates = sorted(
            set(price_frames[benchmark_symbol].index.date).intersection(
                *(set(price_frames[symbol].index.date) for symbol in symbols)
            )
        )
        if not trade_dates:
            raise ValueError("本地缓存价格区间没有共同交易日。")

        instruments = [self._build_instrument(symbol) for symbol in symbols]
        calendar = [TradeCalendarDay(trade_date=trade_date, is_open=True) for trade_date in trade_dates]
        daily_bars = self._build_daily_bars(symbols, price_frames, trade_dates)
        daily_basics = self._build_daily_basics(symbols, price_frames, trade_dates)
        events = self._flatten_events(
            self.load_symbol_events(
                symbols=symbols,
                start_date=trade_dates[0],
                end_date=trade_dates[-1],
            )
        )
        index_bars = self._build_index_bars(benchmark_symbol, price_frames[benchmark_symbol], trade_dates)

        return ImportedDataset(
            dataset_id=self.DATASET_ID,
            market_id=MarketId.US_STOCK,
            name=request.name or "hetero_trade_mas 本地研究缓存",
            source_type="local_research_cache",
            benchmark_symbol=benchmark_symbol,
            instruments=instruments,
            trade_calendar=calendar,
            daily_bars=daily_bars,
            daily_basics=daily_basics,
            financial_indicators=[],
            events=events,
            index_bars=index_bars,
            notes=[
                "导入自 hetero_trade_mas/data/cache，本地缓存不依赖在线抓取。",
                "已导入 prices、news、SEC filings、insider transactions 和 FRED 宏观缓存。",
                f"股票池：{', '.join(symbols)}；基准：{benchmark_symbol}。",
            ],
        )

    def list_available_price_symbols(self) -> List[str]:
        if not self.enabled:
            return []
        price_dir = self.cache_dir / "prices"
        if not price_dir.exists():
            return []
        return sorted(path.stem.upper() for path in price_dir.glob("*.parquet"))

    def load_macro_snapshot(self, trade_date: date) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        frame = self._load_macro_frame()
        if frame is None or frame.empty:
            return {}
        timestamp = pd.Timestamp(trade_date)
        history = frame.loc[frame.index <= timestamp]
        if history.empty:
            return {}
        row = history.iloc[-1]
        return {
            "macro_regime": self._safe_str(row.get("macro_regime")),
            "rates_regime": self._safe_str(row.get("rates_regime")),
            "curve_regime": self._safe_str(row.get("curve_regime")),
            "vol_regime": self._safe_str(row.get("vol_regime")),
            "fed_funds": self._safe_float(row.get("fed_funds")),
            "ust10y": self._safe_float(row.get("ust10y")),
            "ust2y": self._safe_float(row.get("ust2y")),
            "vix": self._safe_float(row.get("vix")),
            "yield_curve_10y2y": self._safe_float(row.get("yield_curve_10y2y")),
            "financial_stress": self._safe_float(row.get("financial_stress")),
        }

    def load_global_market_event(self, trade_date: date) -> Dict[str, Any]:
        if not self.enabled:
            return {}
        frame = self._load_global_news_frame()
        if frame is None or frame.empty:
            return {}
        timestamp = pd.Timestamp(trade_date)
        history = frame.loc[frame.index <= timestamp]
        if history.empty:
            return {}
        row = history.iloc[-1]
        return {
            "macro_news_summary": self._safe_str(row.get("summary")),
            "macro_news_polarity": self._safe_str(row.get("polarity")),
            "macro_news_importance": self._safe_str(row.get("importance")),
            "macro_news_event_type": self._safe_str(row.get("event_type")),
        }

    def _load_news_events(self, symbol: str, start_date: date, end_date: date) -> List[DatasetEventRecord]:
        path = self._cache_file("news", symbol)
        if path is None:
            return []
        frame = self._read_parquet(path)
        if frame is None or frame.empty:
            return []
        frame = self._normalize_indexed_frame(frame, start_date, end_date)
        records: List[DatasetEventRecord] = []
        for row_date, row in frame.iterrows():
            event_date = row_date.date()
            headlines = self._normalize_list(row.get("headlines"))
            title = headlines[0] if headlines else f"{symbol} news event"
            records.append(
                DatasetEventRecord(
                    symbol=symbol,
                    event_date=event_date,
                    event_type="news",
                    source="hetero_news_cache",
                    title=title[:160],
                    published_at=datetime.combine(event_date, time(16, 0)),
                    summary=self._safe_str(row.get("summary")),
                    sentiment=self._safe_str(row.get("polarity")) or "neutral",
                    score=self._score_from_confidence(row.get("confidence"), row.get("sentiment_mean")),
                    value=self._safe_float(row.get("sentiment_mean")),
                    metadata={
                        "event_id": self._safe_str(row.get("event_id")),
                        "importance": self._safe_str(row.get("importance")),
                        "novelty": self._safe_str(row.get("novelty")),
                        "persistence": self._safe_str(row.get("persistence")),
                        "scope": self._safe_str(row.get("scope")),
                        "surprise": self._safe_str(row.get("surprise")),
                        "confidence": self._safe_str(row.get("confidence")),
                        "source_count": self._safe_int(row.get("source_count")),
                        "evidence": self._normalize_list(row.get("evidence")),
                        "headlines": headlines[:8],
                    },
                )
            )
        return records

    def _load_price_frame(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame | None:
        path = self._cache_file("prices", symbol)
        frame = self._read_parquet(path)
        if frame is None or frame.empty:
            return None
        frame = self._normalize_indexed_frame(frame, start_date, end_date)
        return frame if not frame.empty else None

    def _build_instrument(self, symbol: str) -> InstrumentRecord:
        metadata = self.SYMBOL_METADATA.get(symbol, {"name": symbol, "industry": "Unknown", "market": "US"})
        return InstrumentRecord(
            symbol=symbol,
            name=metadata["name"],
            industry=metadata["industry"],
            listed_date=date(2000, 1, 1),
            market=metadata["market"],
        )

    def _build_daily_bars(
        self,
        symbols: List[str],
        price_frames: Mapping[str, pd.DataFrame],
        trade_dates: List[date],
    ) -> List[DailyBar]:
        records: List[DailyBar] = []
        for symbol in symbols:
            frame = price_frames[symbol]
            for trade_date in trade_dates:
                row = frame.loc[pd.Timestamp(trade_date)]
                records.append(
                    DailyBar(
                        trade_date=trade_date,
                        symbol=symbol,
                        open=round(float(row["Open"]), 4),
                        high=round(float(row["High"]), 4),
                        low=round(float(row["Low"]), 4),
                        close=round(float(row["Close"]), 4),
                        volume=int(row["Volume"]),
                        turnover=round(float(row["Close"]) * float(row["Volume"]), 2),
                    )
                )
        return records

    def _build_index_bars(
        self,
        benchmark_symbol: str,
        frame: pd.DataFrame,
        trade_dates: List[date],
    ) -> List[DailyBar]:
        records: List[DailyBar] = []
        for trade_date in trade_dates:
            row = frame.loc[pd.Timestamp(trade_date)]
            records.append(
                DailyBar(
                    trade_date=trade_date,
                    symbol=benchmark_symbol,
                    open=round(float(row["Open"]), 4),
                    high=round(float(row["High"]), 4),
                    low=round(float(row["Low"]), 4),
                    close=round(float(row["Close"]), 4),
                    volume=int(row["Volume"]),
                    turnover=round(float(row["Close"]) * float(row["Volume"]), 2),
                )
            )
        return records

    def _build_daily_basics(
        self,
        symbols: List[str],
        price_frames: Mapping[str, pd.DataFrame],
        trade_dates: List[date],
    ) -> List[DailyBasicRecord]:
        records: List[DailyBasicRecord] = []
        for symbol in symbols:
            frame = price_frames[symbol]
            for trade_date in trade_dates:
                row = frame.loc[pd.Timestamp(trade_date)]
                market_cap = self._safe_float(row.get("market_cap"))
                turnover = float(row["Close"]) * float(row["Volume"])
                turnover_rate = (turnover / market_cap * 100) if market_cap and market_cap > 0 else None
                records.append(
                    DailyBasicRecord(
                        trade_date=trade_date,
                        symbol=symbol,
                        turnover_rate=round(turnover_rate, 4) if turnover_rate is not None else None,
                        pe_ttm=self._safe_float(row.get("pe_ratio")),
                        pb=self._safe_float(row.get("pb_ratio")),
                        total_mv=market_cap,
                        circ_mv=market_cap,
                    )
                )
        return records

    def _flatten_events(
        self,
        event_history: Mapping[str, Mapping[date, List[DatasetEventRecord]]],
    ) -> List[DatasetEventRecord]:
        records: List[DatasetEventRecord] = []
        for history_by_date in event_history.values():
            for events in history_by_date.values():
                records.extend(events)
        return records

    def _load_sec_events(self, symbol: str, start_date: date, end_date: date) -> List[DatasetEventRecord]:
        path = self._cache_file("sec", symbol)
        if path is None:
            return []
        frame = self._read_parquet(path)
        if frame is None or frame.empty:
            return []
        frame = self._normalize_indexed_frame(frame, start_date, end_date)
        records: List[DatasetEventRecord] = []
        for row_date, row in frame.iterrows():
            event_date = row_date.date()
            event_type = self._safe_str(row.get("sec_event_type")) or "disclosure"
            form_type = self._safe_str(row.get("sec_form_type"))
            title = f"{form_type} {event_type}".strip()
            records.append(
                DatasetEventRecord(
                    symbol=symbol,
                    event_date=event_date,
                    event_type=event_type,
                    source="hetero_sec_cache",
                    title=title[:160],
                    published_at=datetime.combine(event_date, time(16, 0)),
                    summary=self._safe_str(row.get("sec_summary")),
                    sentiment=self._safe_str(row.get("sec_polarity")) or "neutral",
                    score=self._score_from_importance(row.get("sec_importance"), row.get("sec_confidence")),
                    metadata={
                        "event_id": self._safe_str(row.get("sec_event_id")),
                        "form_type": form_type,
                        "form_item": self._safe_str(row.get("sec_form_item")),
                        "importance": self._safe_str(row.get("sec_importance")),
                        "novelty": self._safe_str(row.get("sec_novelty")),
                        "persistence": self._safe_str(row.get("sec_persistence")),
                        "scope": self._safe_str(row.get("sec_scope")),
                        "surprise": self._safe_str(row.get("sec_surprise")),
                        "confidence": self._safe_str(row.get("sec_confidence")),
                        "trade_relevance": self._safe_str(row.get("sec_trade_relevance")),
                        "source_count": self._safe_int(row.get("sec_source_count")),
                        "event_tags": self._normalize_list(row.get("sec_event_tags")),
                        "evidence": self._normalize_list(row.get("sec_evidence")),
                        "filing_urls": self._normalize_list(row.get("sec_filing_urls")),
                    },
                )
            )
        return records

    def _load_insider_events(self, symbol: str, start_date: date, end_date: date) -> List[DatasetEventRecord]:
        path = self._cache_file("insiders", symbol)
        if path is None:
            return []
        frame = self._read_parquet(path)
        if frame is None or frame.empty:
            return []
        if "date" not in frame.columns:
            return []
        working = frame.copy()
        working["date"] = pd.to_datetime(working["date"], errors="coerce")
        working = working.dropna(subset=["date"])
        working = working[(working["date"] >= pd.Timestamp(start_date)) & (working["date"] <= pd.Timestamp(end_date))]
        if working.empty:
            return []

        records: List[DatasetEventRecord] = []
        for event_timestamp, group in working.groupby(working["date"].dt.normalize()):
            event_date = event_timestamp.date()
            signed_value = float(group.get("signed_value", pd.Series(dtype=float)).fillna(0.0).sum())
            gross_value = float(group.get("value", pd.Series(dtype=float)).fillna(0.0).abs().sum())
            signed_shares = float(group.get("signed_shares", pd.Series(dtype=float)).fillna(0.0).sum())
            gross_shares = float(group.get("shares", pd.Series(dtype=float)).fillna(0.0).abs().sum())
            direction = "neutral"
            if signed_value > 0 or (gross_value == 0 and signed_shares > 0):
                direction = "positive"
            elif signed_value < 0 or (gross_value == 0 and signed_shares < 0):
                direction = "negative"
            conviction = abs(signed_value) / gross_value if gross_value > 0 else abs(signed_shares) / gross_shares if gross_shares > 0 else 0.0
            top_names = [str(value) for value in group.get("insider_name", pd.Series(dtype=str)).dropna().unique().tolist()[:3]]
            buy_count = int((group.get("direction", pd.Series(dtype=str)) == "buy").sum())
            sell_count = int((group.get("direction", pd.Series(dtype=str)) == "sell").sum())
            records.append(
                DatasetEventRecord(
                    symbol=symbol,
                    event_date=event_date,
                    event_type="insider",
                    source="hetero_sec_insider_cache",
                    title=f"{symbol} insider flow {buy_count}B/{sell_count}S",
                    published_at=datetime.combine(event_date, time(16, 0)),
                    summary=f"{symbol} insider activity on {event_date.isoformat()}: {buy_count} buys, {sell_count} sells.",
                    sentiment=direction,
                    score=round(0.5 + (0.45 * conviction if direction == 'positive' else -0.45 * conviction if direction == 'negative' else 0.0), 4),
                    value=signed_value if gross_value > 0 else signed_shares,
                    metadata={
                        "buy_count": buy_count,
                        "sell_count": sell_count,
                        "gross_value": round(gross_value, 2),
                        "signed_value": round(signed_value, 2),
                        "gross_shares": round(gross_shares, 2),
                        "signed_shares": round(signed_shares, 2),
                        "top_insiders": top_names,
                    },
                )
            )
        return records

    def _load_macro_frame(self) -> pd.DataFrame | None:
        if self._macro_frame is not None:
            return self._macro_frame
        path = self._cache_file("fred", "macro", suffix=".parquet")
        self._macro_frame = self._read_parquet(path) if path else None
        if self._macro_frame is not None and not self._macro_frame.empty:
            self._macro_frame.index = pd.to_datetime(self._macro_frame.index, errors="coerce")
            self._macro_frame = self._macro_frame.sort_index()
        return self._macro_frame

    def _load_global_news_frame(self) -> pd.DataFrame | None:
        if self._global_news_frame is not None:
            return self._global_news_frame
        path = self._cache_file("global_news", "market", suffix=".parquet")
        self._global_news_frame = self._read_parquet(path) if path else None
        if self._global_news_frame is not None and not self._global_news_frame.empty:
            self._global_news_frame.index = pd.to_datetime(self._global_news_frame.index, errors="coerce")
            self._global_news_frame = self._global_news_frame.sort_index()
        return self._global_news_frame

    def _cache_file(self, bucket: str, symbol: str, *, suffix: str = ".parquet") -> Path | None:
        if not self.enabled:
            return None
        path = self.cache_dir / bucket / f"{symbol}{suffix}"
        return path if path.exists() else None

    def _read_parquet(self, path: Path | None) -> pd.DataFrame | None:
        if path is None or not path.exists():
            return None
        return pd.read_parquet(path)

    def _normalize_indexed_frame(self, frame: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
        working = frame.copy()
        working.index = pd.to_datetime(working.index, errors="coerce")
        working = working[working.index.notna()]
        working = working.sort_index()
        return working[(working.index >= pd.Timestamp(start_date)) & (working.index <= pd.Timestamp(end_date))]

    def _normalize_list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, tuple):
            return [str(item) for item in value if str(item).strip()]
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if text.startswith("[") and text.endswith("]"):
                try:
                    payload = json.loads(text)
                    if isinstance(payload, list):
                        return [str(item) for item in payload if str(item).strip()]
                except Exception:
                    return [text]
            return [text]
        return [str(value)]

    def _score_from_confidence(self, confidence: Any, sentiment_mean: Any) -> float:
        confidence_map = {"low": 0.58, "medium": 0.68, "high": 0.8}
        base = confidence_map.get(self._safe_str(confidence).lower(), 0.62)
        sentiment = abs(self._safe_float(sentiment_mean))
        return round(min(0.95, max(0.05, base + min(sentiment, 0.25))), 4)

    def _score_from_importance(self, importance: Any, confidence: Any) -> float:
        importance_map = {"low": 0.56, "medium": 0.68, "high": 0.82}
        confidence_boost = {"low": -0.04, "medium": 0.0, "high": 0.05}
        score = importance_map.get(self._safe_str(importance).lower(), 0.64)
        score += confidence_boost.get(self._safe_str(confidence).lower(), 0.0)
        return round(min(0.95, max(0.05, score)), 4)

    def _safe_str(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        return str(value)

    def _safe_float(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    def _safe_int(self, value: Any) -> int | None:
        parsed = self._safe_float(value)
        return int(parsed) if parsed is not None else None
