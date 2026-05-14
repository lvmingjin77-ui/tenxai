from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import (
    DatasetEventRecord,
    DailyBar,
    DailyBasicRecord,
    FinancialIndicatorRecord,
    InstrumentRecord,
    MarketId,
    TradeCalendarDay,
)


@dataclass(frozen=True)
class ImportedDataset:
    dataset_id: str
    market_id: MarketId
    name: str
    source_type: str
    benchmark_symbol: str
    instruments: list[InstrumentRecord]
    trade_calendar: list[TradeCalendarDay]
    daily_bars: list[DailyBar]
    daily_basics: list[DailyBasicRecord]
    financial_indicators: list[FinancialIndicatorRecord]
    events: list[DatasetEventRecord]
    index_bars: list[DailyBar]
    notes: list[str] = field(default_factory=list)
