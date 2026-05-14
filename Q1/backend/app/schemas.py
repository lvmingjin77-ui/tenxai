from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Optional


class MarketId(str, Enum):
    US_STOCK = "us_stock"


@dataclass(frozen=True)
class Bar:
    trade_date: date
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    turnover: float
    adj_factor: float = 1.0
    is_suspended: bool = False
    is_st: bool = False


DailyBar = Bar


@dataclass(frozen=True)
class TradeCalendarDay:
    trade_date: date
    is_open: bool = True
    pretrade_date: Optional[date] = None


@dataclass(frozen=True)
class DailyBasicRecord:
    trade_date: date
    symbol: str
    turnover_rate: float | None = None
    pe_ttm: float | None = None
    pb: float | None = None
    total_mv: float | None = None
    circ_mv: float | None = None


@dataclass(frozen=True)
class FinancialIndicatorRecord:
    symbol: str
    ann_date: date
    end_date: date
    roe: float | None = None
    roa: float | None = None
    grossprofit_margin: float | None = None
    netprofit_yoy: float | None = None
    revenue_yoy: float | None = None
    ocf_to_or: float | None = None


@dataclass(frozen=True)
class DatasetEventRecord:
    symbol: str
    event_date: date
    event_type: str
    source: str
    title: str
    published_at: Optional[datetime] = None
    summary: str = ""
    sentiment: str = "neutral"
    score: float = 0.5
    value: float | None = None
    metadata: Dict[str, object] | None = None


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    industry: str
    listed_date: date
    market: str
    delisted_date: Optional[date] = None


InstrumentRecord = Instrument


@dataclass(frozen=True)
class DataImportRequest:
    market_id: MarketId = MarketId.US_STOCK
    name: str | None = None
    start_date: date = date(2022, 10, 5)
    end_date: date = date(2023, 6, 10)
    symbols: List[str] = field(default_factory=list)
    symbol_limit: int = 20
    benchmark_symbol: str = "SPY"


@dataclass(frozen=True)
class DatasetSummary:
    dataset_id: str
    name: str
    source_type: str
    benchmark_symbol: str
    start_date: date
    end_date: date
    symbol_count: int
    notes: List[str]


@dataclass(frozen=True)
class DatasetBundle:
    dataset_id: str
    name: str
    benchmark_symbol: str
    trade_dates: List[date]
    benchmark_history: List[Bar]
    instruments: Dict[str, Instrument]
    price_history: Dict[str, List[Bar]]
    daily_basic_history: Dict[str, Dict[date, DailyBasicRecord]]
    financial_history: Dict[str, List[FinancialIndicatorRecord]]
    event_history: Dict[str, Dict[date, List[DatasetEventRecord]]]


@dataclass(frozen=True)
class CandidateSignal:
    symbol: str
    name: str
    industry: str
    market: str
    latest_close: float
    latest_turnover: float
    total_score: float
    factor_scores: Dict[str, float]
    recent_event_count: int
    recent_event_types: List[str]
    event_tags: List[str]
    blocked_reason: str = ""


@dataclass(frozen=True)
class ResearchCard:
    symbol: str
    action: str
    stance: str
    score: float
    confidence: float
    horizon_days: int
    summary: str
    thesis: str
    risks: List[str]
    invalidation: str
    lead_event_type: str


@dataclass(frozen=True)
class MarketContext:
    regime_label: str
    market_view: str
    benchmark_20d_return: float
    benchmark_60d_return: float
    breadth: float
    average_turnover: float
    cash_bias: float


@dataclass(frozen=True)
class AgentDecision:
    decision_date: date
    execution_date: date
    market_context: MarketContext
    candidate_signals: List[CandidateSignal]
    research_cards: List[ResearchCard]
    selected_symbols: List[str]
    target_weights: Dict[str, float]
    cash_weight: float
    summary: str


@dataclass(frozen=True)
class BacktestConfig:
    dataset_id: str = "hetero_trade_mas_cache_us_v1"
    initial_capital: float = 100000.0
    rebalance_interval: int = 10
    max_positions: int = 3
    max_position_weight: float = 0.35
    start_date: str = "2022-10-05"
    end_date: str = "2023-06-10"


@dataclass
class ExperimentMetrics:
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    benchmark_return: float
    excess_return: float
    trade_count: int


@dataclass
class ExperimentResult:
    config: BacktestConfig
    metrics: ExperimentMetrics
    notes: List[str]
    equity_curve: List[Dict[str, float | str]]
    trades: List[Dict[str, float | str]]
    decisions: List[AgentDecision]

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for decision in payload["decisions"]:
            decision["decision_date"] = str(decision["decision_date"])
            decision["execution_date"] = str(decision["execution_date"])
            decision["market_context"] = {
                **decision["market_context"],
                "benchmark_20d_return": round(decision["market_context"]["benchmark_20d_return"], 4),
                "benchmark_60d_return": round(decision["market_context"]["benchmark_60d_return"], 4),
                "breadth": round(decision["market_context"]["breadth"], 4),
                "average_turnover": round(decision["market_context"]["average_turnover"], 2),
                "cash_bias": round(decision["market_context"]["cash_bias"], 4),
            }
            for item in decision["candidate_signals"]:
                item["latest_close"] = round(item["latest_close"], 2)
                item["latest_turnover"] = round(item["latest_turnover"], 2)
                item["total_score"] = round(item["total_score"], 4)
                item["factor_scores"] = {
                    key: round(value, 4) for key, value in item["factor_scores"].items()
                }
            for card in decision["research_cards"]:
                card["score"] = round(card["score"], 4)
                card["confidence"] = round(card["confidence"], 4)
            decision["target_weights"] = {
                key: round(value, 4) for key, value in decision["target_weights"].items()
            }
            decision["cash_weight"] = round(decision["cash_weight"], 4)
        return payload


@dataclass
class RuntimeState:
    positions: Dict[str, float] = field(default_factory=dict)
    cash: float = 0.0
