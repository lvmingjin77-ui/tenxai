from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List
from uuid import uuid4

import requests

from ..config import Settings
from ..schemas import (
    DataImportRequest,
    DatasetEventRecord,
    DailyBar,
    DailyBasicRecord,
    FinancialIndicatorRecord,
    InstrumentRecord,
    MarketId,
    TradeCalendarDay,
)
from .base import ImportedDataset


@dataclass(frozen=True)
class FinancialSnapshot:
    ann_date: date
    end_date: date
    shares_outstanding: float | None
    revenue_ttm: float | None
    gross_profit_ttm: float | None
    net_income_ttm: float | None
    operating_cashflow_ttm: float | None
    equity: float | None
    assets: float | None
    revenue_yoy: float | None
    net_income_yoy: float | None


class FmpProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.fmp_base_url.rstrip("/")

    def fetch_dataset(self, request: DataImportRequest) -> ImportedDataset:
        if not self.settings.fmp_api_key:
            raise ValueError("未配置 FMP_API_KEY。")

        symbols = [item.strip().upper() for item in request.symbols if item.strip()]
        if not symbols:
            raise ValueError("FMP 导入至少需要一个 symbol。")

        instruments: List[InstrumentRecord] = []
        daily_bars: List[DailyBar] = []
        daily_basics: List[DailyBasicRecord] = []
        financial_indicators: List[FinancialIndicatorRecord] = []
        events: List[DatasetEventRecord] = []
        trade_dates_set: set[date] = set()
        import_notes: List[str] = []

        for symbol in symbols[: request.symbol_limit]:
            profile = self._fetch_profile(symbol)
            symbol_bars = self._fetch_bars(symbol, request.start_date, request.end_date)
            if not symbol_bars:
                continue

            quarterly_snapshots = self._fetch_financial_snapshots(symbol)
            instruments.append(self._build_instrument(symbol, profile))
            daily_bars.extend(symbol_bars)
            trade_dates_set.update(bar.trade_date for bar in symbol_bars)
            daily_basics.extend(self._build_daily_basics(symbol_bars, quarterly_snapshots))
            financial_indicators.extend(self._build_financial_indicators(symbol, quarterly_snapshots))
            symbol_events, symbol_notes = self._fetch_symbol_events(symbol, request.start_date, request.end_date)
            events.extend(symbol_events)
            import_notes.extend(symbol_notes)

        if not daily_bars or not trade_dates_set:
            raise ValueError("FMP 未返回有效美股日线数据。请检查 symbol、日期区间和账户权限。")

        benchmark_bars = self._fetch_bars(
            request.benchmark_symbol.upper(),
            request.start_date,
            request.end_date,
        )
        if not benchmark_bars:
            raise ValueError(f"FMP 未返回基准 {request.benchmark_symbol} 的有效行情。")

        trade_dates = sorted(trade_dates_set)
        trade_calendar = [TradeCalendarDay(trade_date=item, is_open=True) for item in trade_dates]
        dataset_id = f"fmp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        imported_symbols = sorted({item.symbol for item in instruments})

        return ImportedDataset(
            dataset_id=dataset_id,
            market_id=MarketId.US_STOCK,
            name=request.name or f"FMP 美股 Phase-1 {request.start_date.isoformat()}",
            source_type="fmp",
            benchmark_symbol=request.benchmark_symbol.upper(),
            instruments=instruments,
            trade_calendar=trade_calendar,
            daily_bars=daily_bars,
            daily_basics=daily_basics,
            financial_indicators=financial_indicators,
            events=events,
            index_bars=benchmark_bars,
            notes=[
                "该数据集由 FMP 导入，作为当前美股日频研究的首选底座，会优先导入价格、公司画像、按披露日生效的财务快照以及可访问的事件端点。",
                f"当前导入股票范围：{', '.join(imported_symbols[:10])}{' ...' if len(imported_symbols) > 10 else ''}",
                "财务与日度基本面优先按披露日 ann_date 生效，避免使用当前快照回填全历史。",
                "事件端点会尽量记录 published_at 和可见时间语义；缺少可靠发布时间的事件会被弱化或跳过。",
                *sorted(set(import_notes)),
            ],
        )

    def _fetch_profile(self, symbol: str) -> Dict[str, Any]:
        payload = self._get_json("profile", {"symbol": symbol.upper()})
        if isinstance(payload, list) and payload:
            return payload[0]
        return {}

    def _fetch_bars(self, symbol: str, start_date: date, end_date: date) -> List[DailyBar]:
        payload = self._get_json(
            "historical-price-eod/full",
            {
                "symbol": symbol.upper(),
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
        )
        history = payload if isinstance(payload, list) else None
        if not history:
            return []
        bars: List[DailyBar] = []
        for item in sorted(history, key=lambda row: row.get("date", "")):
            trade_date = self._parse_date(item.get("date"))
            if trade_date is None:
                continue
            close = self._to_float(item.get("close"))
            volume = int(self._to_float(item.get("volume")))
            bars.append(
                DailyBar(
                    trade_date=trade_date,
                    symbol=symbol.upper(),
                    open=self._to_float(item.get("open")),
                    high=self._to_float(item.get("high")),
                    low=self._to_float(item.get("low")),
                    close=close,
                    volume=volume,
                    turnover=round(close * volume, 2),
                    adj_factor=1.0,
                )
            )
        return bars

    def _fetch_financial_snapshots(self, symbol: str) -> List[FinancialSnapshot]:
        incomes = self._fetch_statement_rows("income-statement", symbol)
        balances = self._fetch_statement_rows("balance-sheet-statement", symbol)
        cashflows = self._fetch_statement_rows("cash-flow-statement", symbol)

        balance_by_date = {row.get("date"): row for row in balances if row.get("date")}
        cashflow_by_date = {row.get("date"): row for row in cashflows if row.get("date")}

        quarterly_rows = [
            row
            for row in incomes
            if row.get("date") and (row.get("fiscalYear") or row.get("calendarYear")) and row.get("period")
        ]
        quarterly_rows.sort(key=lambda row: row.get("date", ""))
        snapshots: List[FinancialSnapshot] = []
        for index, income_row in enumerate(quarterly_rows):
            statement_date = self._parse_date(income_row.get("date"))
            ann_date = self._parse_date(income_row.get("fillingDate")) or statement_date
            if statement_date is None or ann_date is None:
                continue
            balance_row = balance_by_date.get(income_row.get("date"), {})
            cashflow_row = cashflow_by_date.get(income_row.get("date"), {})
            ttm_slice = quarterly_rows[max(0, index - 3) : index + 1]
            if len(ttm_slice) < 4:
                revenue_ttm = None
                gross_profit_ttm = None
                net_income_ttm = None
            else:
                revenue_ttm = self._safe_sum(row.get("revenue") for row in ttm_slice)
                gross_profit_ttm = self._safe_sum(row.get("grossProfit") for row in ttm_slice)
                net_income_ttm = self._safe_sum(row.get("netIncome") for row in ttm_slice)
            cf_slice = quarterly_rows[max(0, index - 3) : index + 1]
            if len(cf_slice) < 4:
                operating_cashflow_ttm = None
            else:
                operating_cashflow_ttm = self._safe_sum(
                    cashflow_by_date.get(row.get("date"), {}).get("operatingCashFlow") for row in cf_slice
                )
            yoy_reference = quarterly_rows[index - 4] if index >= 4 else None
            revenue_yoy = self._growth(income_row.get("revenue"), yoy_reference.get("revenue") if yoy_reference else None)
            net_income_yoy = self._growth(
                income_row.get("netIncome"),
                yoy_reference.get("netIncome") if yoy_reference else None,
            )
            snapshots.append(
                FinancialSnapshot(
                    ann_date=ann_date,
                    end_date=statement_date,
                    shares_outstanding=self._resolve_shares_outstanding(income_row, balance_row),
                    revenue_ttm=revenue_ttm,
                    gross_profit_ttm=gross_profit_ttm,
                    net_income_ttm=net_income_ttm,
                    operating_cashflow_ttm=operating_cashflow_ttm,
                    equity=self._to_optional_float(
                        balance_row.get("totalStockholdersEquity") or balance_row.get("totalEquity")
                    ),
                    assets=self._to_optional_float(balance_row.get("totalAssets")),
                    revenue_yoy=revenue_yoy,
                    net_income_yoy=net_income_yoy,
                )
            )
        return snapshots

    def _fetch_statement_rows(self, path: str, symbol: str) -> List[Dict[str, Any]]:
        payload = self._get_json(path, {"symbol": symbol.upper(), "period": "quarter", "limit": 5})
        if isinstance(payload, list):
            return payload
        return []

    def _build_instrument(self, symbol: str, profile: Dict[str, Any]) -> InstrumentRecord:
        listed_date = self._parse_date(profile.get("ipoDate")) or date(2000, 1, 1)
        return InstrumentRecord(
            symbol=symbol.upper(),
            name=str(profile.get("companyName") or profile.get("name") or symbol.upper()),
            industry=str(profile.get("industry") or profile.get("sector") or "Unknown"),
            listed_date=listed_date,
            market=str(profile.get("exchangeShortName") or profile.get("exchange") or "US"),
        )

    def _build_daily_basics(
        self,
        bars: List[DailyBar],
        snapshots: List[FinancialSnapshot],
    ) -> List[DailyBasicRecord]:
        sorted_snapshots = sorted(snapshots, key=lambda item: item.ann_date)
        snapshot_index = 0
        active_snapshot: FinancialSnapshot | None = None
        records: List[DailyBasicRecord] = []
        for bar in bars:
            while snapshot_index < len(sorted_snapshots) and sorted_snapshots[snapshot_index].ann_date <= bar.trade_date:
                active_snapshot = sorted_snapshots[snapshot_index]
                snapshot_index += 1
            shares_outstanding = active_snapshot.shares_outstanding if active_snapshot else None
            total_mv = bar.close * shares_outstanding if shares_outstanding and shares_outstanding > 0 else None
            pe_ttm = (
                total_mv / active_snapshot.net_income_ttm
                if total_mv and active_snapshot and active_snapshot.net_income_ttm and active_snapshot.net_income_ttm > 0
                else None
            )
            pb = (
                total_mv / active_snapshot.equity
                if total_mv and active_snapshot and active_snapshot.equity and active_snapshot.equity > 0
                else None
            )
            turnover_rate = bar.volume / shares_outstanding * 100 if shares_outstanding and shares_outstanding > 0 else None
            records.append(
                DailyBasicRecord(
                    trade_date=bar.trade_date,
                    symbol=bar.symbol,
                    turnover_rate=round(turnover_rate, 4) if turnover_rate is not None else None,
                    pe_ttm=round(pe_ttm, 4) if pe_ttm is not None else None,
                    pb=round(pb, 4) if pb is not None else None,
                    total_mv=round(total_mv, 2) if total_mv is not None else None,
                    circ_mv=round(total_mv, 2) if total_mv is not None else None,
                )
            )
        return records

    def _build_financial_indicators(
        self,
        symbol: str,
        snapshots: List[FinancialSnapshot],
    ) -> List[FinancialIndicatorRecord]:
        records: List[FinancialIndicatorRecord] = []
        for item in snapshots:
            roe = self._ratio(item.net_income_ttm, item.equity)
            roa = self._ratio(item.net_income_ttm, item.assets)
            grossprofit_margin = self._ratio(item.gross_profit_ttm, item.revenue_ttm)
            ocf_to_or = self._ratio(item.operating_cashflow_ttm, item.revenue_ttm)
            if all(
                value is None
                for value in [roe, roa, grossprofit_margin, item.net_income_yoy, item.revenue_yoy, ocf_to_or]
            ):
                continue
            records.append(
                FinancialIndicatorRecord(
                    symbol=symbol.upper(),
                    ann_date=item.ann_date,
                    end_date=item.end_date,
                    roe=roe,
                    roa=roa,
                    grossprofit_margin=grossprofit_margin,
                    netprofit_yoy=item.net_income_yoy,
                    revenue_yoy=item.revenue_yoy,
                    ocf_to_or=ocf_to_or,
                )
            )
        return records

    def _fetch_symbol_events(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[List[DatasetEventRecord], List[str]]:
        events: List[DatasetEventRecord] = []
        notes: List[str] = []
        news_events, news_notes = self._fetch_news_events(symbol, start_date, end_date)
        events.extend(news_events)
        notes.extend(news_notes)
        earnings_events, earnings_notes = self._fetch_earnings_events(symbol, start_date, end_date)
        events.extend(earnings_events)
        notes.extend(earnings_notes)
        estimate_events, estimate_notes = self._fetch_estimate_events(symbol, start_date, end_date)
        events.extend(estimate_events)
        notes.extend(estimate_notes)
        transcript_events, transcript_notes = self._fetch_transcript_events(symbol, earnings_events)
        events.extend(transcript_events)
        notes.extend(transcript_notes)
        deduped: Dict[tuple[str, date, str, str], DatasetEventRecord] = {}
        for item in events:
            deduped[(item.symbol, item.event_date, item.event_type, item.title)] = item
        return list(deduped.values()), notes

    def _fetch_news_events(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[List[DatasetEventRecord], List[str]]:
        payload = self._get_optional_json(
            "news/stock",
            {
                "symbols": symbol.upper(),
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
                "limit": 200,
            },
            restriction_note="当前 FMP 订阅不支持 `news/stock`，本次导入跳过新闻事件。",
        )
        if payload is None:
            return [], ["当前 FMP 订阅不支持 `news/stock`，本次导入跳过新闻事件。"]
        if not isinstance(payload, list):
            return [], []
        events: List[DatasetEventRecord] = []
        for item in payload:
            published_at = self._parse_datetime(item.get("publishedDate"))
            event_date = published_at.date() if published_at else self._parse_date(item.get("date"))
            if event_date is None:
                continue
            summary = str(item.get("text") or item.get("title") or "")
            source = str(item.get("site") or "fmp_news")
            title = str(item.get("title") or f"{symbol.upper()} news")
            events.append(
                DatasetEventRecord(
                    symbol=symbol.upper(),
                    event_date=event_date,
                    event_type="news",
                    source=source,
                    title=title[:200],
                    published_at=published_at,
                    summary=summary[:1200],
                    sentiment="neutral",
                    score=0.55,
                    metadata={
                        "availability_semantics": "published_at",
                        "source_event_date": item.get("date"),
                        "url": item.get("url"),
                        "image": item.get("image"),
                    },
                )
            )
        return events, []

    def _fetch_earnings_events(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[List[DatasetEventRecord], List[str]]:
        payload = self._get_optional_json(
            "earnings-calendar",
            {
                "symbol": symbol.upper(),
                "from": start_date.isoformat(),
                "to": end_date.isoformat(),
            },
            restriction_note="当前 FMP 订阅不支持历史 `earnings-calendar` 查询，本次导入跳过财报日历事件。",
        )
        if payload is None:
            return [], ["当前 FMP 订阅不支持历史 `earnings-calendar` 查询，本次导入跳过财报日历事件。"]
        if not isinstance(payload, list):
            return [], []
        events: List[DatasetEventRecord] = []
        for item in payload:
            event_date = self._parse_date(item.get("date"))
            if event_date is None:
                continue
            eps_actual = self._to_optional_float(item.get("eps"))
            eps_estimated = self._to_optional_float(item.get("epsEstimated"))
            revenue_actual = self._to_optional_float(item.get("revenue"))
            revenue_estimated = self._to_optional_float(item.get("revenueEstimated"))
            surprise = max(
                abs(self._surprise_ratio(eps_actual, eps_estimated)),
                abs(self._surprise_ratio(revenue_actual, revenue_estimated)),
            )
            sentiment = "neutral"
            if eps_actual is not None and eps_estimated is not None:
                sentiment = "positive" if eps_actual >= eps_estimated else "negative"
            score = max(0.35, min(0.9, 0.5 + surprise * 0.8))
            quarter = int(item.get("quarter") or 0)
            year = int(item.get("year") or event_date.year)
            title = f"{symbol.upper()} earnings Q{quarter or '?'} {year}"
            events.append(
                DatasetEventRecord(
                    symbol=symbol.upper(),
                    event_date=event_date,
                    event_type="earnings",
                    source="fmp_earnings_calendar",
                    title=title,
                    summary=(
                        f"EPS actual {eps_actual}, estimate {eps_estimated}; "
                        f"revenue actual {revenue_actual}, estimate {revenue_estimated}."
                    ),
                    sentiment=sentiment,
                    score=round(score, 4),
                    value=round(surprise, 4),
                    metadata={
                        "availability_semantics": "release_date",
                        "quarter": quarter,
                        "year": year,
                        "time": item.get("time"),
                        "eps_actual": eps_actual,
                        "eps_estimated": eps_estimated,
                        "revenue_actual": revenue_actual,
                        "revenue_estimated": revenue_estimated,
                    },
                )
            )
        return events, []

    def _fetch_estimate_events(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> tuple[List[DatasetEventRecord], List[str]]:
        payload = self._get_optional_json(
            "analyst-estimates",
            {
                "symbol": symbol.upper(),
                "period": "quarter",
                "limit": 5,
            },
            restriction_note="当前 FMP 订阅不支持 `analyst-estimates`，本次导入跳过分析师预期事件。",
        )
        if payload is None:
            return [], ["当前 FMP 订阅不支持 `analyst-estimates`，本次导入跳过分析师预期事件。"]
        if not isinstance(payload, list) or not payload:
            return [], []
        return [], ["`analyst-estimates` 当前缺少可靠发布时间，已跳过以避免把估计快照误当成 point-in-time 事件。"]

    def _fetch_transcript_events(
        self,
        symbol: str,
        earnings_events: List[DatasetEventRecord],
    ) -> tuple[List[DatasetEventRecord], List[str]]:
        events: List[DatasetEventRecord] = []
        notes: List[str] = []
        requested_pairs: set[tuple[int, int]] = set()
        for earnings_event in earnings_events:
            metadata = earnings_event.metadata or {}
            quarter = int(metadata.get("quarter") or 0)
            year = int(metadata.get("year") or 0)
            if quarter <= 0 or year <= 0 or (year, quarter) in requested_pairs:
                continue
            requested_pairs.add((year, quarter))
            payload = self._get_optional_json(
                "earning-call-transcript",
                {"symbol": symbol.upper(), "year": year, "quarter": quarter},
                allow_404=True,
                restriction_note="当前 FMP 订阅不支持 `earning-call-transcript`，本次导入跳过电话会转录。",
            )
            if payload is None:
                notes.append("当前 FMP 订阅不支持 `earning-call-transcript`，本次导入跳过电话会转录。")
                continue
            transcript_rows = payload if isinstance(payload, list) else []
            for item in transcript_rows:
                event_date = self._parse_date(item.get("date")) or earnings_event.event_date
                title = str(item.get("title") or f"{symbol.upper()} earnings call Q{quarter} {year}")
                content = str(item.get("content") or "")
                if not content:
                    continue
                published_at = self._parse_datetime(item.get("date"))
                effective_event_date = published_at.date() if published_at else earnings_event.event_date
                events.append(
                    DatasetEventRecord(
                        symbol=symbol.upper(),
                        event_date=effective_event_date,
                        event_type="transcript",
                        source="fmp_transcript",
                        title=title[:200],
                        published_at=published_at,
                        summary=content[:1500],
                        sentiment="neutral",
                        score=0.58,
                        metadata={
                            "availability_semantics": "transcript_date",
                            "source_event_date": event_date.isoformat() if event_date else None,
                            "quarter": quarter,
                            "year": year,
                            "content_length": len(content),
                        },
                    )
                )
        return events, notes

    def _get_json(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
        *,
        allow_404: bool = False,
    ) -> Any:
        request_params = dict(params or {})
        request_params["apikey"] = self.settings.fmp_api_key or ""
        response = requests.get(f"{self.base_url}/{path}", params=request_params, timeout=45)
        if allow_404 and response.status_code == 404:
            return []
        try:
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"FMP 请求失败（{path}）：{str(exc)}") from exc
        payload = response.json()
        if isinstance(payload, dict) and payload.get("Error Message"):
            raise ValueError(f"FMP 请求失败（{path}）：{payload['Error Message']}")
        return payload

    def _get_optional_json(
        self,
        path: str,
        params: Dict[str, Any] | None = None,
        *,
        allow_404: bool = False,
        restriction_note: str,
    ) -> Any:
        try:
            return self._get_json(path, params, allow_404=allow_404)
        except ValueError as exc:
            message = str(exc)
            if "402" in message or "Restricted Endpoint" in message or "Premium Query Parameter" in message:
                return None
            raise

    def _safe_sum(self, values: Iterable[Any]) -> float | None:
        cleaned = [self._to_optional_float(value) for value in values]
        cleaned = [value for value in cleaned if value is not None]
        if not cleaned:
            return None
        return float(sum(cleaned))

    def _resolve_shares_outstanding(
        self,
        income_row: Dict[str, Any],
        balance_row: Dict[str, Any],
    ) -> float | None:
        for value in (
            balance_row.get("commonStockSharesOutstanding"),
            balance_row.get("commonStock"),
            income_row.get("weightedAverageShsOutDil"),
            income_row.get("weightedAverageShsOut"),
        ):
            shares = self._to_optional_float(value)
            if shares is not None and shares > 0:
                return shares
        return None

    def _ratio(self, numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator == 0:
            return None
        return round(numerator / denominator, 6)

    def _growth(self, current: Any, previous: Any) -> float | None:
        current_value = self._to_optional_float(current)
        previous_value = self._to_optional_float(previous)
        if current_value is None or previous_value is None or previous_value == 0:
            return None
        return round(current_value / previous_value - 1, 6)

    def _surprise_ratio(self, actual: float | None, estimate: float | None) -> float:
        if actual is None or estimate is None or estimate == 0:
            return 0.0
        return actual / estimate - 1

    def _to_float(self, value: Any) -> float:
        optional = self._to_optional_float(value)
        return optional if optional is not None else 0.0

    def _to_optional_float(self, value: Any) -> float | None:
        if value in (None, "", "None"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _parse_date(self, value: Any) -> date | None:
        if not value:
            return None
        text = str(value)
        for candidate in (text[:10], text):
            try:
                return date.fromisoformat(candidate)
            except ValueError:
                continue
        return None

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        text = str(value).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            parsed_date = self._parse_date(text)
            return datetime.combine(parsed_date, datetime.min.time()) if parsed_date else None
