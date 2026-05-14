from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from time import monotonic, sleep
from typing import Any, Dict, List
from uuid import uuid4

import requests

from ..config import Settings
from ..schemas import (
    DataImportRequest,
    DailyBar,
    DailyBasicRecord,
    FinancialIndicatorRecord,
    InstrumentRecord,
    MarketId,
    TradeCalendarDay,
)
from .base import ImportedDataset


SEC_FORMS = {"10-K", "10-Q", "20-F", "40-F", "8-K"}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


@dataclass(frozen=True)
class SecFact:
    value: float
    end_date: date
    filed_date: date
    fiscal_period: str | None
    form: str | None


@dataclass(frozen=True)
class SecMetadata:
    company_name: str | None
    industry: str | None
    shares_outstanding: float | None
    equity: float | None
    assets: float | None
    revenue: float | None
    gross_profit: float | None
    net_income: float | None
    operating_cashflow: float | None
    revenue_yoy: float | None
    net_income_yoy: float | None
    ann_date: date | None
    end_date: date | None
    effective_date: date | None


class AlpacaProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ticker_to_cik: Dict[str, str] | None = None
        self._last_sec_request_at = 0.0

    def fetch_dataset(self, request: DataImportRequest) -> ImportedDataset:
        if not self.settings.alpaca_api_key or not self.settings.alpaca_secret_key:
            raise ValueError("未配置 Alpaca API Key / Secret。")

        symbols = [item.strip().upper() for item in request.symbols if item.strip()]
        if not symbols:
            raise ValueError("美股导入至少需要一个 symbol。")

        instruments: List[InstrumentRecord] = []
        daily_bars: List[DailyBar] = []
        daily_basics: List[DailyBasicRecord] = []
        financial_indicators: List[FinancialIndicatorRecord] = []
        trade_dates_set: set[date] = set()

        for symbol in symbols[: request.symbol_limit]:
            asset = self._fetch_asset(symbol)
            metadata = self._fetch_sec_metadata(symbol)
            symbol_bars = self._fetch_bars(symbol, request.start_date, request.end_date)
            if not symbol_bars:
                continue

            instruments.append(self._build_instrument(symbol, asset, metadata))
            daily_bars.extend(symbol_bars)
            trade_dates_set.update(bar.trade_date for bar in symbol_bars)
            daily_basics.extend(self._build_daily_basics(symbol_bars, metadata))
            financial = self._build_financial_indicator(symbol, metadata)
            if financial is not None:
                financial_indicators.append(financial)

        if not daily_bars or not trade_dates_set:
            raise ValueError("Alpaca 未返回有效美股日线数据。请检查 symbol、日期区间和账户权限。")

        benchmark_bars = self._fetch_bars(
            request.benchmark_symbol.upper(),
            request.start_date,
            request.end_date,
        )
        if not benchmark_bars:
            raise ValueError(f"Alpaca 未返回基准 {request.benchmark_symbol} 的有效行情。")

        trade_dates = sorted(trade_dates_set)
        trade_calendar = [TradeCalendarDay(trade_date=item, is_open=True) for item in trade_dates]
        dataset_id = f"alpaca_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

        return ImportedDataset(
            dataset_id=dataset_id,
            market_id=MarketId.US_STOCK,
            name=request.name or f"Alpaca 美股日线 {request.start_date.isoformat()}",
            source_type="alpaca",
            benchmark_symbol=request.benchmark_symbol.upper(),
            instruments=instruments,
            trade_calendar=trade_calendar,
            daily_bars=daily_bars,
            daily_basics=daily_basics,
            financial_indicators=financial_indicators,
            events=[],
            index_bars=benchmark_bars,
            notes=[
                "该数据集由 Alpaca 导入，以日频价格为主，并使用 SEC 官方披露接口补充行业与基础财务。",
                "SEC 补充字段只会在已知披露日 effective_date 之后生效，避免把最新快照回填到整段历史。",
                "Alpaca 链路仍缺少完整事件层与多期历史股本，适合作为价格底座，不应单独承担正式 point-in-time 研究结论。",
            ],
        )

    def _alpaca_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key or "",
            "APCA-API-SECRET-KEY": self.settings.alpaca_secret_key or "",
        }

    def _sec_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self.settings.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        }

    def _fetch_asset(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.settings.alpaca_trading_base_url.rstrip('/')}/v2/assets/{symbol}"
        try:
            response = requests.get(url, headers=self._alpaca_headers(), timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"Alpaca 资产信息请求失败（{symbol}）：{str(exc)}") from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Alpaca 资产信息返回格式非法（{symbol}）。")
        return payload

    def _fetch_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[DailyBar]:
        url = f"{self.settings.alpaca_data_base_url.rstrip('/')}/stocks/bars"
        params = {
            "symbols": symbol,
            "timeframe": "1Day",
            "start": f"{start_date.isoformat()}T00:00:00Z",
            "end": f"{end_date.isoformat()}T23:59:59Z",
            "adjustment": "all",
            "feed": self.settings.alpaca_data_feed,
            "sort": "asc",
            "limit": 10000,
        }
        try:
            response = requests.get(url, headers=self._alpaca_headers(), params=params, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"Alpaca 行情请求失败（{symbol}）：{str(exc)}") from exc
        payload = response.json()
        bars_payload = payload.get("bars", {}).get(symbol, [])
        if not isinstance(bars_payload, list):
            raise ValueError(f"Alpaca 行情返回格式非法（{symbol}）。")

        bars: List[DailyBar] = []
        for item in bars_payload:
            timestamp = item.get("t")
            if not timestamp:
                continue
            trade_date = self._parse_timestamp(timestamp)
            close = self._to_float(item.get("c"))
            volume = int(self._to_float(item.get("v")))
            bars.append(
                DailyBar(
                    trade_date=trade_date,
                    symbol=symbol,
                    open=self._to_float(item.get("o")),
                    high=self._to_float(item.get("h")),
                    low=self._to_float(item.get("l")),
                    close=close,
                    volume=volume,
                    turnover=round(close * volume, 2),
                    adj_factor=1.0,
                )
            )
        return bars

    def _fetch_sec_metadata(self, symbol: str) -> SecMetadata:
        cik = self._resolve_cik(symbol)
        if cik is None:
            return SecMetadata(None, None, None, None, None, None, None, None, None, None, None, None, None, None)

        submissions = self._sec_get_json(SEC_SUBMISSIONS_URL.format(cik=cik))
        companyfacts = self._sec_get_json(SEC_COMPANYFACTS_URL.format(cik=cik))

        company_name = self._as_text(submissions.get("name"))
        industry = self._as_text(submissions.get("sicDescription"))

        shares = self._latest_fact_value(
            companyfacts,
            namespaces=["dei"],
            concepts=["EntityCommonStockSharesOutstanding", "CommonStockSharesOutstanding"],
        )
        equity = self._latest_fact_value(
            companyfacts,
            namespaces=["us-gaap"],
            concepts=["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
        )
        assets = self._latest_fact_value(
            companyfacts,
            namespaces=["us-gaap"],
            concepts=["Assets"],
        )
        revenue = self._latest_fact(
            companyfacts,
            namespaces=["us-gaap"],
            concepts=["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
        )
        gross_profit = self._latest_fact(
            companyfacts,
            namespaces=["us-gaap"],
            concepts=["GrossProfit"],
        )
        net_income = self._latest_fact(
            companyfacts,
            namespaces=["us-gaap"],
            concepts=["NetIncomeLoss", "ProfitLoss"],
        )
        operating_cashflow = self._latest_fact(
            companyfacts,
            namespaces=["us-gaap"],
            concepts=[
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            ],
        )

        revenue_yoy = self._year_over_year_growth(
            companyfacts,
            namespaces=["us-gaap"],
            concepts=["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
        )
        net_income_yoy = self._year_over_year_growth(
            companyfacts,
            namespaces=["us-gaap"],
            concepts=["NetIncomeLoss", "ProfitLoss"],
        )

        ann_date = revenue.filed_date if revenue else (net_income.filed_date if net_income else None)
        end_date = revenue.end_date if revenue else (net_income.end_date if net_income else None)
        effective_date = self._max_filed_date(
            shares,
            equity,
            assets,
            revenue,
            gross_profit,
            net_income,
            operating_cashflow,
        ) or ann_date

        return SecMetadata(
            company_name=company_name,
            industry=industry,
            shares_outstanding=shares.value if shares else None,
            equity=equity.value if equity else None,
            assets=assets.value if assets else None,
            revenue=revenue.value if revenue else None,
            gross_profit=gross_profit.value if gross_profit else None,
            net_income=net_income.value if net_income else None,
            operating_cashflow=operating_cashflow.value if operating_cashflow else None,
            revenue_yoy=revenue_yoy,
            net_income_yoy=net_income_yoy,
            ann_date=ann_date,
            end_date=end_date,
            effective_date=effective_date,
        )

    def _resolve_cik(self, symbol: str) -> str | None:
        if self._ticker_to_cik is None:
            payload = self._sec_get_json(SEC_TICKERS_URL)
            mapping: Dict[str, str] = {}
            for item in payload.values():
                ticker = self._as_text(item.get("ticker"))
                cik_str = item.get("cik_str")
                if ticker and cik_str is not None:
                    mapping[ticker.upper()] = f"{int(cik_str):010d}"
            self._ticker_to_cik = mapping
        return self._ticker_to_cik.get(symbol.upper())

    def _sec_get_json(self, url: str) -> Dict[str, Any]:
        elapsed = monotonic() - self._last_sec_request_at
        if elapsed < 0.12:
            sleep(0.12 - elapsed)
        try:
            response = requests.get(url, headers=self._sec_headers(), timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"SEC 元数据请求失败：{str(exc)}") from exc
        payload = response.json()
        self._last_sec_request_at = monotonic()
        if not isinstance(payload, dict):
            raise ValueError("SEC 元数据返回格式非法。")
        return payload

    def _build_instrument(
        self,
        symbol: str,
        asset: Dict[str, Any],
        metadata: SecMetadata,
    ) -> InstrumentRecord:
        exchange = str(asset.get("exchange") or "US")
        return InstrumentRecord(
            symbol=symbol,
            name=metadata.company_name or str(asset.get("name") or symbol),
            industry=metadata.industry or "Unknown",
            listed_date=date(2000, 1, 1),
            market=exchange,
        )

    def _build_daily_basics(
        self,
        bars: List[DailyBar],
        metadata: SecMetadata,
    ) -> List[DailyBasicRecord]:
        shares_outstanding = metadata.shares_outstanding or 0.0
        equity = metadata.equity or 0.0
        net_income = metadata.net_income or 0.0
        effective_date = metadata.effective_date or metadata.ann_date
        records: List[DailyBasicRecord] = []
        for bar in bars:
            if effective_date is None or bar.trade_date < effective_date:
                continue
            total_mv = bar.close * shares_outstanding if shares_outstanding > 0 else None
            pe_ttm = total_mv / net_income if total_mv and net_income > 0 else None
            pb = total_mv / equity if total_mv and equity > 0 else None
            records.append(
                DailyBasicRecord(
                    trade_date=bar.trade_date,
                    symbol=bar.symbol,
                    turnover_rate=round(bar.volume / shares_outstanding * 100, 4) if shares_outstanding > 0 else None,
                    pe_ttm=round(pe_ttm, 4) if pe_ttm else None,
                    pb=round(pb, 4) if pb else None,
                    total_mv=round(total_mv, 2) if total_mv else None,
                    circ_mv=round(total_mv, 2) if total_mv else None,
                )
            )
        return records

    def _build_financial_indicator(
        self,
        symbol: str,
        metadata: SecMetadata,
    ) -> FinancialIndicatorRecord | None:
        ann_date = metadata.effective_date or metadata.ann_date
        if ann_date is None or metadata.end_date is None:
            return None
        roe = self._ratio(metadata.net_income, metadata.equity)
        roa = self._ratio(metadata.net_income, metadata.assets)
        grossprofit_margin = self._ratio(metadata.gross_profit, metadata.revenue)
        ocf_to_or = self._ratio(metadata.operating_cashflow, metadata.revenue)
        if all(
            value is None
            for value in [roe, roa, grossprofit_margin, metadata.net_income_yoy, metadata.revenue_yoy, ocf_to_or]
        ):
            return None
        return FinancialIndicatorRecord(
            symbol=symbol,
            ann_date=ann_date,
            end_date=metadata.end_date,
            roe=roe,
            roa=roa,
            grossprofit_margin=grossprofit_margin,
            netprofit_yoy=metadata.net_income_yoy,
            revenue_yoy=metadata.revenue_yoy,
            ocf_to_or=ocf_to_or,
        )

    def _max_filed_date(self, *facts: SecFact | None) -> date | None:
        filed_dates = [fact.filed_date for fact in facts if fact is not None]
        return max(filed_dates) if filed_dates else None

    def _latest_fact_value(
        self,
        payload: Dict[str, Any],
        *,
        namespaces: List[str],
        concepts: List[str],
    ) -> SecFact | None:
        return self._latest_fact(payload, namespaces=namespaces, concepts=concepts)

    def _latest_fact(
        self,
        payload: Dict[str, Any],
        *,
        namespaces: List[str],
        concepts: List[str],
    ) -> SecFact | None:
        best: SecFact | None = None
        facts = payload.get("facts", {})
        for namespace in namespaces:
            namespace_payload = facts.get(namespace, {})
            for concept in concepts:
                concept_payload = namespace_payload.get(concept, {})
                units = concept_payload.get("units", {})
                for fact_list in units.values():
                    for item in fact_list:
                        fact = self._parse_sec_fact(item)
                        if fact is None:
                            continue
                        if best is None or (fact.end_date, fact.filed_date) > (best.end_date, best.filed_date):
                            best = fact
                if best is not None:
                    return best
        return None

    def _year_over_year_growth(
        self,
        payload: Dict[str, Any],
        *,
        namespaces: List[str],
        concepts: List[str],
    ) -> float | None:
        latest = self._latest_fact(payload, namespaces=namespaces, concepts=concepts)
        if latest is None:
            return None
        previous = self._prior_year_fact(payload, namespaces=namespaces, concepts=concepts, latest=latest)
        if previous is None or previous.value == 0:
            return None
        return round((latest.value - previous.value) / abs(previous.value) * 100, 2)

    def _prior_year_fact(
        self,
        payload: Dict[str, Any],
        *,
        namespaces: List[str],
        concepts: List[str],
        latest: SecFact,
    ) -> SecFact | None:
        candidate: SecFact | None = None
        facts = payload.get("facts", {})
        for namespace in namespaces:
            namespace_payload = facts.get(namespace, {})
            for concept in concepts:
                concept_payload = namespace_payload.get(concept, {})
                units = concept_payload.get("units", {})
                for fact_list in units.values():
                    for item in fact_list:
                        fact = self._parse_sec_fact(item)
                        if fact is None:
                            continue
                        days = (latest.end_date - fact.end_date).days
                        same_period = latest.fiscal_period and fact.fiscal_period and latest.fiscal_period == fact.fiscal_period
                        if days < 330 or days > 400:
                            continue
                        if not same_period:
                            continue
                        if candidate is None or abs(days - 365) < abs((latest.end_date - candidate.end_date).days - 365):
                            candidate = fact
                if candidate is not None:
                    return candidate
        return None

    def _parse_sec_fact(self, item: Dict[str, Any]) -> SecFact | None:
        form = self._as_text(item.get("form"))
        if form and form not in SEC_FORMS:
            return None
        end_text = self._as_text(item.get("end"))
        filed_text = self._as_text(item.get("filed"))
        value = item.get("val")
        if end_text is None or filed_text is None or value is None:
            return None
        try:
            end_date = date.fromisoformat(end_text)
            filed_date = date.fromisoformat(filed_text)
            numeric_value = float(value)
        except (TypeError, ValueError):
            return None
        return SecFact(
            value=numeric_value,
            end_date=end_date,
            filed_date=filed_date,
            fiscal_period=self._as_text(item.get("fp")),
            form=form,
        )

    def _parse_timestamp(self, value: str) -> date:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).date()

    def _ratio(self, numerator: float | None, denominator: float | None) -> float | None:
        if numerator is None or denominator is None or denominator <= 0:
            return None
        return round(numerator / denominator * 100, 2)

    def _as_text(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _to_float(self, value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
