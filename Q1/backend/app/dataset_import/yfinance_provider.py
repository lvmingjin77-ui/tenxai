from __future__ import annotations

from datetime import date, datetime
from typing import Dict, List
from uuid import uuid4

import pandas as pd

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


class YFinanceProvider:
    def fetch_dataset(self, request: DataImportRequest) -> ImportedDataset:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ValueError("当前环境未安装 yfinance，请先更新后端依赖。") from exc

        symbols = [item.strip().upper() for item in request.symbols if item.strip()]
        if not symbols:
            raise ValueError("美股导入至少需要一个 symbol。")

        instruments: List[InstrumentRecord] = []
        daily_bars: List[DailyBar] = []
        daily_basics: List[DailyBasicRecord] = []
        financial_indicators: List[FinancialIndicatorRecord] = []
        trade_dates = None

        for symbol in symbols[: request.symbol_limit]:
            ticker = yf.Ticker(symbol)
            try:
                history = ticker.history(
                    start=request.start_date.isoformat(),
                    end=(request.end_date).isoformat(),
                    auto_adjust=False,
                    actions=False,
                )
            except Exception as exc:
                raise ValueError(f"yfinance 拉取 {symbol} 行情失败：{str(exc)}") from exc
            if history is None or history.empty:
                continue
            history = history.reset_index()
            info = self._safe_info(ticker)
            instruments.append(
                InstrumentRecord(
                    symbol=symbol,
                    name=str(info.get("longName") or info.get("shortName") or symbol),
                    industry=str(info.get("sector") or info.get("industry") or "Unknown"),
                    listed_date=self._listed_date(info),
                    market=str(info.get("exchange") or "US"),
                )
            )
            if trade_dates is None:
                trade_dates = [self._to_date(item) for item in history["Date"].tolist()]
            market_cap = self._to_float(info.get("marketCap"))
            shares_outstanding = self._to_float(info.get("sharesOutstanding"))
            trailing_pe = self._to_float(info.get("trailingPE"))
            price_to_book = self._to_float(info.get("priceToBook"))
            roe = self._ratio_to_pct(info.get("returnOnEquity"))
            roa = self._ratio_to_pct(info.get("returnOnAssets"))
            gross_margin = self._ratio_to_pct(info.get("grossMargins"))
            earnings_growth = self._ratio_to_pct(info.get("earningsGrowth"))
            revenue_growth = self._ratio_to_pct(info.get("revenueGrowth"))
            ocf_to_or = self._ocf_to_or(info)

            financial_indicators.append(
                FinancialIndicatorRecord(
                    symbol=symbol,
                    ann_date=request.start_date,
                    end_date=request.start_date,
                    roe=roe,
                    roa=roa,
                    grossprofit_margin=gross_margin,
                    netprofit_yoy=earnings_growth,
                    revenue_yoy=revenue_growth,
                    ocf_to_or=ocf_to_or,
                )
            )

            for item in history.itertuples(index=False):
                trade_date = self._to_date(item.Date)
                close = self._to_float(item.Close)
                volume = int(self._to_float(item.Volume))
                turnover = round(close * volume, 2)
                daily_bars.append(
                    DailyBar(
                        trade_date=trade_date,
                        symbol=symbol,
                        open=self._to_float(item.Open),
                        high=self._to_float(item.High),
                        low=self._to_float(item.Low),
                        close=close,
                        volume=volume,
                        turnover=turnover,
                    )
                )
                total_mv = market_cap if market_cap > 0 else (close * shares_outstanding if shares_outstanding > 0 else None)
                daily_basics.append(
                    DailyBasicRecord(
                        trade_date=trade_date,
                        symbol=symbol,
                        turnover_rate=None,
                        pe_ttm=trailing_pe if trailing_pe > 0 else None,
                        pb=price_to_book if price_to_book > 0 else None,
                        total_mv=total_mv,
                        circ_mv=total_mv,
                    )
                )

        if not daily_bars or trade_dates is None:
            raise ValueError("yfinance 未返回有效美股日线数据。")

        benchmark_ticker = yf.Ticker(request.benchmark_symbol.upper())
        try:
            benchmark_history = benchmark_ticker.history(
                start=request.start_date.isoformat(),
                end=request.end_date.isoformat(),
                auto_adjust=False,
                actions=False,
            )
        except Exception as exc:
            raise ValueError(f"yfinance 拉取基准 {request.benchmark_symbol} 失败：{str(exc)}") from exc
        if benchmark_history is None or benchmark_history.empty:
            raise ValueError(f"基准指数 {request.benchmark_symbol} 未返回有效行情。")
        benchmark_history = benchmark_history.reset_index()
        index_bars = [
            DailyBar(
                trade_date=self._to_date(item.Date),
                symbol=request.benchmark_symbol.upper(),
                open=self._to_float(item.Open),
                high=self._to_float(item.High),
                low=self._to_float(item.Low),
                close=self._to_float(item.Close),
                volume=int(self._to_float(item.Volume)),
                turnover=round(self._to_float(item.Close) * self._to_float(item.Volume), 2),
            )
            for item in benchmark_history.itertuples(index=False)
        ]
        trade_calendar = [TradeCalendarDay(trade_date=item, is_open=True) for item in trade_dates]
        dataset_id = f"yfinance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        return ImportedDataset(
            dataset_id=dataset_id,
            market_id=MarketId.US_STOCK,
            name=request.name or f"yfinance 美股日线 {request.start_date.isoformat()}",
            source_type="yfinance",
            benchmark_symbol=request.benchmark_symbol.upper(),
            instruments=instruments,
            trade_calendar=trade_calendar,
            daily_bars=daily_bars,
            daily_basics=daily_basics,
            financial_indicators=financial_indicators,
            events=[],
            index_bars=index_bars,
            notes=[
                "该数据集由 yfinance 导入，适合在现有本地平台上新增美股研究与回测能力。",
                "当前财务字段主要来自 snapshot/info，适合研究原型，不适合作为严格 point-in-time 财务回测底座。",
            ],
        )

    def connectivity_check(self) -> Dict[str, object]:
        try:
            import yfinance as yf
        except ImportError as exc:
            return {
                "enabled": False,
                "base_url": "yfinance",
                "status": "error",
                "detail": f"当前环境未安装 yfinance：{exc}",
            }

        try:
            history = yf.Ticker("SPY").history(period="5d", auto_adjust=False, actions=False)
        except Exception as exc:
            return {
                "enabled": True,
                "base_url": "yfinance",
                "status": "error",
                "detail": f"yfinance 请求失败：{exc}",
            }
        row_count = 0 if history is None else len(history.index)
        if row_count == 0:
            return {
                "enabled": True,
                "base_url": "yfinance",
                "status": "error",
                "detail": "sector ETF 请求未返回有效价格，通常意味着当前环境没有可用外网或 Yahoo 响应异常。",
            }
        return {
            "enabled": True,
            "base_url": "yfinance",
            "status": "ok",
            "detail": f"sector ETF 价格可访问，SPY 最近返回 {row_count} 行。",
        }

    def fetch_sector_relative_strength(
        self,
        *,
        benchmark_symbol: str,
        sector_symbols: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[str, object]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise ValueError("当前环境未安装 yfinance，请先更新后端依赖。") from exc

        normalized_benchmark = benchmark_symbol.strip().upper()
        normalized_sectors = [item.strip().upper() for item in sector_symbols if item.strip()]
        if not normalized_sectors:
            raise ValueError("sector ETF 列表不能为空。")

        try:
            benchmark_history = yf.Ticker(normalized_benchmark).history(
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                auto_adjust=False,
                actions=False,
            )
        except Exception as exc:
            raise ValueError(f"sector ETF 基准请求失败：{exc}") from exc
        if benchmark_history is None or benchmark_history.empty:
            raise ValueError(f"未获取到基准 {normalized_benchmark} 的有效价格。")
        benchmark_close = benchmark_history["Close"].dropna()
        if benchmark_close.empty:
            raise ValueError(f"基准 {normalized_benchmark} 缺少收盘价数据。")

        sector_payloads: Dict[str, object] = {}
        for symbol in normalized_sectors:
            try:
                history = yf.Ticker(symbol).history(
                    start=start_date.isoformat(),
                    end=end_date.isoformat(),
                    auto_adjust=False,
                    actions=False,
                )
            except Exception as exc:
                sector_payloads[symbol] = {"status": "error", "detail": str(exc)}
                continue
            if history is None or history.empty:
                sector_payloads[symbol] = {"status": "empty"}
                continue
            closes = history["Close"].dropna()
            if closes.empty:
                sector_payloads[symbol] = {"status": "empty"}
                continue
            aligned = closes.to_frame("sector").join(benchmark_close.to_frame("benchmark"), how="inner")
            if aligned.empty:
                sector_payloads[symbol] = {"status": "unaligned"}
                continue
            sector_payloads[symbol] = {
                "status": "ok",
                "latest_close": round(float(aligned["sector"].iloc[-1]), 4),
                "latest_benchmark_close": round(float(aligned["benchmark"].iloc[-1]), 4),
                "relative_20d": round(self._relative_return(aligned, 20), 4),
                "relative_60d": round(self._relative_return(aligned, 60), 4),
                "points": len(aligned.index),
            }
        return {
            "source": "yfinance",
            "benchmark_symbol": normalized_benchmark,
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "sector_strength": sector_payloads,
        }

    def _safe_info(self, ticker):
        try:
            return ticker.info or {}
        except Exception:
            return {}

    def _listed_date(self, info) -> datetime.date:
        epoch = info.get("firstTradeDateEpochUtc")
        if epoch:
            return datetime.utcfromtimestamp(int(epoch)).date()
        return datetime(2000, 1, 1).date()

    def _ratio_to_pct(self, value) -> float | None:
        raw = self._to_float(value)
        if raw == 0:
            return None
        return round(raw * 100, 2)

    def _ocf_to_or(self, info) -> float | None:
        ocf = self._to_float(info.get("operatingCashflow"))
        revenue = self._to_float(info.get("totalRevenue"))
        if ocf <= 0 or revenue <= 0:
            return None
        return round(ocf / revenue * 100, 2)

    def _to_date(self, value) -> datetime.date:
        timestamp = pd.Timestamp(value)
        return timestamp.to_pydatetime().date()

    def _to_float(self, value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _relative_return(self, aligned_frame, lookback: int) -> float:
        if len(aligned_frame.index) <= lookback:
            return 0.0
        current_sector = float(aligned_frame["sector"].iloc[-1])
        current_benchmark = float(aligned_frame["benchmark"].iloc[-1])
        previous_sector = float(aligned_frame["sector"].iloc[-1 - lookback])
        previous_benchmark = float(aligned_frame["benchmark"].iloc[-1 - lookback])
        if previous_sector <= 0 or previous_benchmark <= 0 or current_benchmark <= 0:
            return 0.0
        sector_return = current_sector / previous_sector - 1
        benchmark_return = current_benchmark / previous_benchmark - 1
        return sector_return - benchmark_return
