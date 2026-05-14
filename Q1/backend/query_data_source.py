from __future__ import annotations

import argparse
import json
from datetime import date

from app.config import get_settings
from app.dataset_import.yfinance_provider import YFinanceProvider
from app.providers.eodhd_provider import EodhdProvider
from app.providers.finnhub_provider import FinnhubProvider
from app.providers.fred_provider import FredProvider
from app.providers.sec_api_provider import SecApiProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query Q1 auxiliary market data sources.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("health", help="检查所有辅助数据源连通性。")

    earnings = subparsers.add_parser("earnings-calendar", help="查询 Finnhub 财报日历。")
    earnings.add_argument("--symbol", default="")
    earnings.add_argument("--date-from", required=True)
    earnings.add_argument("--date-to", required=True)

    revisions = subparsers.add_parser("analyst-revisions", help="查询 EODHD 分析师修正。")
    revisions.add_argument("--symbol", required=True)

    shares = subparsers.add_parser("shares-float", help="查询 SEC API IO 历史流通股/float。")
    shares.add_argument("--symbol", required=True)

    macro = subparsers.add_parser("macro-regime", help="查询 FRED 宏观序列。")
    macro.add_argument("--date-from", required=True)
    macro.add_argument("--date-to", required=True)
    macro.add_argument("--series", default="VIXCLS,DGS2,DGS10,T10Y2Y,FEDFUNDS")

    sector = subparsers.add_parser("sector-strength", help="查询 yfinance 行业 ETF 相对强弱。")
    sector.add_argument("--date-from", required=True)
    sector.add_argument("--date-to", required=True)
    sector.add_argument("--benchmark", default="SPY")
    sector.add_argument("--symbols", default="XLK,XLF,XLE,XLV,XLY,XLP,XLI,XLB,XLU,XLRE,XLC")

    return parser.parse_args()


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main() -> None:
    args = parse_args()
    settings = get_settings()

    finnhub = FinnhubProvider(settings)
    eodhd = EodhdProvider(settings)
    sec_api = SecApiProvider(settings)
    fred = FredProvider(settings)
    yfinance = YFinanceProvider()

    if args.command == "health":
        payload = {
            "finnhub": finnhub.connectivity_check(),
            "eodhd": eodhd.connectivity_check(),
            "sec_api_io": sec_api.connectivity_check(),
            "fred": fred.connectivity_check(),
            "sector_etf": yfinance.connectivity_check(),
        }
    elif args.command == "earnings-calendar":
        payload = finnhub.fetch_earnings_calendar(
            symbol=args.symbol or None,
            date_from=parse_date(args.date_from),
            date_to=parse_date(args.date_to),
        )
    elif args.command == "analyst-revisions":
        payload = eodhd.fetch_analyst_revisions(args.symbol)
    elif args.command == "shares-float":
        payload = sec_api.fetch_historical_shares_float(args.symbol)
    elif args.command == "macro-regime":
        payload = fred.fetch_macro_regime(
            series_ids=args.series.split(","),
            date_from=parse_date(args.date_from),
            date_to=parse_date(args.date_to),
        )
    elif args.command == "sector-strength":
        payload = yfinance.fetch_sector_relative_strength(
            benchmark_symbol=args.benchmark,
            sector_symbols=args.symbols.split(","),
            start_date=parse_date(args.date_from),
            end_date=parse_date(args.date_to),
        )
    else:
        raise ValueError(f"未知命令：{args.command}")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
