from __future__ import annotations

from datetime import date, timedelta
import json
from statistics import pstdev
from typing import Any, Dict, List, Tuple

from .llm_client import OpenAICompatibleClient
from .prompts import build_underwriting_system_prompt, build_underwriting_user_prompt, compact_candidate_packet
from .schemas import (
    AgentDecision,
    Bar,
    CandidateSignal,
    DatasetBundle,
    DatasetEventRecord,
    FinancialIndicatorRecord,
    MarketContext,
    ResearchCard,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _lookback_return(history: List[Bar], current_index: int, lookback: int) -> float:
    if current_index < lookback:
        return 0.0
    base = history[current_index - lookback].close
    if base <= 0:
        return 0.0
    return history[current_index].close / base - 1


def _volatility(history: List[Bar], current_index: int, window: int) -> float:
    if current_index < window:
        return 0.0
    returns: List[float] = []
    for index in range(current_index - window + 1, current_index + 1):
        base = history[index - 1].close
        if base <= 0:
            continue
        returns.append(history[index].close / base - 1)
    if len(returns) < 2:
        return 0.0
    return pstdev(returns) * (252 ** 0.5)


def _latest_financial(records: List[FinancialIndicatorRecord], trade_date: date) -> FinancialIndicatorRecord | None:
    latest: FinancialIndicatorRecord | None = None
    for item in records:
        if item.ann_date <= trade_date:
            latest = item
        else:
            break
    return latest


def _recent_events(event_map: Dict[date, List[DatasetEventRecord]], trade_date: date) -> List[DatasetEventRecord]:
    recent: List[DatasetEventRecord] = []
    for available_date, items in event_map.items():
        if available_date > trade_date or (trade_date - available_date).days > 45:
            continue
        recent.extend(items)
    recent.sort(
        key=lambda item: (
            item.published_at.isoformat() if item.published_at else "",
            item.event_date.isoformat(),
            item.score,
        ),
        reverse=True,
    )
    return recent[:6]


def _upcoming_events(event_map: Dict[date, List[DatasetEventRecord]], trade_date: date) -> List[DatasetEventRecord]:
    upcoming: List[DatasetEventRecord] = []
    cutoff = trade_date + timedelta(days=21)
    for available_date, items in event_map.items():
        if available_date < trade_date or available_date > cutoff:
            continue
        upcoming.extend(items)
    upcoming.sort(key=lambda item: (item.event_date.isoformat(), item.score), reverse=False)
    return upcoming[:4]


def build_market_context(bundle: DatasetBundle, current_index: int) -> MarketContext:
    benchmark = bundle.benchmark_history
    benchmark_20d = _lookback_return(benchmark, current_index, 20)
    benchmark_60d = _lookback_return(benchmark, current_index, 60)
    breadth_flags = []
    turnover_values = []
    for symbol in bundle.instruments:
        history = bundle.price_history.get(symbol, [])
        if current_index >= len(history):
            continue
        breadth_flags.append(1.0 if _lookback_return(history, current_index, 20) > 0 else 0.0)
        turnover_values.append(history[current_index].turnover)
    breadth = sum(breadth_flags) / max(len(breadth_flags), 1)
    average_turnover = sum(turnover_values) / max(len(turnover_values), 1)

    if benchmark_20d > 0.08 and breadth >= 0.65:
        regime_label = "risk_on"
        cash_bias = 0.08
        market_view = "Benchmark trend and market breadth are aligned, so capital can stay relatively engaged."
    elif benchmark_20d < -0.04 or benchmark_60d < 0:
        regime_label = "defensive"
        cash_bias = 0.32
        market_view = "Trend support is weak and new entries should clear a higher evidence bar."
    else:
        regime_label = "selective"
        cash_bias = 0.18
        market_view = "Opportunity exists, but leadership is uneven and crowded names need cleaner setups."

    return MarketContext(
        regime_label=regime_label,
        market_view=market_view,
        benchmark_20d_return=benchmark_20d,
        benchmark_60d_return=benchmark_60d,
        breadth=breadth,
        average_turnover=average_turnover,
        cash_bias=cash_bias,
    )


def build_candidate_signals(bundle: DatasetBundle, current_index: int) -> List[CandidateSignal]:
    trade_date = bundle.trade_dates[current_index]
    benchmark_20d = _lookback_return(bundle.benchmark_history, current_index, 20)
    benchmark_60d = _lookback_return(bundle.benchmark_history, current_index, 60)
    signals: List[CandidateSignal] = []

    for symbol, instrument in bundle.instruments.items():
        history = bundle.price_history.get(symbol)
        if not history or current_index >= len(history):
            continue
        bar = history[current_index]
        daily_basic = bundle.daily_basic_history.get(symbol, {}).get(trade_date)
        financial = _latest_financial(bundle.financial_history.get(symbol, []), trade_date)
        recent = _recent_events(bundle.event_history.get(symbol, {}), trade_date)
        upcoming = _upcoming_events(bundle.event_history.get(symbol, {}), trade_date)

        stock_20d = _lookback_return(history, current_index, 20)
        stock_60d = _lookback_return(history, current_index, 60)
        volatility_20d = _volatility(history, current_index, 20)
        event_score = sum(item.score for item in recent[:2]) / max(min(len(recent), 2), 1) if recent else 0.0
        quality_score = (
            (financial.roe or 0.0) * 0.35
            + (financial.grossprofit_margin or 0.0) * 0.35
            + (financial.ocf_to_or or 0.0) * 0.3
        ) if financial else 0.0
        growth_score = (
            (financial.revenue_yoy or 0.0) * 0.45
            + (financial.netprofit_yoy or 0.0) * 0.55
        ) if financial else 0.0
        value_score = 0.0
        if daily_basic and daily_basic.pe_ttm and daily_basic.pe_ttm > 0:
            value_score += 1 / min(daily_basic.pe_ttm, 80)
        if daily_basic and daily_basic.pb and daily_basic.pb > 0:
            value_score += 1 / min(daily_basic.pb, 20)

        total_score = (
            0.26 * (stock_20d - benchmark_20d)
            + 0.2 * (stock_60d - benchmark_60d)
            + 0.18 * quality_score
            + 0.14 * growth_score
            + 0.1 * value_score
            + 0.12 * event_score
            - 0.08 * volatility_20d
        )

        lead_event = recent[0] if recent else None
        event_tags = []
        if lead_event:
            event_tags.append(f"{lead_event.event_type}:{lead_event.title[:36]}")
        if upcoming:
            event_tags.append(f"upcoming:{upcoming[0].event_type}")

        blocked_reason = ""
        for event in upcoming:
            if event.event_type == "earnings":
                blocked_reason = "earnings_window"
                break

        signals.append(
            CandidateSignal(
                symbol=symbol,
                name=instrument.name,
                industry=instrument.industry,
                market=instrument.market,
                latest_close=bar.close,
                latest_turnover=bar.turnover,
                total_score=total_score,
                factor_scores={
                    "relative_20d": stock_20d - benchmark_20d,
                    "relative_60d": stock_60d - benchmark_60d,
                    "quality": quality_score,
                    "growth": growth_score,
                    "value": value_score,
                    "event_score": event_score,
                    "volatility_20d": -volatility_20d,
                },
                recent_event_count=len(recent),
                recent_event_types=[item.event_type for item in recent[:3]],
                event_tags=event_tags[:4],
                blocked_reason=blocked_reason,
            )
        )
    signals.sort(key=lambda item: item.total_score, reverse=True)
    return signals


class UnderwritingAgent:
    def __init__(self, bundle: DatasetBundle, llm_client: OpenAICompatibleClient | None = None) -> None:
        self.bundle = bundle
        self.llm_client = llm_client or OpenAICompatibleClient()

    def decide(
        self,
        *,
        current_index: int,
        execution_index: int,
        current_weights: Dict[str, float],
        max_positions: int,
        max_position_weight: float,
    ) -> AgentDecision:
        decision_date = self.bundle.trade_dates[current_index]
        execution_date = self.bundle.trade_dates[execution_index]
        market_context = build_market_context(self.bundle, current_index)
        candidate_signals = build_candidate_signals(self.bundle, current_index)
        research_cards = self._underwrite(
            trade_date=decision_date,
            current_weights=current_weights,
            market_context=market_context,
            current_index=current_index,
            candidate_signals=candidate_signals,
        )
        selected_symbols, target_weights, cash_weight = self._construct_portfolio(
            market_context=market_context,
            research_cards=research_cards,
            max_positions=max_positions,
            max_position_weight=max_position_weight,
        )
        summary = (
            f"{market_context.regime_label} 环境下建议持有 "
            f"{', '.join(selected_symbols) if selected_symbols else '现金'}，"
            f"目标现金 {round(cash_weight * 100)}%。"
        )
        return AgentDecision(
            decision_date=decision_date,
            execution_date=execution_date,
            market_context=market_context,
            candidate_signals=candidate_signals,
            research_cards=research_cards,
            selected_symbols=selected_symbols,
            target_weights=target_weights,
            cash_weight=cash_weight,
            summary=summary,
        )

    def _underwrite(
        self,
        *,
        trade_date: date,
        current_weights: Dict[str, float],
        market_context: MarketContext,
        current_index: int,
        candidate_signals: List[CandidateSignal],
    ) -> List[ResearchCard]:
        candidate_packets = [
            compact_candidate_packet(
                signal,
                self._build_candidate_context(
                    symbol=signal.symbol,
                    trade_date=trade_date,
                    current_index=current_index,
                    is_incumbent=current_weights.get(signal.symbol, 0.0) > 0,
                ),
            )
            for signal in candidate_signals
        ]
        response = self.llm_client.create_chat_completion(
            system_prompt=build_underwriting_system_prompt(),
            user_prompt=build_underwriting_user_prompt(
                trade_date=trade_date.isoformat(),
                market_context=market_context,
                candidate_packets=candidate_packets,
                current_weights={key: round(value, 4) for key, value in current_weights.items()},
            ),
            temperature=0.05,
            max_tokens=2400,
        )
        return self._parse_cards(candidate_signals, response["content"])

    def _build_candidate_context(
        self,
        *,
        symbol: str,
        trade_date: date,
        current_index: int,
        is_incumbent: bool,
    ) -> Dict[str, Any]:
        history = self.bundle.price_history[symbol]
        financial = _latest_financial(self.bundle.financial_history.get(symbol, []), trade_date)
        daily_basic = self.bundle.daily_basic_history.get(symbol, {}).get(trade_date)
        recent = _recent_events(self.bundle.event_history.get(symbol, {}), trade_date)
        upcoming = _upcoming_events(self.bundle.event_history.get(symbol, {}), trade_date)
        return {
            "is_incumbent": is_incumbent,
            "one_day_return": round(_lookback_return(history, current_index, 1), 4),
            "twenty_day_return": round(_lookback_return(history, current_index, 20), 4),
            "sixty_day_return": round(_lookback_return(history, current_index, 60), 4),
            "volatility_20d": round(_volatility(history, current_index, 20), 4),
            "pe_ttm": daily_basic.pe_ttm if daily_basic else None,
            "pb": daily_basic.pb if daily_basic else None,
            "total_mv": daily_basic.total_mv if daily_basic else None,
            "roe": financial.roe if financial else None,
            "roa": financial.roa if financial else None,
            "grossprofit_margin": financial.grossprofit_margin if financial else None,
            "netprofit_yoy": financial.netprofit_yoy if financial else None,
            "revenue_yoy": financial.revenue_yoy if financial else None,
            "ocf_to_or": financial.ocf_to_or if financial else None,
            "recent_events": [
                {
                    "event_date": item.event_date.isoformat(),
                    "event_type": item.event_type,
                    "title": item.title,
                    "score": round(item.score, 4),
                    "summary": item.summary[:180],
                }
                for item in recent[:3]
            ],
            "upcoming_events": [
                {
                    "event_date": item.event_date.isoformat(),
                    "event_type": item.event_type,
                    "title": item.title,
                    "score": round(item.score, 4),
                }
                for item in upcoming[:2]
            ],
        }

    def _parse_cards(self, candidate_signals: List[CandidateSignal], content: str) -> List[ResearchCard]:
        payload = json.loads(self._extract_json_text(content))
        cards_payload = payload.get("research_cards")
        if not isinstance(cards_payload, list) or not cards_payload:
            raise RuntimeError("LLM underwriting response does not include research_cards.")
        signal_map = {item.symbol: item for item in candidate_signals}
        cards: List[ResearchCard] = []
        for item in cards_payload:
            symbol = str(item.get("symbol") or "")
            if symbol not in signal_map:
                continue
            action = str(item.get("action") or "reject").strip().lower()
            stance = str(item.get("stance") or "monitor").strip().lower()
            if action not in {"enter", "hold", "trim", "reject"}:
                action = "reject"
            if stance not in {"high_conviction", "standard", "monitor"}:
                stance = "monitor"
            risks = item.get("risks") or []
            if not isinstance(risks, list):
                risks = [str(risks)]
            cards.append(
                ResearchCard(
                    symbol=symbol,
                    action=action,
                    stance=stance,
                    score=_clamp(float(item.get("score") or 0.0), 0.0, 1.0),
                    confidence=_clamp(float(item.get("confidence") or 0.0), 0.0, 1.0),
                    horizon_days=max(int(item.get("horizon_days") or 10), 1),
                    summary=str(item.get("summary") or ""),
                    thesis=str(item.get("thesis") or ""),
                    risks=[str(value) for value in risks][:4],
                    invalidation=str(item.get("invalidation") or ""),
                    lead_event_type=str(item.get("lead_event_type") or ""),
                )
            )
        if len(cards) != len(candidate_signals):
            missing = [item.symbol for item in candidate_signals if item.symbol not in {card.symbol for card in cards}]
            raise RuntimeError(f"LLM underwriting omitted candidates: {', '.join(missing)}")
        return cards

    def _extract_json_text(self, content: str) -> str:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            raise RuntimeError("LLM response does not contain a JSON object.")
        return content[start : end + 1]

    def _construct_portfolio(
        self,
        *,
        market_context: MarketContext,
        research_cards: List[ResearchCard],
        max_positions: int,
        max_position_weight: float,
    ) -> Tuple[List[str], Dict[str, float], float]:
        investable = [item for item in research_cards if item.action in {"enter", "hold", "trim"}]
        investable.sort(key=lambda item: (item.score, item.confidence), reverse=True)
        selected = investable[:max_positions]
        if not selected:
            return [], {}, 1.0

        gross_target = min(1.0 - market_context.cash_bias, max_positions * max_position_weight)
        raw = {item.symbol: max(item.score * item.confidence, 0.01) for item in selected}
        total = sum(raw.values())
        weights: Dict[str, float] = {}
        for symbol, score in raw.items():
            weights[symbol] = min(max_position_weight, gross_target * score / total)

        allocated = sum(weights.values())
        cash_weight = max(0.0, 1.0 - allocated)
        return list(weights.keys()), weights, cash_weight
