from __future__ import annotations

import json
from typing import Any, Dict, List

from .schemas import CandidateSignal, MarketContext


def build_underwriting_system_prompt() -> str:
    return (
        "You are the underwriting phase of a single-agent US equity research workflow. "
        "This is a long-only backtest system. Work strictly inside the provided candidate set and point-in-time evidence. "
        "Your job is to underwrite each candidate and return structured JSON only. "
        "Do not invent data, do not discuss unavailable information, and do not emit markdown."
    )


def build_underwriting_user_prompt(
    *,
    trade_date: str,
    market_context: MarketContext,
    candidate_packets: List[Dict[str, Any]],
    current_weights: Dict[str, float],
) -> str:
    payload = {
        "phase": "candidate_underwriting",
        "trade_date": trade_date,
        "market_context": {
            "regime_label": market_context.regime_label,
            "market_view": market_context.market_view,
            "benchmark_20d_return": round(market_context.benchmark_20d_return, 4),
            "benchmark_60d_return": round(market_context.benchmark_60d_return, 4),
            "breadth": round(market_context.breadth, 4),
            "average_turnover": round(market_context.average_turnover, 2),
            "cash_bias": round(market_context.cash_bias, 4),
        },
        "current_weights": current_weights,
        "candidate_packets": candidate_packets,
        "required_json_schema": {
            "research_cards": [
                {
                    "symbol": "candidate symbol",
                    "action": "enter|hold|trim|reject",
                    "stance": "high_conviction|standard|monitor",
                    "score": "0..1 float",
                    "confidence": "0..1 float",
                    "horizon_days": "integer",
                    "summary": "one concise sentence",
                    "thesis": "short paragraph grounded in the packet",
                    "risks": ["list of specific risks"],
                    "invalidation": "single explicit invalidation condition",
                    "lead_event_type": "event type or empty string",
                }
            ]
        },
        "rules": [
            "Underwrite every provided candidate; do not omit symbols.",
            "Use enter for new deployable names, hold for incumbents still worth keeping, trim for weakened incumbents, reject for names that should not receive capital now.",
            "If blocked_reason is earnings_window, do not use enter for a fresh position.",
            "Keep score and confidence conservative and evidence-based.",
            "Do not propose portfolio weights here.",
            "Return strict JSON only.",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def compact_candidate_packet(candidate: CandidateSignal, extra: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate": {
            "symbol": candidate.symbol,
            "name": candidate.name,
            "industry": candidate.industry,
            "market": candidate.market,
            "latest_close": round(candidate.latest_close, 4),
            "latest_turnover": round(candidate.latest_turnover, 2),
            "total_score": round(candidate.total_score, 4),
            "factor_scores": {key: round(value, 4) for key, value in candidate.factor_scores.items()},
            "recent_event_count": candidate.recent_event_count,
            "recent_event_types": candidate.recent_event_types,
            "event_tags": candidate.event_tags,
            "blocked_reason": candidate.blocked_reason,
        },
        "context": extra,
    }
