from __future__ import annotations

from datetime import date
from typing import Dict, List, Tuple

from .research import UnderwritingAgent
from .schemas import BacktestConfig, DatasetBundle, ExperimentMetrics, ExperimentResult, RuntimeState


def _parse_date(value: str) -> date | None:
    if not value:
        return None
    year, month, day = [int(part) for part in value.split("-")]
    return date(year, month, day)


def _slice_indexes(bundle: DatasetBundle, config: BacktestConfig, warmup: int = 60) -> Tuple[int, int]:
    start_limit = _parse_date(config.start_date)
    end_limit = _parse_date(config.end_date)
    start_index = warmup
    end_index = len(bundle.trade_dates) - 1

    if start_limit is not None:
        for index, trade_day in enumerate(bundle.trade_dates):
            if trade_day >= start_limit:
                start_index = max(index, warmup)
                break
    if end_limit is not None:
        for index, trade_day in enumerate(bundle.trade_dates):
            if trade_day > end_limit:
                end_index = max(index - 1, start_index + 1)
                break
    return start_index, end_index


def _compute_weights(state: RuntimeState) -> Dict[str, float]:
    equity = state.cash + sum(state.positions.values())
    if equity <= 0:
        return {}
    return {
        symbol: value / equity
        for symbol, value in state.positions.items()
        if value > 1e-6
    }


def _compute_metrics(equity_curve: List[Dict[str, float | str]], trade_count: int) -> ExperimentMetrics:
    start_equity = float(equity_curve[0]["equity"])
    end_equity = float(equity_curve[-1]["equity"])
    start_benchmark = float(equity_curve[0]["benchmark_equity"])
    end_benchmark = float(equity_curve[-1]["benchmark_equity"])
    total_return = end_equity / start_equity - 1
    benchmark_return = end_benchmark / start_benchmark - 1
    days = max(len(equity_curve) - 1, 1)
    annual_return = (1 + total_return) ** (252 / days) - 1

    daily_returns: List[float] = []
    peak = start_equity
    max_drawdown = 0.0
    previous = start_equity
    for point in equity_curve[1:]:
        equity = float(point["equity"])
        daily_returns.append(equity / previous - 1)
        previous = equity
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1)

    sharpe_ratio = 0.0
    if daily_returns:
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((value - mean) ** 2 for value in daily_returns) / len(daily_returns)
        if variance > 0:
            sharpe_ratio = mean / (variance ** 0.5) * (252 ** 0.5)

    return ExperimentMetrics(
        total_return=total_return,
        annual_return=annual_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio,
        benchmark_return=benchmark_return,
        excess_return=total_return - benchmark_return,
        trade_count=trade_count,
    )


class BacktestEngine:
    def __init__(self, bundle: DatasetBundle, llm_client=None) -> None:
        self.bundle = bundle
        self.agent = UnderwritingAgent(bundle, llm_client=llm_client)

    def run(self, config: BacktestConfig) -> ExperimentResult:
        start_index, end_index = _slice_indexes(self.bundle, config)
        state = RuntimeState(cash=config.initial_capital)
        decisions = []
        trades: List[Dict[str, float | str]] = []
        equity_curve: List[Dict[str, float | str]] = []

        start_date = self.bundle.trade_dates[start_index]
        benchmark_start = self.bundle.benchmark_history[start_index].close
        benchmark_units = config.initial_capital / benchmark_start
        rebalance_indexes = set(range(start_index, end_index, max(config.rebalance_interval, 1)))
        pending_target: Dict[str, float] | None = None
        pending_decision_date = ""
        trade_count = 0

        equity_curve.append(
            {
                "trade_date": str(start_date),
                "equity": round(config.initial_capital, 2),
                "benchmark_equity": round(config.initial_capital, 2),
                "drawdown": 0.0,
            }
        )

        if start_index < end_index:
            first_decision = self.agent.decide(
                current_index=start_index,
                execution_index=start_index + 1,
                current_weights={},
                max_positions=config.max_positions,
                max_position_weight=config.max_position_weight,
            )
            decisions.append(first_decision)
            pending_target = first_decision.target_weights
            pending_decision_date = str(first_decision.decision_date)

        for index in range(start_index + 1, end_index + 1):
            current_date = self.bundle.trade_dates[index]

            for symbol, value in list(state.positions.items()):
                prev_close = self.bundle.price_history[symbol][index - 1].close
                open_price = self.bundle.price_history[symbol][index].open
                if prev_close > 0:
                    state.positions[symbol] = value * open_price / prev_close

            if pending_target is not None:
                equity_at_open = state.cash + sum(state.positions.values())
                new_positions: Dict[str, float] = {}
                target_symbols = set(pending_target)
                for symbol in sorted(set(state.positions) | target_symbols):
                    old_value = state.positions.get(symbol, 0.0)
                    new_value = equity_at_open * pending_target.get(symbol, 0.0)
                    delta = new_value - old_value
                    if abs(delta) > 1.0:
                        trade_count += 1
                        trades.append(
                            {
                                "trade_date": str(current_date),
                                "decision_date": pending_decision_date,
                                "symbol": symbol,
                                "side": "BUY" if delta > 0 else "SELL",
                                "notional": round(abs(delta), 2),
                            }
                        )
                    if new_value > 1.0:
                        new_positions[symbol] = new_value
                state.positions = new_positions
                state.cash = max(0.0, equity_at_open - sum(new_positions.values()))
                pending_target = None

            for symbol, value in list(state.positions.items()):
                open_price = self.bundle.price_history[symbol][index].open
                close_price = self.bundle.price_history[symbol][index].close
                if open_price > 0:
                    state.positions[symbol] = value * close_price / open_price

            equity = state.cash + sum(state.positions.values())
            benchmark_equity = benchmark_units * self.bundle.benchmark_history[index].close
            peak = max(float(point["equity"]) for point in equity_curve)
            equity_curve.append(
                {
                    "trade_date": str(current_date),
                    "equity": round(equity, 2),
                    "benchmark_equity": round(benchmark_equity, 2),
                    "drawdown": round(equity / peak - 1, 4),
                }
            )

            if index in rebalance_indexes and index < end_index:
                decision = self.agent.decide(
                    current_index=index,
                    execution_index=index + 1,
                    current_weights=_compute_weights(state),
                    max_positions=config.max_positions,
                    max_position_weight=config.max_position_weight,
                )
                decisions.append(decision)
                pending_target = decision.target_weights
                pending_decision_date = str(decision.decision_date)

        metrics = _compute_metrics(equity_curve, trade_count)
        notes = [
            f"当前数据集：{self.bundle.dataset_id}",
            "数据由 Q1 本地导入脚本生成并写入 SQLite，不再使用手工样例序列。",
            "回测按交易日推进，决策在当日收盘后生成，执行固定在下一交易日开盘。",
            "候选先经过相对强弱、基础面和事件打分，再进入单 Agent underwriting。",
            "财报窗口会阻断新开仓，Agent 只负责研究判断，仓位分配走确定性构造器。",
        ]
        return ExperimentResult(
            config=config,
            metrics=metrics,
            notes=notes,
            equity_curve=equity_curve,
            trades=trades,
            decisions=decisions,
        )
