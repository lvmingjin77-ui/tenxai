import json
import os
import textwrap
import traceback
from io import StringIO
import contextlib
import openai

DEEPSEEK_API_KEY  = os.getenv("DEEPSEEK_API_KEY",  "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://openrouter.fans/v1")
MODEL             = "deepseek/deepseek-chat"

_SYS = """你是一名量化研究员，负责金融时序数据的特征工程。
通过 exec_python 工具执行Python代码，通过 log_decision 记录关键决策。
规则：
1. 所有数据通过 state 字典访问（state['df'], state['x_cols'] 等）
2. 严禁数据泄漏：统计量只在 state['train_dates'] 对应的行上计算
3. 代码出错自行修复，重试不超过3次
4. 关键结果写回 state（如 state['diagnose_report'] = df）"""

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "exec_python",
            "description": "执行Python代码，state为共享命名空间",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_decision",
            "description": "记录分析决策",
            "parameters": {
                "type": "object",
                "properties": {
                    "phase":         {"type": "string"},
                    "feature_group": {"type": "string"},
                    "action":        {"type": "string"},
                    "rationale":     {"type": "string"}
                },
                "required": ["phase", "feature_group", "action", "rationale"]
            }
        }
    }
]


class FeatureAgent:
    def __init__(self, state: dict, verbose: bool = True):
        self.client  = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.state   = state
        self.verbose = verbose
        self.decision_log: list[dict] = []
        self._msgs: list[dict] = [{"role": "system", "content": _SYS}]
        self._exec_ns: dict = {}

    def _exec(self, code: str) -> str:
        import numpy as np
        import pandas as pd
        from scipy import stats
        if not self._exec_ns:
            self._exec_ns = {
                "state": self.state,
                "np": np,
                "pd": pd,
                "stats": stats,
            }
        else:
            # 同一 phase 内保留中间变量，避免多步代码执行时变量丢失
            self._exec_ns["state"] = self.state
            self._exec_ns["np"] = np
            self._exec_ns["pd"] = pd
            self._exec_ns["stats"] = stats
        buf = StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(textwrap.dedent(code), self._exec_ns)
        except Exception:
            buf.write(f"[ERROR]\n{traceback.format_exc()}")
        out = buf.getvalue()
        return out[-4000:] if len(out) > 4000 else out

    def run(self, task: str, max_iter: int = 25) -> str:
        self._msgs.append({"role": "user", "content": task})
        for step in range(max_iter):
            resp = self.client.chat.completions.create(
                model=MODEL, messages=self._msgs, tools=_TOOLS, tool_choice="auto"
            )
            msg = resp.choices[0].message
            self._msgs.append(msg)

            if not msg.tool_calls:
                if self.verbose and msg.content:
                    print(f"\n{'='*55}\n[Agent 结论]\n{msg.content}\n{'='*55}")
                return msg.content or ""

            for tc in msg.tool_calls:
                fn   = tc.function.name
                args = json.loads(tc.function.arguments)
                if fn == "exec_python":
                    if self.verbose:
                        print(f"\n[Step {step+1}] 生成代码:\n{'-'*45}\n{args['code']}\n{'-'*45}")
                    result = self._exec(args["code"])
                    if self.verbose and result.strip():
                        print(f"[输出] {result}")
                elif fn == "log_decision":
                    self.decision_log.append(args)
                    result = "ok"
                    if self.verbose:
                        print(f"[决策] {args.get('phase')} | {args.get('action')} — {args.get('rationale')}")
                else:
                    result = f"未知工具: {fn}"
                self._msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result or "(empty)"})

        return "(max_iter)"

    def reset(self):
        self._msgs = [self._msgs[0]]
        self._exec_ns = {}
