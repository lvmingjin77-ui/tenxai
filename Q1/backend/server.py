from __future__ import annotations

import json
import mimetypes
import os
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from app.backtest import BacktestEngine
from app.data_store import DEFAULT_DB_PATH, SQLiteMarketStore
from app.llm_client import OpenAICompatibleClient
from app.schemas import BacktestConfig


ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = ROOT.parent / "frontend"


class Application:
    def __init__(self) -> None:
        self.store = SQLiteMarketStore()
        self.fixed_dataset_id = "hetero_trade_mas_cache_us_v1"
        self.llm_client = OpenAICompatibleClient()

    def bootstrap(self) -> dict:
        datasets = self.list_datasets()
        default_config = BacktestConfig(dataset_id=self.fixed_dataset_id)
        return {
            "datasets": datasets,
            "data_sources": self.store.get_data_sources(),
            "llm": self.llm_client.connectivity_check(),
            "default_config": asdict(default_config),
            "notes": [
                f"本地数据库：{DEFAULT_DB_PATH.name}",
                "当前默认数据集通过终端脚本导入，前端直接复用同一份 SQLite 数据。",
                "Q1 支持 local cache、Alpaca、FMP、yfinance 四条导入链路。",
            ],
        }

    def list_datasets(self) -> list[dict]:
        return [
            {
                "dataset_id": item.dataset_id,
                "name": item.name,
                "source_type": item.source_type,
                "benchmark_symbol": item.benchmark_symbol,
                "start_date": str(item.start_date),
                "end_date": str(item.end_date),
                "symbol_count": item.symbol_count,
                "notes": item.notes,
            }
            for item in self.store.list_datasets()
            if item.dataset_id == self.fixed_dataset_id
        ]

    def run_backtest(self, payload: dict) -> dict:
        config = BacktestConfig(
            dataset_id=self.fixed_dataset_id,
            initial_capital=float(payload.get("initial_capital") or 100000.0),
            rebalance_interval=int(payload.get("rebalance_interval") or 10),
            max_positions=int(payload.get("max_positions") or 3),
            max_position_weight=float(payload.get("max_position_weight") or 0.4),
            start_date=str(payload.get("start_date") or ""),
            end_date=str(payload.get("end_date") or ""),
        )
        bundle = self.store.load_dataset_bundle(config.dataset_id)
        engine = BacktestEngine(bundle, llm_client=self.llm_client)
        result = engine.run(config)
        return result.to_dict()


APP = Application()


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "Q1Server/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._write_json({"status": "ok"})
            return
        if parsed.path == "/api/bootstrap":
            self._write_json(APP.bootstrap())
            return
        if parsed.path == "/api/llm/health":
            self._write_json(APP.llm_client.connectivity_check())
            return
        if parsed.path == "/api/datasets":
            self._write_json({"datasets": APP.list_datasets()})
            return
        if parsed.path == "/api/data-sources":
            self._write_json({"data_sources": APP.store.get_data_sources()})
            return
        if parsed.path.startswith("/api/datasets/"):
            dataset_id = parsed.path.split("/api/datasets/", 1)[1]
            if not dataset_id:
                self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)
                return
            if dataset_id != APP.fixed_dataset_id:
                self._write_json({"error": f"dataset not found: {dataset_id}"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                self._write_json(APP.store.get_dataset_detail(dataset_id))
            except ValueError as exc:
                self._write_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
            return
        self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw.decode("utf-8") or "{}")
        if parsed.path == "/api/backtests/run":
            try:
                result = APP.run_backtest(payload)
                self._write_json(result)
            except Exception as exc:
                self._write_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._write_json({"error": "not_found"}, status=HTTPStatus.NOT_FOUND)

    def _serve_static(self, request_path: str) -> None:
        relative_path = request_path.lstrip("/") or "index.html"
        file_path = (FRONTEND_ROOT / relative_path).resolve()
        if not str(file_path).startswith(str(FRONTEND_ROOT.resolve())) or not file_path.exists():
            file_path = FRONTEND_ROOT / "index.html"
        media_type, _ = mimetypes.guess_type(str(file_path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", media_type or "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(file_path.read_bytes())

    def _write_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    host = "127.0.0.1"
    port = int(os.environ.get("Q1_PORT", "8010"))
    server = ThreadingHTTPServer((host, port), RequestHandler)
    print(f"Q1 server running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
