"""Local HTML test server (standard library only).

    python -m promptmaker.server --port 8765
    -> open http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .core import GenerateOptions, PromptMaker
from .safety import SafetyError

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


def make_handler(maker: PromptMaker):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data) -> None:
            self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"))

        def _vision(self, method: str):
            from . import vision_api  # optional feature, loaded on first use

            try:
                payload = {}
                if method == "POST":
                    length = int(self.headers.get("Content-Length", 0))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                code, data = vision_api.handle(maker, method, self.path, payload)
            except ImportError as exc:
                code, data = 400, {"error": f"이미지 분석에 필요한 패키지가 없습니다: {exc}"}
            except Exception as exc:
                code, data = 400, {"error": f"{type(exc).__name__}: {exc}"}
            self._json(code, data)

        def do_GET(self):
            if self.path.startswith("/api/vision/"):
                return self._vision("GET")
            if self.path.startswith("/api/info"):
                return self._json(200, maker.info())
            if self.path.startswith("/api/reload"):
                maker.reload()
                return self._json(200, {"ok": True})
            if self.path in ("/", "/index.html"):
                with open(os.path.join(WEB_DIR, "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path.startswith("/api/vision/"):
                return self._vision("POST")
            if not self.path.startswith("/api/generate"):
                return self._json(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
                count = max(1, min(int(payload.pop("count", 1)), 20))
                user_input = payload.pop("user_input", "")
                opts = GenerateOptions.from_dict(payload)
                results = maker.generate_batch(user_input, count, opts)
                self._json(200, {"results": [r.to_dict() for r in results]})
            except SafetyError as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:  # show the message in the page
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt, *args):  # quieter console
            pass

    return Handler


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="PromptMaker HTML 테스트 서버")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--user-data", default=None, help="사용자 데이터 폴더 (선택)")
    args = parser.parse_args(argv)
    maker = PromptMaker(user_data_dir=args.user_data)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(maker))
    print(f"PromptMaker 테스트 페이지: http://{args.host}:{args.port}  (종료: Ctrl+C)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
