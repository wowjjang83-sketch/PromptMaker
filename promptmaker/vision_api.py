"""HTTP routes for image analysis used by ``promptmaker.server`` (``/api/vision/*``)."""
from __future__ import annotations

from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from . import vision


def handle(maker, method: str, path: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
    url = urlparse(path)
    route = url.path[len("/api/vision"):].strip("/")
    query = {k: v[-1] for k, v in parse_qs(url.query).items()}

    if method == "GET" and route == "config":
        cfg = vision.load_config()
        return 200, {"config": cfg.to_dict(hide_key=True), "config_path": vision.CONFIG_PATH,
                     "backends": list(vision.BACKENDS), "default_model_id": vision.DEFAULT_MODEL_ID,
                     "check": vision.check_model_path(cfg.model_path)}
    if method == "POST" and route == "config":
        old = vision.load_config()
        data = dict(payload.get("config", payload))
        if data.get("api_key") == "***":
            data["api_key"] = old.api_key
        cfg = vision.VisionConfig.from_dict({**old.to_dict(), **data})
        vision.save_config(cfg)
        vision.unload_backends()
        return 200, {"ok": True, "config": cfg.to_dict(hide_key=True), "check": vision.check_model_path(cfg.model_path)}
    if method == "GET" and route == "browse":
        try:
            return 200, vision.browse(query.get("path", ""))
        except (FileNotFoundError, NotADirectoryError):
            return 400, {"error": "폴더를 찾을 수 없습니다."}
    if method == "GET" and route == "check":
        return 200, vision.check_model_path(query.get("path", ""))
    if method == "POST" and route == "analyze":
        if not payload.get("image"):
            return 400, {"error": "이미지가 없습니다."}
        analyzer = vision.ImageAnalyzer(maker)
        result = analyzer.analyze(payload["image"])
        if payload.get("apply"):
            result["report"] = analyzer.apply(result["new"], payload.get("rating", "sfw"))
        return 200, result
    if method == "POST" and route == "apply":
        items = {k: [str(v) for v in vals] for k, vals in (payload.get("items") or {}).items() if vals}
        report = vision.ImageAnalyzer(maker).apply(items, payload.get("rating", "sfw"))
        return 200, report
    if method == "POST" and route == "unload":
        vision.unload_backends()
        return 200, {"ok": True}
    return 404, {"error": "not found"}
