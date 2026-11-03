"""HTTP routes for image analysis used by ``promptmaker.server`` (``/api/vision/*``)."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Tuple
from urllib.parse import parse_qs, urlparse

from . import vision

# Background analysis jobs, polled by the page for progress.
_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()
_run_lock = threading.Lock()  # one analysis at a time (one model in memory)


def _start_job(maker, payload: Dict[str, Any]) -> Dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    cancel = threading.Event()
    job = {"id": job_id, "status": "queued", "created": time.time(), "cancel": cancel,
           "progress": vision.Progress(stage="prepare", message="대기 중 (앞선 분석이 끝나면 시작)").to_dict(),
           "result": None, "error": None}
    with _jobs_lock:
        # keep the last few finished jobs only
        for old in sorted(_jobs, key=lambda k: _jobs[k]["created"])[:-20]:
            if _jobs[old]["status"] in ("done", "error", "cancelled"):
                del _jobs[old]
        _jobs[job_id] = job

    def on_progress(p: vision.Progress) -> None:
        job["progress"] = p.to_dict()

    def run() -> None:
        with _run_lock:
            if cancel.is_set():
                job["status"] = "cancelled"
                return
            job["status"] = "running"
            try:
                analyzer = vision.ImageAnalyzer(maker)
                result = analyzer.analyze(payload["image"], progress=on_progress, cancel=cancel)
                if payload.get("apply"):
                    result["report"] = analyzer.apply(result["new"], payload.get("rating", "sfw"))
                job["result"], job["status"] = result, "done"
            except vision.AnalysisCancelled:
                job["status"] = "cancelled"
            except Exception as exc:  # shown in the page
                job["error"], job["status"] = f"{type(exc).__name__}: {exc}", "error"

    threading.Thread(target=run, daemon=True, name=f"vision-{job_id}").start()
    return _public(job)


def _public(job: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in job.items() if k != "cancel"}


# One shared image queue per server: images dropped on the page are analysed in order.
_queue: Dict[str, Any] = {"q": None, "thread": None}


def _get_queue(maker) -> vision.AnalysisQueue:
    if _queue["q"] is None:
        _queue["q"] = vision.AnalysisQueue(vision.ImageAnalyzer(maker))
    return _queue["q"]


def _queue_state(q: vision.AnalysisQueue) -> Dict[str, Any]:
    return {"summary": q.summary(), "items": [i.to_dict(with_result=False) for i in q.items],
            "options": {"auto_apply": q.auto_apply, "rating": q.rating,
                        "auto_save_full_prompt": q.auto_save_full_prompt}}


def _start_worker(maker, q: vision.AnalysisQueue) -> None:
    t = _queue["thread"]
    if t is not None and t.is_alive():
        return  # the running loop picks up new items by itself

    def work() -> None:
        with _run_lock:
            q.analyzer = vision.ImageAnalyzer(maker)  # pick up the latest model settings
            q.run()

    _queue["thread"] = threading.Thread(target=work, daemon=True, name="vision-queue")
    _queue["thread"].start()


def _handle_queue(maker, method: str, parts, payload: Dict[str, Any]) -> Tuple[int, Any]:
    q = _get_queue(maker)
    for opt in ("auto_apply", "auto_save_full_prompt"):
        if opt in payload:
            setattr(q, opt, bool(payload[opt]))
    if payload.get("rating"):
        q.rating = payload["rating"]
    if method == "GET" and not parts:
        return 200, _queue_state(q)
    if method == "POST" and not parts:
        images = payload.get("images") or []
        if not images:  # options only
            return 200, _queue_state(q)
        for img in images:
            if isinstance(img, dict):
                q.add(img["image"], img.get("name"))
            else:
                q.add(img)
        _start_worker(maker, q)
        return 200, _queue_state(q)
    if method == "POST" and parts == ["stop"]:
        q.stop()
        return 200, _queue_state(q)
    if method == "POST" and parts == ["clear"]:
        q.clear_finished()
        return 200, _queue_state(q)
    item = q.get(parts[0]) if parts else None
    if item is None:
        return 404, {"error": "대기열 항목을 찾을 수 없습니다."}
    if method == "GET" and len(parts) == 1:
        return 200, item.to_dict()
    if method == "POST" and parts[1:] == ["remove"]:
        return (200, _queue_state(q)) if q.remove(item.id) else (400, {"error": "분석 중인 항목은 뺄 수 없습니다."})
    return 404, {"error": "not found"}


def handle(maker, method: str, path: str, payload: Dict[str, Any]) -> Tuple[int, Any]:
    url = urlparse(path)
    route = url.path[len("/api/vision"):].strip("/")
    query = {k: v[-1] for k, v in parse_qs(url.query).items()}

    if route == "queue" or route.startswith("queue/"):
        return _handle_queue(maker, method, [p for p in route.split("/")[1:] if p], payload)
    if route == "full_prompts" and method == "POST":
        prompts = [str(p) for p in payload.get("prompts") or [] if str(p).strip()]
        return 200, vision.save_full_prompts(prompts, vision.load_config().full_prompt_path)
    if route == "jobs" and method == "POST":
        if not payload.get("image"):
            return 400, {"error": "이미지가 없습니다."}
        return 200, _start_job(maker, payload)
    if route.startswith("jobs/"):
        parts = route.split("/")
        job = _jobs.get(parts[1])
        if job is None:
            return 404, {"error": "작업을 찾을 수 없습니다."}
        if method == "POST" and parts[2:] == ["cancel"]:
            job["cancel"].set()
            if job["status"] == "queued":  # never started: done right away
                job["status"] = "cancelled"
            return 200, {"ok": True}
        if method == "GET":
            return 200, _public(job)

    if method == "GET" and route == "config":
        cfg = vision.load_config()
        return 200, {"config": cfg.to_dict(hide_key=True), "config_path": vision.CONFIG_PATH,
                     "backends": list(vision.BACKENDS), "default_model_id": vision.DEFAULT_MODEL_ID,
                     "check": vision.check_model_path(cfg.model_path), "runtime": vision.runtime_info(),
                     "full_prompt_path": cfg.full_prompt_path}
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
