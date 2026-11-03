"""Image analysis -> element txt files (optional, needs a vision-language model).

Pipeline::

    image -> VLM (default: Qwen3-VL-4B-Instruct) -> JSON {element_key: [phrases]}
    -> parse -> safety + duplicate check -> append new lines to elements/<key>.txt

The engine itself stays LLM-free; this module is only used when the user asks
for it (HTML page, ComfyUI node or ``python -m promptmaker.vision``).

Backends
--------
``transformers``  local model folder (or Hugging Face id) loaded with
                  ``transformers`` + ``torch``. Default.
``openai``        any OpenAI-compatible server (Ollama, LM Studio,
                  llama.cpp ``llama-server`` with a GGUF + mmproj, vLLM ...).

Settings live in ``config/vision.json`` next to the package (or the path in
``PROMPTMAKER_VISION_CONFIG``) so the model can be changed without code edits.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import gc
import io
import json
import os
import re
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .data_loader import PACKAGE_DATA_DIR, RATINGS, load_elements
from .safety import is_safe_item

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.environ.get("PROMPTMAKER_VISION_CONFIG") or os.path.join(REPO_ROOT, "config", "vision.json")
PROMPT_PATH = os.path.join(PACKAGE_DATA_DIR, "vision", "analyze_prompt.txt")

# Elements that should not be guessed from a picture.
DEFAULT_SKIP_KEYS = ["casting", "age", "outfit_type", "nationality", "camera_movement", "pacing"]
APPEND_HEADER = "# 이미지 분석 자동 추가 ({date})"
FULL_PROMPT_KEY = "full_prompt"
DEFAULT_FULL_PROMPT_FILE = os.path.join(REPO_ROOT, "output", "full_prompts.txt")
FULL_PROMPT_RULE = (
    f'- Also add the key "{FULL_PROMPT_KEY}": ONE English paragraph (3 to 6 sentences) that could be used as a '
    "text-to-image prompt to recreate this image: subject, appearance, attire, pose, setting, lighting, camera "
    "and mood, in natural language. Describe every person as an adult. Do not use placeholders in it."
)


@dataclass
class VisionConfig:
    backend: str = "transformers"          # "transformers" | "openai"
    model_path: str = ""                   # local folder or single .safetensors file; empty -> model_id
    model_id: str = DEFAULT_MODEL_ID       # HF id / folder (also config source for a single file) or server model name
    api_base: str = "http://127.0.0.1:11434/v1"  # for backend "openai"
    api_key: str = ""
    device: str = "auto"                   # "auto" | "cpu" | "cuda"
    dtype: str = "auto"                    # "auto" | "bfloat16" | "float16" | "float32"
    max_new_tokens: int = 1024
    max_items_per_key: int = 3
    max_image_side: int = 1024
    skip_keys: List[str] = field(default_factory=lambda: list(DEFAULT_SKIP_KEYS))
    full_prompt: bool = False              # also ask for one full English image prompt per image
    full_prompt_file: str = ""             # where saved full prompts go; empty -> output/full_prompts.txt

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionConfig":
        cfg = cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})
        if isinstance(cfg.skip_keys, str):
            cfg.skip_keys = [s.strip() for s in cfg.skip_keys.split(",") if s.strip()]
        return cfg

    def to_dict(self, hide_key: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if hide_key and d["api_key"]:
            d["api_key"] = "***"
        return d

    @property
    def model_ref(self) -> str:
        return self.model_path.strip() or self.model_id

    @property
    def full_prompt_path(self) -> str:
        return self.full_prompt_file.strip() or DEFAULT_FULL_PROMPT_FILE


def load_config(path: Optional[str] = None) -> VisionConfig:
    path = path or CONFIG_PATH
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return VisionConfig.from_dict(json.load(f))
    return VisionConfig()


def save_config(cfg: VisionConfig, path: Optional[str] = None) -> str:
    path = path or CONFIG_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    return path


_runtime: Dict[str, Any] = {}


def runtime_info() -> Dict[str, Any]:
    """Python / torch / GPU status of this process, shown next to the device option."""
    if not _runtime:
        info: Dict[str, Any] = {"python": sys.executable, "torch": None, "cuda": False, "gpu": ""}
        try:
            import torch

            info["torch"] = torch.__version__
            info["cuda"] = torch.cuda.is_available()
            if info["cuda"]:
                info["gpu"] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        _runtime.update(info)
    return dict(_runtime)


def is_single_file(path: str) -> bool:
    return path.lower().endswith(".safetensors") and os.path.isfile(path)


def load_single_file(path: str, dtype, on_tensor: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    """Read a single .safetensors file into a transformers state dict.

    ComfyUI style fp8 checkpoints (``<name>.weight`` in float8 + ``weight_scale`` +
    ``comfy_quant``) are dequantised to ``dtype``. ``on_tensor(done, total)`` reports progress.
    """
    from safetensors import safe_open

    sd: Dict[str, Any] = {}
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        skip = {k for k in keys if k.endswith((".comfy_quant", ".weight_scale", ".input_scale"))}
        for i, key in enumerate(keys, 1):
            if on_tensor and (i % 50 == 0 or i == len(keys)):
                on_tensor(i, len(keys))
            if key in skip:
                continue
            t = f.get_tensor(key)
            if t.dtype.is_floating_point and t.element_size() == 1:  # float8
                scale_key = key[:-len("weight")] + "weight_scale" if key.endswith("weight") else None
                t = t.to(dtype)
                if scale_key in skip:
                    t = t * f.get_tensor(scale_key).to(dtype)
            elif t.dtype.is_floating_point:
                t = t.to(dtype)
            sd[key] = t
    return sd


def check_model_path(path: str) -> Dict[str, Any]:
    """Quick sanity check for the model location picked in the UI."""
    path = (path or "").strip()
    if not path:
        return {"ok": True, "message": f"비어 있음: 기본 모델 {DEFAULT_MODEL_ID} 을(를) 자동 다운로드합니다."}
    if not os.path.exists(path):
        return {"ok": False, "message": "경로가 존재하지 않습니다."}
    if os.path.isfile(path):
        if path.lower().endswith(".gguf"):
            return {"ok": False, "message": "GGUF 파일은 llama-server 등으로 띄운 뒤 'openai' 백엔드로 연결하세요."}
        if is_single_file(path):
            return {"ok": True, "message": "단일 가중치 파일: 설정·토크나이저는 '모델 이름'의 기본 모델에서 가져옵니다."}
        return {"ok": False, "message": "모델 폴더(config.json 이 있는 폴더)나 .safetensors 파일을 지정하세요."}
    files = set(os.listdir(path))
    if "config.json" not in files:
        return {"ok": False, "message": "폴더에 config.json 이 없습니다. 모델 폴더가 맞는지 확인하세요."}
    has_weights = any(f.endswith((".safetensors", ".bin")) for f in files)
    return {"ok": has_weights, "message": "모델 폴더 확인됨." if has_weights else "가중치(.safetensors) 파일이 없습니다."}


def browse(path: str = "") -> Dict[str, Any]:
    """List folders/files for the model path picker in the HTML page."""
    if not path:
        if os.name == "nt":
            import string
            drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
            return {"path": "", "parent": "", "dirs": drives, "files": []}
        path = "/"
    path = os.path.abspath(path)
    if os.path.isfile(path):  # a picked model file: show the folder it is in
        path = os.path.dirname(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    dirs, files = [], []
    try:
        names = sorted(os.listdir(path), key=str.lower)
    except PermissionError:
        names = []
    for name in names:
        (dirs if os.path.isdir(os.path.join(path, name)) else files).append(name)
    parent = os.path.dirname(path)
    return {"path": path, "parent": "" if parent == path else parent, "dirs": dirs, "files": files[:200],
            "check": check_model_path(path)}


# ------------------------------------------------------------------ images
def load_image(image: Any, max_side: int = 1024):
    """Accept a PIL image, a path, raw bytes or a (data URL) base64 string."""
    from PIL import Image

    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    elif isinstance(image, str) and os.path.isfile(image):
        img = Image.open(image)
    elif isinstance(image, str):
        data = image.split(",", 1)[1] if image.startswith("data:") else image
        img = Image.open(io.BytesIO(base64.b64decode(data)))
    else:
        raise TypeError("지원하지 않는 이미지 형식입니다.")
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


def _to_data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------ progress
class AnalysisCancelled(RuntimeError):
    """Raised inside an analysis when the caller asked to stop."""


@dataclass
class Progress:
    """One progress report. ``percent`` is the overall progress of the whole analysis."""

    stage: str                      # "load_model" | "prepare" | "generate" | "parse" | "done"
    message: str = ""
    percent: float = 0.0
    input_tokens: int = 0           # prompt + image tokens
    output_tokens: int = 0          # generated so far
    expected_tokens: int = 0        # estimate of the final answer length
    max_new_tokens: int = 0
    tokens_per_sec: float = 0.0
    elapsed: float = 0.0            # seconds since the analysis started
    eta: Optional[float] = None     # estimated seconds left (None = unknown)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        parts = [f"[{self.percent:5.1f}%] {self.message}"]
        if self.input_tokens:
            parts.append(f"입력 {self.input_tokens} 토큰")
        if self.stage == "generate" or self.output_tokens:
            parts.append(f"출력 {self.output_tokens}/{self.expected_tokens or self.max_new_tokens} 토큰")
        if self.tokens_per_sec:
            parts.append(f"{self.tokens_per_sec:.1f} tok/s")
        parts.append(f"경과 {_fmt_time(self.elapsed)}")
        if self.eta is not None:
            parts.append(f"남은 시간 약 {_fmt_time(self.eta)}")
        return " | ".join(parts)


def _fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 60}:{sec % 60:02d}"


# Overall percent range of each stage.
_STAGE_RANGE = {"load_model": (0, 15), "prepare": (15, 20), "generate": (20, 95), "parse": (95, 100), "done": (100, 100)}


class ProgressReporter:
    """Turns stage updates into :class:`Progress` objects for a callback, and handles cancel.

    ``callback(progress)`` is called from the analysis thread; ``cancel`` is a
    ``threading.Event`` (or anything with ``is_set()``) checked at every update.
    """

    def __init__(self, callback: Optional[Callable[[Progress], None]] = None, cancel=None,
                 max_new_tokens: int = 0):
        self.callback = callback
        self.cancel = cancel
        self.start = time.monotonic()
        self.state = Progress(stage="prepare", max_new_tokens=max_new_tokens)
        self._gen_start: Optional[float] = None
        self._last_emit = 0.0

    def check(self) -> None:
        if self.cancel is not None and self.cancel.is_set():
            raise AnalysisCancelled("분석이 취소되었습니다.")

    def update(self, stage: Optional[str] = None, message: Optional[str] = None,
               fraction: Optional[float] = None, force: bool = True, **fields) -> None:
        """``fraction`` (0..1) is the progress inside the current stage."""
        self.check()
        s = self.state
        if stage:
            s.stage = stage
        if message is not None:
            s.message = message
        for k, v in fields.items():
            setattr(s, k, v)
        lo, hi = _STAGE_RANGE.get(s.stage, (0, 100))
        s.percent = round(lo + (hi - lo) * min(max(fraction or 0.0, 0.0), 1.0), 1)
        s.elapsed = round(time.monotonic() - self.start, 1)
        now = time.monotonic()
        if self.callback and (force or now - self._last_emit >= 0.25):
            self._last_emit = now
            self.callback(Progress(**asdict(s)))

    # --- generation helpers used by the backends
    def generation_started(self, input_tokens: int, expected_tokens: int) -> None:
        self._gen_start = time.monotonic()
        self.update("generate", f"답변 생성 중 (입력 {input_tokens} 토큰 처리)", 0.0,
                    input_tokens=input_tokens, expected_tokens=expected_tokens, output_tokens=0)

    def token(self, count: int) -> None:
        s = self.state
        out = s.output_tokens + count
        elapsed = time.monotonic() - (self._gen_start or self.start)
        rate = out / elapsed if elapsed > 0 else 0.0
        expected = max(s.expected_tokens, out + 1) if out < s.max_new_tokens else s.max_new_tokens
        eta = (expected - out) / rate if rate > 0 else None
        self.update(None, "답변 생성 중", out / expected if expected else 0.0, force=False,
                    output_tokens=out, expected_tokens=expected, tokens_per_sec=round(rate, 2),
                    eta=None if eta is None else round(eta, 1))


def _hf_kwargs(ref: str) -> Dict[str, Any]:
    """Use the local Hugging Face cache without contacting the Hub when the model is already there."""
    if os.path.isdir(ref):
        return {}
    try:
        from huggingface_hub import try_to_load_from_cache

        if isinstance(try_to_load_from_cache(ref, "config.json"), str):
            return {"local_files_only": True}
    except Exception:
        pass
    return {}


def _accepts(fn, name: str) -> bool:
    import inspect

    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class _TokenStreamer:
    """transformers streamer that counts generated tokens (the first ``put`` is the prompt)."""

    def __init__(self, reporter: ProgressReporter):
        self.reporter = reporter
        self.prompt_seen = False

    def put(self, value) -> None:
        if not self.prompt_seen:
            self.prompt_seen = True
            return
        self.reporter.token(int(value.numel()) if hasattr(value, "numel") else 1)

    def end(self) -> None:
        pass


# ------------------------------------------------------------------ backends
class TransformersBackend:
    def __init__(self, cfg: VisionConfig, progress: Optional[ProgressReporter] = None):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"이 서버의 파이썬({sys.executable}, torch {torch.__version__})은 GPU(CUDA)를 쓸 수 없습니다. "
                "장치를 '자동' 또는 'CPU'로 바꾸거나, CUDA 용 torch 가 설치된 파이썬으로 서버를 실행하세요 "
                "(예: set PROMPTMAKER_PYTHON=ComfyUI\\python_embeded\\python.exe 후 run_server.bat).")
        dtype = "auto" if cfg.dtype == "auto" else getattr(torch, cfg.dtype)
        if is_single_file(cfg.model_ref):
            # One .safetensors file (e.g. a fine-tune saved for ComfyUI): weights from the file,
            # config / tokenizer / image processor from the base model (model_id).
            from transformers import AutoConfig

            base = cfg.model_id
            config = AutoConfig.from_pretrained(base, **_hf_kwargs(base))
            load_dtype = torch.bfloat16 if dtype == "auto" else dtype
            try:  # skip random init (minutes on CPU); every weight comes from the file
                from transformers.initialization import no_init_weights
            except ImportError:
                from contextlib import nullcontext as no_init_weights
            with no_init_weights():
                self.model = AutoModelForImageTextToText.from_config(config, dtype=load_dtype)

            def on_tensor(done: int, total: int) -> None:
                if progress:
                    progress.update("load_model", f"가중치 파일 읽는 중 ({done}/{total})", 0.1 + 0.8 * done / total,
                                    force=done == total)

            state = load_single_file(cfg.model_ref, load_dtype, on_tensor)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            del state
            missing = [k for k in missing if k != "lm_head.weight"]  # tied to embed_tokens
            if missing or unexpected:
                raise ValueError(f"가중치 파일이 기본 모델({base})과 맞지 않습니다. "
                                 f"누락 {len(missing)}개 (예: {missing[:2]}), 불필요 {len(unexpected)}개 (예: {unexpected[:2]})")
            self.model.tie_weights()
            self.model.eval()
            device = cfg.device if cfg.device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
            if progress:
                progress.update("load_model", f"모델을 {device.upper()} 로 옮기는 중", 0.95)
            self.model.to(device)
        else:
            base = cfg.model_ref
            if progress:
                progress.update("load_model", f"모델 불러오는 중: {base}", 0.1)
            kwargs: Dict[str, Any] = {"dtype": dtype}
            if cfg.device == "auto":
                kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
            self.model = AutoModelForImageTextToText.from_pretrained(base, **kwargs, **_hf_kwargs(base))
            if cfg.device in ("cpu", "cuda"):
                self.model.to(cfg.device)
        self.processor = AutoProcessor.from_pretrained(base, **_hf_kwargs(base))
        self.max_new_tokens = cfg.max_new_tokens
        self.history: List[int] = []  # output lengths of earlier runs, for the time estimate
        self.usage: Dict[str, int] = {}

    def chat(self, image, prompt: str, progress: Optional[ProgressReporter] = None,
             expected_tokens: int = 0) -> str:
        import torch

        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        kwargs: Dict[str, Any] = {}
        if progress:
            if self.history:
                expected_tokens = int(sum(self.history[-5:]) / len(self.history[-5:]) * 1.1)
            progress.generation_started(n_in, min(expected_tokens or self.max_new_tokens, self.max_new_tokens))
            kwargs["streamer"] = _TokenStreamer(progress)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False, **kwargs)
        out = out[:, n_in:]
        self.history.append(int(out.shape[1]))
        self.usage = {"input_tokens": n_in, "output_tokens": int(out.shape[1])}
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]


class OpenAIBackend:
    def __init__(self, cfg: VisionConfig, progress: Optional[ProgressReporter] = None):
        self.cfg = cfg
        self.history: List[int] = []
        self.usage: Dict[str, int] = {}

    def chat(self, image, prompt: str, progress: Optional[ProgressReporter] = None,
             expected_tokens: int = 0) -> str:
        body = {
            "model": self.cfg.model_id,
            "temperature": 0,
            "max_tokens": self.cfg.max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
                {"type": "text", "text": prompt},
            ]}],
        }
        req = urllib.request.Request(self.cfg.api_base.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.cfg.api_key or 'none'}"})
        if progress:
            if self.history:
                expected_tokens = int(sum(self.history[-5:]) / len(self.history[-5:]) * 1.1)
            progress.generation_started(0, min(expected_tokens or self.cfg.max_new_tokens, self.cfg.max_new_tokens))
        text, usage, chunks = [], {}, 0
        with urllib.request.urlopen(req, timeout=600) as res:
            if "text/event-stream" not in res.headers.get("Content-Type", ""):  # server ignored "stream"
                data = json.loads(res.read().decode("utf-8"))
                usage = data.get("usage") or {}
                text.append(data["choices"][0]["message"]["content"])
            else:
                for raw in res:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    data = json.loads(line[5:])
                    usage = data.get("usage") or usage
                    for choice in data.get("choices") or []:
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece:
                            text.append(piece)
                            chunks += 1
                            if progress:
                                progress.token(1)  # one streamed chunk is usually one token
        self.usage = {"input_tokens": int(usage.get("prompt_tokens", 0)),
                      "output_tokens": int(usage.get("completion_tokens", chunks))}
        self.history.append(self.usage["output_tokens"])
        if progress and self.usage["input_tokens"]:
            progress.update(None, None, 1.0, input_tokens=self.usage["input_tokens"],
                            output_tokens=self.usage["output_tokens"])
        return "".join(text)


BACKENDS: Dict[str, Callable[[VisionConfig], Any]] = {
    "transformers": TransformersBackend,
    "openai": OpenAIBackend,
}

_backend_cache: Dict[Tuple, Any] = {}
_backend_lock = threading.Lock()


def _backend_key(cfg: VisionConfig) -> Tuple:
    return (cfg.backend, cfg.model_ref, cfg.model_id, cfg.api_base, cfg.device, cfg.dtype, cfg.max_new_tokens)


def is_loaded(cfg: VisionConfig) -> bool:
    return _backend_key(cfg) in _backend_cache


def get_backend(cfg: VisionConfig, progress: Optional[ProgressReporter] = None):
    key = _backend_key(cfg)
    with _backend_lock:
        if key not in _backend_cache:
            if cfg.backend not in BACKENDS:
                raise ValueError(f"알 수 없는 백엔드: {cfg.backend} (사용 가능: {', '.join(BACKENDS)})")
            if cfg.backend == "transformers":
                _backend_cache.clear()  # keep only one local model in memory
                gc.collect()
            if progress:
                progress.update("load_model", f"모델 준비 중 ({cfg.backend})", 0.0)
            _backend_cache[key] = BACKENDS[cfg.backend](cfg, progress)
            if progress:
                progress.update("load_model", "모델 준비 완료", 1.0)
        return _backend_cache[key]


def unload_backends() -> None:
    with _backend_lock:
        _backend_cache.clear()


# ------------------------------------------------------------------ prompt
def element_files(data_dir: str) -> Dict[str, Dict[str, str]]:
    """``{key: {rating: path}}`` for every txt element file in ``data_dir``."""
    out: Dict[str, Dict[str, str]] = {}
    edir = os.path.join(data_dir, "elements")
    if not os.path.isdir(edir):
        return out
    for fname in sorted(os.listdir(edir)):
        stem, ext = os.path.splitext(fname)
        if ext.lower() != ".txt":
            continue
        parts = stem.split(".")
        rating = parts[1] if len(parts) > 1 and parts[1] in RATINGS else "sfw"
        out.setdefault(parts[0], {})[rating] = os.path.join(edir, fname)
    return out


def _file_title(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip()
            if line:
                break
    return ""


def build_prompt(data_dirs: List[str], cfg: VisionConfig) -> Tuple[str, List[str]]:
    """Prompt listing every analysable element with a few existing examples."""
    elements = load_elements(data_dirs)
    files: Dict[str, str] = {}
    for d in data_dirs:
        for key, by_rating in element_files(d).items():
            if "sfw" in by_rating:
                files.setdefault(key, by_rating["sfw"])
    lines, keys = [], []
    for key, el in elements.items():
        if key in cfg.skip_keys:
            continue
        path = files.get(key)
        if not path:  # json-only elements (e.g. casting) are not written to
            continue
        sfw = [i.text for i in el.items if i.rating == "sfw"][:3]
        title = (_file_title(path) or el.description or key).rstrip(". ")
        examples = "; ".join(f'"{t}"' for t in sfw)
        lines.append(f"- {key}: {title}. e.g. {examples}")
        keys.append(key)
    with open(PROMPT_PATH, encoding="utf-8") as f:
        template = f.read()
    prompt = template.replace("{elements}", "\n".join(lines)).replace("{max_items}", str(cfg.max_items_per_key))
    extra = FULL_PROMPT_RULE if cfg.full_prompt else ""
    if "{extra_rules}" in prompt:
        prompt = prompt.replace("{extra_rules}", extra)
    elif extra:
        prompt += "\n" + extra + "\n"
    return prompt, keys


# ------------------------------------------------------------------ parsing
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# Placeholder answers small models give instead of leaving a key out.
_EMPTY_RE = re.compile(r"^(?:none|n/?a|null|unknown|nothing|no (?:person|people|one|man|woman|human)s?"
                       r"|(?:no|not)\b.*\b(?:visible|applicable|present|shown|clear|available|detected|seen|identifiable))$",
                       re.IGNORECASE)


def extract_json(text: str) -> Dict[str, Any]:
    """The JSON object inside a model reply (code fences and trailing commas tolerated)."""
    m = _FENCE_RE.search(text)
    body = m.group(1) if m else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("모델 응답에서 JSON 을 찾지 못했습니다.")
    raw = body[start:end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(re.sub(r",\s*([}\]])", r"\1", raw))


def split_full_prompt(data: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Separate the optional full prompt from the element lists."""
    data = dict(data)
    value = data.pop(FULL_PROMPT_KEY, "")
    if isinstance(value, list):
        value = " ".join(v for v in value if isinstance(v, str))
    text = " ".join(str(value or "").split())
    return data, "" if _EMPTY_RE.match(text) else text


def parse_response(text: Any) -> Dict[str, List[str]]:
    """Pull the JSON object out of a model reply (or take a parsed dict) and normalise it to lists of strings."""
    data = text if isinstance(text, dict) else extract_json(text)
    out: Dict[str, List[str]] = {}
    for key, value in data.items():
        if key == FULL_PROMPT_KEY:
            continue
        values = value if isinstance(value, list) else [value]
        clean = []
        for v in values:
            if not isinstance(v, str):
                continue
            v = " ".join(v.replace("\n", " ").split()).strip(" .,;")
            if v and not _EMPTY_RE.match(v) and v not in clean:
                clean.append(v)
        if clean:
            out[str(key).strip()] = clean
    return out


# ------------------------------------------------------------------ duplicates
_WEIGHT_RE = re.compile(r"^\s*\d+(?:\.\d+)?::")
_TAG_RE = re.compile(r"(?:\s+[#+!][a-z0-9_]+)+\s*$")


def normalize(text: str) -> str:
    """Comparison key: no weight prefix, no trailing tags, no braces, lowercase, single spaces."""
    text = _WEIGHT_RE.sub("", text)
    text = _TAG_RE.sub("", text)
    text = text.replace("{", "").replace("}", "")
    text = re.sub(r"[\s,.;]+", " ", text.lower())
    return re.sub(r"^(?:a|an|the) ", "", text.strip())


def existing_keys(data_dirs: List[str], key: str) -> set:
    seen = set()
    for d in data_dirs:
        for path in element_files(d).get(key, {}).values():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        seen.add(normalize(line))
    return seen


def classify(data_dirs: List[str], found: Dict[str, List[str]], allowed_keys: List[str]) -> Dict[str, List[Dict[str, str]]]:
    """Mark every candidate as new / duplicate / unsafe / unknown_key."""
    result: Dict[str, List[Dict[str, str]]] = {}
    for key, values in found.items():
        seen = existing_keys(data_dirs, key) if key in allowed_keys else set()
        rows = []
        for v in values:
            if key not in allowed_keys:
                status = "unknown_key"
            elif not is_safe_item(v):
                status = "unsafe"
            elif normalize(v) in seen:
                status = "duplicate"
            else:
                status = "new"
                seen.add(normalize(v))
            rows.append({"text": v, "status": status})
        result[key] = rows
    return result


_write_lock = threading.Lock()


def append_items(data_dir: str, items: Dict[str, List[str]], rating: str = "sfw",
                 search_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
    """Append new lines to ``<data_dir>/elements/<key>[.<rating>].txt``.

    Duplicates (checked across ``search_dirs`` and every rating file of the key),
    unsafe items and keys without a txt file are skipped. Files are only appended to.
    """
    if rating not in RATINGS:
        raise ValueError(f"rating 은 {RATINGS} 중 하나여야 합니다.")
    search_dirs = search_dirs or [data_dir]
    if data_dir not in search_dirs:
        search_dirs = search_dirs + [data_dir]
    added: Dict[str, List[str]] = {}
    skipped: Dict[str, List[Dict[str, str]]] = {}
    known = set()
    for d in search_dirs:
        known.update(element_files(d))
    header = APPEND_HEADER.format(date=datetime.date.today().isoformat())
    with _write_lock:
        for key, values in items.items():
            if key not in known or not re.fullmatch(r"[a-z_][a-z0-9_]*", key):
                skipped[key] = [{"text": v, "reason": "unknown_key"} for v in values]
                continue
            seen = existing_keys(search_dirs, key)
            new = []
            for v in values:
                v = " ".join(str(v).split()).strip()
                reason = None
                if not v:
                    continue
                if not is_safe_item(v):
                    reason = "unsafe"
                elif normalize(v) in seen:
                    reason = "duplicate"
                if reason:
                    skipped.setdefault(key, []).append({"text": v, "reason": reason})
                    continue
                seen.add(normalize(v))
                new.append(v)
            if not new:
                continue
            fname = f"{key}.txt" if rating == "sfw" else f"{key}.{rating}.txt"
            path = os.path.join(data_dir, "elements", fname)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            content = ""
            if os.path.isfile(path):
                with open(path, encoding="utf-8", newline="") as f:
                    content = f.read()
            eol = "\r\n" if "\r\n" in content else "\n"  # keep the file's line endings
            chunk = "" if content.endswith("\n") or not content else eol
            if header not in content:
                chunk += header + eol
            chunk += eol.join(new) + eol
            with open(path, "a", encoding="utf-8", newline="") as f:
                f.write(chunk)
            added[key] = new
    return {"added": added, "skipped": skipped,
            "added_count": sum(len(v) for v in added.values())}


# ------------------------------------------------------------------ high level
class ImageAnalyzer:
    """``ImageAnalyzer(maker).analyze(img)`` then ``.apply(result["new"])``."""

    def __init__(self, maker=None, config: Optional[VisionConfig] = None,
                 data_dirs: Optional[List[str]] = None, write_dir: Optional[str] = None,
                 backend=None):
        self.maker = maker
        self.config = config or load_config()
        self.data_dirs = data_dirs or (maker.data_dirs if maker else [PACKAGE_DATA_DIR])
        # Write to the first (package / base) data dir so items land in the original txt files.
        self.write_dir = write_dir or self.data_dirs[0]
        self._backend = backend

    @property
    def backend(self):
        return self._backend or get_backend(self.config)

    def analyze(self, image: Any, progress: Optional[Callable[[Progress], None]] = None,
                cancel=None) -> Dict[str, Any]:
        """Analyse one image.

        ``progress(Progress)`` receives stage, token counts, speed, elapsed time and the
        estimated time left; ``cancel`` (e.g. ``threading.Event``) stops the analysis with
        :class:`AnalysisCancelled` when set.
        """
        rep = ProgressReporter(progress, cancel, max_new_tokens=int(self.config.max_new_tokens))
        rep.update("prepare", "이미지와 지시문 준비 중", 0.0)
        img = load_image(image, self.config.max_image_side)
        prompt, keys = build_prompt(self.data_dirs, self.config)
        if self._backend is not None:
            backend = self._backend
        else:
            if not is_loaded(self.config):
                rep.update("load_model", "모델 불러오는 중 (처음 한 번만)", 0.0)
            backend = get_backend(self.config, rep)
        rep.update("prepare", f"요소 {len(keys)}개로 지시문 구성 완료", 1.0)
        # First-run estimate: roughly a short phrase for half of the keys (+ the paragraph).
        expected = min(int(self.config.max_new_tokens),
                       20 + 12 * len(keys) + (160 if self.config.full_prompt else 0))
        if _accepts(backend.chat, "progress"):
            raw = backend.chat(img, prompt, progress=rep, expected_tokens=expected)
        else:
            raw = backend.chat(img, prompt)
        rep.update("parse", "응답 해석 및 중복 확인 중", 0.0)
        data, full_prompt = split_full_prompt(extract_json(raw))
        found = parse_response(data)
        limit = max(1, int(self.config.max_items_per_key))
        found = {k: v[:limit] for k, v in found.items() if k not in self.config.skip_keys}
        rows = classify(self.data_dirs, found, keys)
        new = {k: [r["text"] for r in v if r["status"] == "new"] for k, v in rows.items()}
        usage = dict(getattr(backend, "usage", {}) or {})
        n_new = sum(len(v) for v in new.values())
        rep.update("done", f"완료: 새 항목 {n_new}개", 1.0, eta=0.0, **{k: v for k, v in usage.items()
                                                                       if k in ("input_tokens", "output_tokens")})
        stats = rep.state.to_dict()
        return {"elements": rows, "new": {k: v for k, v in new.items() if v}, "raw": raw, "stats": stats,
                "full_prompt": full_prompt if self.config.full_prompt else ""}

    def apply(self, items: Dict[str, List[str]], rating: str = "sfw") -> Dict[str, Any]:
        report = append_items(self.write_dir, items, rating=rating, search_dirs=self.data_dirs)
        if self.maker is not None and report["added_count"]:
            self.maker.reload()
        return report

    def save_full_prompts(self, prompts: List[str], path: Optional[str] = None) -> Dict[str, Any]:
        """Append full prompts to ``path`` (default: the config's ``full_prompt_path``)."""
        return save_full_prompts(prompts, path or self.config.full_prompt_path)

    def analyze_and_apply(self, image: Any, rating: str = "sfw",
                          progress: Optional[Callable[[Progress], None]] = None, cancel=None,
                          save_full_prompt: bool = False) -> Dict[str, Any]:
        result = self.analyze(image, progress, cancel)
        result["report"] = self.apply(result["new"], rating)
        if save_full_prompt and result["full_prompt"]:
            result["full_prompt_report"] = self.save_full_prompts([result["full_prompt"]])
        return result


def save_full_prompts(prompts: List[str], path: str = DEFAULT_FULL_PROMPT_FILE) -> Dict[str, Any]:
    """Append one prompt per line to a text file, skipping duplicates and unsafe text.

    The file uses the element file conventions (``#`` comments, one item per line), so it
    can be read back with any text tool or used as a wildcard list.
    """
    saved, skipped = [], []
    with _write_lock:
        content = ""
        if os.path.isfile(path):
            with open(path, encoding="utf-8", newline="") as f:
                content = f.read()
        seen = {normalize(line) for line in content.splitlines() if line.strip() and not line.startswith("#")}
        for p in prompts:
            p = " ".join(str(p).split())
            if not p:
                continue
            if not is_safe_item(p):
                skipped.append({"text": p, "reason": "unsafe"})
            elif normalize(p) in seen:
                skipped.append({"text": p, "reason": "duplicate"})
            else:
                seen.add(normalize(p))
                saved.append(p)
        if saved:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            eol = "\r\n" if "\r\n" in content else "\n"
            chunk = "" if content.endswith("\n") or not content else eol
            if not content:
                chunk += "# 이미지 분석 풀 프롬프트 (한 줄에 하나)" + eol
            chunk += eol.join(saved) + eol
            with open(path, "a", encoding="utf-8", newline="") as f:
                f.write(chunk)
    return {"path": path, "saved": saved, "skipped": skipped, "saved_count": len(saved)}


# ------------------------------------------------------------------ queue
@dataclass
class QueueItem:
    id: str
    name: str
    status: str = "pending"        # pending | running | done | error | cancelled
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: str = ""
    report: Optional[Dict[str, Any]] = None              # element txt report (auto apply)
    full_prompt_report: Optional[Dict[str, Any]] = None  # full prompt save report (auto save)
    image: Any = field(default=None, repr=False)

    def to_dict(self, with_result: bool = True) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k != "image"}
        if not with_result:
            d.pop("result")
        return d


class AnalysisQueue:
    """Analyse many images one after another.

    ``add()`` may be called while ``run()`` is working (from another thread); ``run()``
    keeps going until no pending item is left. With ``auto_apply`` new element items are
    written right away, with ``auto_save_full_prompt`` the full prompts are saved.

        q = AnalysisQueue(ImageAnalyzer(pm), auto_apply=True)
        q.add_many(["a.jpg", "b.jpg"])
        q.run(progress=lambda item, p: print(item.name, p))
    """

    def __init__(self, analyzer: ImageAnalyzer, auto_apply: bool = False, rating: str = "sfw",
                 auto_save_full_prompt: bool = False):
        self.analyzer = analyzer
        self.auto_apply = auto_apply
        self.rating = rating
        self.auto_save_full_prompt = auto_save_full_prompt
        self.items: List[QueueItem] = []
        self.cancel = threading.Event()
        self._lock = threading.Lock()
        self._counter = 0
        self.running = False

    def add(self, image: Any, name: Optional[str] = None) -> QueueItem:
        with self._lock:
            self._counter += 1
            if name is None:
                name = os.path.basename(image) if isinstance(image, str) and os.path.isfile(image) else f"image {self._counter}"
            item = QueueItem(id=f"{self._counter:04d}", name=name, image=image)
            self.items.append(item)
            return item

    def add_many(self, images: List[Any]) -> List[QueueItem]:
        return [self.add(i) for i in images]

    def add_folder(self, folder: str, exts=(".png", ".jpg", ".jpeg", ".webp", ".bmp")) -> List[QueueItem]:
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
        return [self.add(os.path.join(folder, f)) for f in files]

    def get(self, item_id: str) -> Optional[QueueItem]:
        return next((i for i in self.items if i.id == item_id), None)

    def remove(self, item_id: str) -> bool:
        with self._lock:
            item = self.get(item_id)
            if item is None or item.status == "running":
                return False
            self.items.remove(item)
            return True

    def clear_finished(self) -> None:
        with self._lock:
            self.items = [i for i in self.items if i.status in ("pending", "running")]

    def stop(self) -> None:
        """Cancel the running image and every pending one."""
        self.cancel.set()

    def _next(self) -> Optional[QueueItem]:
        with self._lock:
            return next((i for i in self.items if i.status == "pending"), None)

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        for i in self.items:
            counts[i.status] = counts.get(i.status, 0) + 1
        total = len(self.items)
        finished = sum(counts.get(s, 0) for s in ("done", "error", "cancelled"))
        running = next((i for i in self.items if i.status == "running"), None)
        frac = (running.progress or {}).get("percent", 0) / 100 if running else 0
        return {"total": total, "counts": counts, "running": self.running,
                "percent": round(100 * (finished + frac) / total, 1) if total else 0.0}

    def run(self, progress: Optional[Callable[[QueueItem, Progress], None]] = None) -> List[QueueItem]:
        """Process pending items in order. ``progress(item, Progress)`` is called for each update."""
        self.running = True
        self.cancel.clear()
        try:
            while True:
                item = self._next()
                if item is None:
                    break
                if self.cancel.is_set():
                    for i in self.items:
                        if i.status == "pending":
                            i.status = "cancelled"
                    break
                item.status = "running"

                def on_progress(p: Progress, item=item) -> None:
                    item.progress = p.to_dict()
                    if progress:
                        progress(item, p)

                try:
                    result = self.analyzer.analyze(item.image, progress=on_progress, cancel=self.cancel)
                    item.result = result
                    if self.auto_apply and result["new"]:
                        item.report = self.analyzer.apply(result["new"], self.rating)
                    if self.auto_save_full_prompt and result.get("full_prompt"):
                        item.full_prompt_report = self.analyzer.save_full_prompts([result["full_prompt"]])
                    item.status = "done"
                except AnalysisCancelled:
                    item.status = "cancelled"
                except Exception as exc:  # keep going with the next image
                    item.error, item.status = f"{type(exc).__name__}: {exc}", "error"
                item.image = None  # free memory (data URLs can be large)
        finally:
            self.running = False
        return self.items


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="이미지를 분석해 요소 txt 에 새 항목을 추가합니다.")
    parser.add_argument("images", nargs="+", help="이미지 파일 또는 폴더 경로 (여러 개면 차례로 분석)")
    parser.add_argument("--apply", action="store_true", help="새 항목을 txt 에 실제로 추가 (없으면 미리보기)")
    parser.add_argument("--full-prompt", action="store_true", help="이미지마다 풀 프롬프트도 생성")
    parser.add_argument("--save-full-prompt", nargs="?", const="", default=None, metavar="FILE",
                        help="풀 프롬프트를 파일에 저장 (경로 생략 시 설정의 기본 파일)")
    parser.add_argument("--rating", default="sfw", choices=RATINGS)
    parser.add_argument("--backend", choices=list(BACKENDS))
    parser.add_argument("--model-path", help="모델 폴더 (설정 파일 값 대신 사용)")
    parser.add_argument("--model-id")
    parser.add_argument("--api-base")
    parser.add_argument("--save-config", action="store_true", help="위 모델 설정을 config/vision.json 에 저장")
    parser.add_argument("--user-data", default=None)
    args = parser.parse_args(argv)

    from .core import PromptMaker

    cfg = load_config()
    for name in ("backend", "model_path", "model_id", "api_base"):
        if getattr(args, name):
            setattr(cfg, name, getattr(args, name))
    if args.full_prompt or args.save_full_prompt is not None:
        cfg.full_prompt = True
    if args.save_full_prompt:
        cfg.full_prompt_file = args.save_full_prompt
    if args.save_config:
        print("설정 저장:", save_config(cfg))
    analyzer = ImageAnalyzer(PromptMaker(user_data_dir=args.user_data), cfg)
    queue = AnalysisQueue(analyzer, auto_apply=args.apply, rating=args.rating,
                          auto_save_full_prompt=args.save_full_prompt is not None)
    for path in args.images:
        queue.add_folder(path) if os.path.isdir(path) else queue.add(path)
    total = len(queue.items)

    def show(item: QueueItem, p: Progress) -> None:
        print("\r" + f"[{item.id}/{total:04d} {item.name}] {p}"[:160].ljust(160), end="", flush=True)

    for item in queue.run(progress=show):
        print(f"\n== {item.name}: {item.status} {item.error}")
        if not item.result:
            continue
        for key, rows in item.result["elements"].items():
            for r in rows:
                print(f"  [{r['status']:<11}] {key}: {r['text']}")
        if item.result.get("full_prompt"):
            print(f"  풀 프롬프트: {item.result['full_prompt']}")
        if item.report:
            print(f"  -> {item.report['added_count']}개 추가")
        if item.full_prompt_report:
            print(f"  -> 풀 프롬프트 {item.full_prompt_report['saved_count']}개 저장: {item.full_prompt_report['path']}")


if __name__ == "__main__":
    main()
