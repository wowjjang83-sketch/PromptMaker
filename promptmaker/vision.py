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
import io
import json
import os
import re
import threading
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


@dataclass
class VisionConfig:
    backend: str = "transformers"          # "transformers" | "openai"
    model_path: str = ""                   # local folder; empty -> model_id
    model_id: str = DEFAULT_MODEL_ID       # Hugging Face id or server model name
    api_base: str = "http://127.0.0.1:11434/v1"  # for backend "openai"
    api_key: str = ""
    device: str = "auto"                   # "auto" | "cpu" | "cuda"
    dtype: str = "auto"                    # "auto" | "bfloat16" | "float16" | "float32"
    max_new_tokens: int = 1024
    max_items_per_key: int = 3
    max_image_side: int = 1024
    skip_keys: List[str] = field(default_factory=lambda: list(DEFAULT_SKIP_KEYS))

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
        return {"ok": False, "message": "모델 파일이 아니라 모델 폴더(config.json 이 있는 폴더)를 지정하세요."}
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


# ------------------------------------------------------------------ backends
class TransformersBackend:
    def __init__(self, cfg: VisionConfig):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = "auto" if cfg.dtype == "auto" else getattr(torch, cfg.dtype)
        kwargs: Dict[str, Any] = {"dtype": dtype}
        if cfg.device == "auto":
            kwargs["device_map"] = "auto" if torch.cuda.is_available() else None
        self.model = AutoModelForImageTextToText.from_pretrained(cfg.model_ref, **kwargs)
        if cfg.device in ("cpu", "cuda"):
            self.model.to(cfg.device)
        self.processor = AutoProcessor.from_pretrained(cfg.model_ref)
        self.max_new_tokens = cfg.max_new_tokens

    def chat(self, image, prompt: str) -> str:
        import torch

        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                 {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)[0]


class OpenAIBackend:
    def __init__(self, cfg: VisionConfig):
        self.cfg = cfg

    def chat(self, image, prompt: str) -> str:
        body = {
            "model": self.cfg.model_id,
            "temperature": 0,
            "max_tokens": self.cfg.max_new_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
                {"type": "text", "text": prompt},
            ]}],
        }
        req = urllib.request.Request(self.cfg.api_base.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.cfg.api_key or 'none'}"})
        with urllib.request.urlopen(req, timeout=600) as res:
            data = json.loads(res.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


BACKENDS: Dict[str, Callable[[VisionConfig], Any]] = {
    "transformers": TransformersBackend,
    "openai": OpenAIBackend,
}

_backend_cache: Dict[Tuple, Any] = {}
_backend_lock = threading.Lock()


def get_backend(cfg: VisionConfig):
    key = (cfg.backend, cfg.model_ref, cfg.api_base, cfg.device, cfg.dtype, cfg.max_new_tokens)
    with _backend_lock:
        if key not in _backend_cache:
            if cfg.backend not in BACKENDS:
                raise ValueError(f"알 수 없는 백엔드: {cfg.backend} (사용 가능: {', '.join(BACKENDS)})")
            if cfg.backend == "transformers":
                _backend_cache.clear()  # keep only one local model in memory
            _backend_cache[key] = BACKENDS[cfg.backend](cfg)
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
    return prompt, keys


# ------------------------------------------------------------------ parsing
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
# Placeholder answers small models give instead of leaving a key out.
_EMPTY_RE = re.compile(r"^(?:none|n/?a|null|unknown|nothing|not (?:visible|applicable|present|shown|clear|available)"
                       r"|no (?:person|people|one|man|woman|human)s?(?: visible| present)?)$", re.IGNORECASE)


def parse_response(text: str) -> Dict[str, List[str]]:
    """Pull the JSON object out of a model reply and normalise it to lists of strings."""
    m = _FENCE_RE.search(text)
    body = m.group(1) if m else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("모델 응답에서 JSON 을 찾지 못했습니다.")
    raw = body[start:end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = json.loads(re.sub(r",\s*([}\]])", r"\1", raw))  # trailing commas
    out: Dict[str, List[str]] = {}
    for key, value in data.items():
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

    def analyze(self, image: Any) -> Dict[str, Any]:
        img = load_image(image, self.config.max_image_side)
        prompt, keys = build_prompt(self.data_dirs, self.config)
        raw = self.backend.chat(img, prompt)
        found = parse_response(raw)
        limit = max(1, int(self.config.max_items_per_key))
        found = {k: v[:limit] for k, v in found.items() if k not in self.config.skip_keys}
        rows = classify(self.data_dirs, found, keys)
        new = {k: [r["text"] for r in v if r["status"] == "new"] for k, v in rows.items()}
        return {"elements": rows, "new": {k: v for k, v in new.items() if v}, "raw": raw}

    def apply(self, items: Dict[str, List[str]], rating: str = "sfw") -> Dict[str, Any]:
        report = append_items(self.write_dir, items, rating=rating, search_dirs=self.data_dirs)
        if self.maker is not None and report["added_count"]:
            self.maker.reload()
        return report

    def analyze_and_apply(self, image: Any, rating: str = "sfw") -> Dict[str, Any]:
        result = self.analyze(image)
        result["report"] = self.apply(result["new"], rating)
        return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="이미지를 분석해 요소 txt 에 새 항목을 추가합니다.")
    parser.add_argument("images", nargs="+", help="이미지 파일 경로")
    parser.add_argument("--apply", action="store_true", help="새 항목을 txt 에 실제로 추가 (없으면 미리보기)")
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
    if args.save_config:
        print("설정 저장:", save_config(cfg))
    analyzer = ImageAnalyzer(PromptMaker(user_data_dir=args.user_data), cfg)
    for path in args.images:
        result = analyzer.analyze(path)
        print(f"== {path}")
        for key, rows in result["elements"].items():
            for r in rows:
                print(f"  [{r['status']:<11}] {key}: {r['text']}")
        if args.apply:
            rep = analyzer.apply(result["new"], args.rating)
            print(f"  -> {rep['added_count']}개 추가")


if __name__ == "__main__":
    main()
