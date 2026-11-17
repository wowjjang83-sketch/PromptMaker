"""로컬 텍스트 LLM (선택 기능) - 요소 분류 검증.

PromptMaker 엔진 자체는 LLM 없이 동작합니다. 이 모듈은 사용자가 원할 때만 쓰입니다.

용도
----
프롬프트를 요소별로 나눈 결과(``[{"key": "hair", "text": "long wavy black hair"}, ...]``)를
요소 txt 에 저장하기 전에 로컬 LLM(기본: Qwen) 에게 "이 문구가 이 요소에 맞는가?" 를 묻고,

* 맞으면 그대로 (``ok``)
* 다른 요소가 더 맞으면 그 요소로 옮김 (``move``)
* 어떤 요소에도 맞지 않으면 저장하지 않음 (``drop``)

속도 (최소 토큰)
----------------
* 틀린 항목만 ``번호=요소`` 로 답하게 하고 모두 맞으면 ``ok`` 한 단어만 받음 → 출력 토큰 대부분 1~10개
* 탐욕 디코딩(temperature 0), Qwen3 의 생각(thinking) 모드 끔, 요소 설명은 짧은 영어 한 줄
* 모델은 한 번 올리면 프로세스 안에 유지 (``unload_llm()`` 으로 내림)

수위
----
성인 콘텐츠(노출·선정적 표현)도 일반 문구처럼 분류합니다(거절·검열하지 않음).
미성년 연상 문구는 이 모듈 이전 단계(``safety.is_safe_item``)에서 이미 제외됩니다.

백엔드
------
``transformers``  모델 폴더(config.json + safetensors). Qwen3 / Qwen2.5 Instruct, Qwen3-VL 폴더도 가능
``llama_cpp``     .gguf 파일 (llama-cpp-python 설치 필요)
``openai``        OpenAI 호환 서버 (Ollama, LM Studio, llama-server, vLLM ...)
``auto``          모델 경로를 보고 자동 선택 (기본)

설정 파일: ``config/llm.json`` (환경변수 ``PROMPTMAKER_LLM_CONFIG`` 로 위치 지정)
"""
from __future__ import annotations

import gc
import json
import os
import re
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.environ.get("PROMPTMAKER_LLM_CONFIG") or os.path.join(REPO_ROOT, "config", "llm.json")
BACKENDS = ("auto", "vision", "transformers", "llama_cpp", "llama_server", "openai")
LLAMA_CPP_HELP = (
    "GGUF 모델은 llama.cpp 로 실행합니다. 둘 중 하나를 준비하세요: "
    "① 실행 파이썬에 llama-cpp-python 설치 (Qwen3-VL 처럼 새 모델은 최신 빌드 필요 · 이 모델을 이미 쓰는 ComfyUI 노드가 있으면 "
    "그 ComfyUI 파이썬에 이미 들어 있을 수 있음) ② llama.cpp 공식 배포(github.com/ggml-org/llama.cpp/releases)의 "
    "llama-server.exe 경로 지정 → 서버를 자동으로 띄워 사용. "
    "③ 설치 없이 쓰려면 모델 폴더를 비워 이미지 분석(VL) 모델을 함께 쓰세요."
)


@dataclass
class LLMConfig:
    backend: str = "auto"                  # auto | transformers | llama_cpp | openai
    model_path: str = ""                   # 모델 폴더 또는 .gguf 파일 (openai 백엔드는 비움)
    model_id: str = ""                     # openai 서버의 모델 이름
    api_base: str = "http://127.0.0.1:11434/v1"
    api_key: str = ""
    device: str = "auto"                   # auto | cuda | cpu
    dtype: str = "auto"                    # auto | bfloat16 | float16 | float32
    n_ctx: int = 4096                      # llama_cpp 컨텍스트 길이
    n_gpu_layers: int = -1                 # llama_cpp: -1 = 전부 GPU
    server_path: str = ""                  # llama_server: llama-server(.exe) 파일 또는 그 폴더
    server_port: int = 8091                # llama_server: 자동으로 띄울 포트
    batch_size: int = 40                   # 한 번에 검증할 항목 수
    max_new_tokens: int = 160              # 출력 상한 (틀린 항목만 답하므로 보통 10 토큰 이하)
    drop_unfit: bool = True                # 어떤 요소에도 맞지 않는 문구는 저장하지 않음

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "LLMConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    def to_dict(self, hide_key: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if hide_key and d.get("api_key"):
            d["api_key"] = "********"
        return d

    def resolved_backend(self) -> str:
        """실제로 쓸 백엔드. GGUF 는 transformers 로 열 수 없으므로 백엔드를 무엇으로 골랐든 llama.cpp 계열로 보냄."""
        if self.backend in ("openai", "vision"):
            return self.backend
        if not self.model_path:  # 모델을 따로 지정하지 않으면 이미지 분석(VL) 모델을 함께 씀
            return "openai" if self.model_id else "vision"
        if detect_model(self.model_path).get("kind") == "gguf":
            if self.backend in ("llama_cpp", "llama_server"):
                return self.backend
            if has_llama_cpp() or not find_llama_server(self.server_path):
                return "llama_cpp"
            return "llama_server"
        return "transformers"


def load_config(path: Optional[str] = None) -> LLMConfig:
    path = path or CONFIG_PATH
    try:
        with open(path, encoding="utf-8") as f:
            return LLMConfig.from_dict(json.load(f))
    except (OSError, ValueError):
        return LLMConfig()


def save_config(cfg: LLMConfig, path: Optional[str] = None) -> str:
    path = path or CONFIG_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def has_llama_cpp() -> bool:
    import importlib.util

    return importlib.util.find_spec("llama_cpp") is not None


def llama_cpp_version() -> str:
    try:
        import importlib.metadata as md

        return md.version("llama_cpp_python")
    except Exception:
        return ""


def find_llama_server(path: str) -> str:
    """llama-server 실행 파일 경로 (파일 또는 폴더 지정). 없으면 ""."""
    path = (path or "").strip().strip('"')
    if not path:
        return ""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for name in ("llama-server.exe", "llama-server"):
            for sub in ("", "bin", os.path.join("build", "bin"), os.path.join("build", "bin", "Release")):
                cand = os.path.join(path, sub, name)
                if os.path.isfile(cand):
                    return cand
    return ""


# ------------------------------------------------------------------ 모델 폴더 확인
def _gguf_files(folder: str) -> List[str]:
    try:
        names = sorted(n for n in os.listdir(folder) if n.lower().endswith(".gguf") and "mmproj" not in n.lower())
    except OSError:
        return []
    return [os.path.join(folder, n) for n in names]


def detect_model(path: str) -> Dict[str, Any]:
    """모델 경로 점검. ``{"ok", "kind": transformers|vl|gguf|"", "path", "file", "name", "size_gb", "message"}``"""
    path = (path or "").strip().strip('"')
    info: Dict[str, Any] = {"ok": False, "kind": "", "path": path, "file": "", "name": "", "size_gb": 0.0,
                            "architectures": [], "message": ""}
    if not path:
        info["message"] = "모델 폴더를 지정하세요."
        return info
    if os.path.isfile(path):
        if path.lower().endswith(".gguf"):
            info.update(ok=True, kind="gguf", file=path, name=os.path.basename(path),
                        size_gb=round(os.path.getsize(path) / 1024 ** 3, 2), message="GGUF 파일 (llama-cpp-python 필요)")
        else:
            info["message"] = "모델 폴더(config.json 이 있는 폴더) 또는 .gguf 파일을 지정하세요."
        return info
    if not os.path.isdir(path):
        info["message"] = f"경로가 없습니다: {path}"
        return info
    cfg_path = os.path.join(path, "config.json")
    if os.path.isfile(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                conf = json.load(f)
        except (OSError, ValueError):
            conf = {}
        arch = [str(a) for a in conf.get("architectures") or []]
        size = 0
        for n in os.listdir(path):
            if n.endswith((".safetensors", ".bin")):
                try:
                    size += os.path.getsize(os.path.join(path, n))
                except OSError:
                    pass
        vl = any("ConditionalGeneration" in a or "VL" in a for a in arch) or "vision_config" in conf
        info.update(ok=size > 0, kind="vl" if vl else "transformers", architectures=arch,
                    name=os.path.basename(os.path.normpath(path)), size_gb=round(size / 1024 ** 3, 2))
        info["message"] = ((f"{'비전-언어 모델(텍스트만 사용)' if vl else '텍스트 모델'} · {', '.join(arch) or '구조 정보 없음'}")
                           if size else "가중치 파일(.safetensors)이 없습니다.")
        return info
    ggufs = _gguf_files(path)
    if ggufs:
        f0 = ggufs[0]
        info.update(ok=True, kind="gguf", file=f0, name=os.path.basename(f0),
                    size_gb=round(os.path.getsize(f0) / 1024 ** 3, 2),
                    message=f"GGUF 파일 {len(ggufs)}개 중 '{os.path.basename(f0)}' 사용 (llama-cpp-python 필요)")
        return info
    info["message"] = "config.json 또는 .gguf 파일이 없는 폴더입니다."
    return info


# ------------------------------------------------------------------ 백엔드
_THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text or "").strip()


class TransformersLLM:
    def __init__(self, cfg: LLMConfig):
        info = detect_model(cfg.model_path)
        if info.get("kind") == "gguf":
            raise RuntimeError("GGUF 모델은 transformers 로 열 수 없습니다. 백엔드를 '자동' 또는 llama.cpp 로 바꾸세요. "
                               + LLAMA_CPP_HELP)
        if not info["ok"]:
            raise RuntimeError(f"LLM 모델 폴더 확인 실패: {info['message']}")
        try:
            import torch
            from transformers import AutoTokenizer
        except ImportError as err:
            raise RuntimeError(f"이 파이썬({sys.executable})에 torch / transformers 가 없습니다. "
                               "CUDA torch 가 설치된 파이썬(분석용 파이썬)에서 실행하세요.") from err
        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"이 파이썬({sys.executable}, torch {torch.__version__})은 GPU(CUDA)를 쓸 수 없습니다.")
        device = cfg.device if cfg.device in ("cpu", "cuda") else ("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.dtype != "auto":
            dtype = getattr(torch, cfg.dtype)
        else:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        path = cfg.model_path
        if info["kind"] == "vl":
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=dtype)
            proc = AutoProcessor.from_pretrained(path)
            self.tokenizer = getattr(proc, "tokenizer", proc)
            self._template = proc
        else:
            from transformers import AutoModelForCausalLM

            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self._template = self.tokenizer
        self.model.eval()
        self.model.to(device)
        self.device = device
        self.name = info["name"]
        self.usage: Dict[str, int] = {}

    def _render(self, messages: List[Dict[str, str]]) -> str:
        msgs = messages
        if self._template is not self.tokenizer:  # 프로세서 템플릿은 content 를 목록으로 받음
            msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        try:
            return self._template.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        except TypeError:
            return self._template.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def chat(self, messages: List[Dict[str, str]], max_new_tokens: int = 64) -> str:
        import torch

        text = self._render(messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None, pad_token_id=pad)
        out = out[:, n_in:]
        self.usage = {"input_tokens": n_in, "output_tokens": int(out.shape[1])}
        return strip_thinking(self.tokenizer.decode(out[0], skip_special_tokens=True))

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        self._template = None


class LlamaCppLLM:
    def __init__(self, cfg: LLMConfig):
        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise RuntimeError(f"이 파이썬({sys.executable})에 llama-cpp-python 이 없습니다. " + LLAMA_CPP_HELP) from err
        info = detect_model(cfg.model_path)
        if info.get("kind") != "gguf":
            raise RuntimeError(f"GGUF 파일을 찾지 못했습니다: {info['message']}")
        gpu_layers = 0 if cfg.device == "cpu" else int(cfg.n_gpu_layers)
        try:
            self.model = Llama(model_path=info["file"], n_ctx=int(cfg.n_ctx), n_gpu_layers=gpu_layers, verbose=False)
        except Exception as err:
            raise RuntimeError(f"llama-cpp-python {llama_cpp_version() or ''} 로 '{info['name']}' 을(를) 열지 못했습니다 ({err}). "
                               "모델 구조가 이 버전보다 새로우면(예: Qwen3-VL) 최신 llama-cpp-python 이 필요합니다. "
                               + LLAMA_CPP_HELP) from err
        self.name = info["name"]
        self.no_think = "qwen3" in self.name.lower() and "2507" not in self.name
        self.usage: Dict[str, int] = {}

    def chat(self, messages: List[Dict[str, str]], max_new_tokens: int = 64) -> str:
        msgs = [dict(m) for m in messages]
        if self.no_think and msgs:
            msgs[-1]["content"] += " /no_think"
        res = self.model.create_chat_completion(messages=msgs, max_tokens=max_new_tokens, temperature=0.0)
        usage = res.get("usage") or {}
        self.usage = {"input_tokens": int(usage.get("prompt_tokens", 0)),
                      "output_tokens": int(usage.get("completion_tokens", 0))}
        return strip_thinking(res["choices"][0]["message"].get("content") or "")

    def close(self) -> None:
        try:
            self.model.close()
        except Exception:
            pass
        self.model = None


class OpenAILLM:
    def __init__(self, cfg: LLMConfig):
        if not cfg.model_id:
            raise RuntimeError("OpenAI 호환 서버의 모델 이름(model_id)을 지정하세요.")
        self.cfg = cfg
        self.name = cfg.model_id
        self.usage: Dict[str, int] = {}

    def chat(self, messages: List[Dict[str, str]], max_new_tokens: int = 64) -> str:
        body = {"model": self.cfg.model_id, "messages": messages, "temperature": 0, "max_tokens": max_new_tokens,
                "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.cfg.api_base.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.cfg.api_key or 'none'}"})
        with urllib.request.urlopen(req, timeout=300) as res:
            data = json.loads(res.read().decode("utf-8"))
        usage = data.get("usage") or {}
        self.usage = {"input_tokens": int(usage.get("prompt_tokens", 0)),
                      "output_tokens": int(usage.get("completion_tokens", 0))}
        return strip_thinking(data["choices"][0]["message"].get("content") or "")

    def close(self) -> None:
        pass


class VisionShareLLM:
    """이미지 분석(VL) 모델을 텍스트 전용으로 함께 사용 (promptmaker.vision 설정 · 이미 올라가 있으면 그대로 재사용).

    따로 모델을 올리지 않으므로 VRAM 을 더 쓰지 않고, 추가 설치도 필요 없습니다.
    모델을 내리는 것은 이미지 분석 쪽(``vision.unload_backends``)이 맡습니다.
    """

    def __init__(self, cfg: LLMConfig):
        from . import vision

        self.vision = vision
        self.vcfg = vision.load_config()
        self.backend = vision.get_backend(self.vcfg)
        self.name = f"이미지 분석 모델 ({self.vcfg.model_ref if self.vcfg.backend == 'transformers' else self.vcfg.model_id})"
        self.device = getattr(self.backend, "device", "")
        self.usage: Dict[str, int] = {}

    def chat(self, messages: List[Dict[str, str]], max_new_tokens: int = 64) -> str:
        if not self.vision.is_loaded(self.vcfg):  # 이미지 분석 후 내려갔으면 다시 올림
            self.backend = self.vision.get_backend(self.vcfg)
        if self.vcfg.backend != "transformers":
            api = OpenAILLM(LLMConfig(backend="openai", model_id=self.vcfg.model_id, api_base=self.vcfg.api_base,
                                      api_key=self.vcfg.api_key))
            out = api.chat(messages, max_new_tokens=max_new_tokens)
            self.usage = api.usage
            return out
        import torch

        proc, model = self.backend.processor, self.backend.model
        msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]
        try:
            inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt", enable_thinking=False)
        except TypeError:
            inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt")
        inputs = inputs.to(model.device)
        n_in = int(inputs["input_ids"].shape[1])
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        out = out[:, n_in:]
        self.usage = {"input_tokens": n_in, "output_tokens": int(out.shape[1])}
        return strip_thinking(proc.batch_decode(out, skip_special_tokens=True)[0])

    def close(self) -> None:
        self.backend = None  # 모델은 이미지 분석 쪽 캐시에 남음 (vision.unload_backends 로 내림)


class LlamaServerLLM:
    """llama.cpp 의 llama-server 를 자동으로 띄워 OpenAI 호환 API 로 사용 (최신 모델 구조 지원이 가장 빠름)."""

    def __init__(self, cfg: LLMConfig):
        import subprocess

        exe = find_llama_server(cfg.server_path)
        if not exe:
            raise RuntimeError("llama-server 실행 파일을 찾지 못했습니다. " + LLAMA_CPP_HELP)
        info = detect_model(cfg.model_path)
        if info.get("kind") != "gguf":
            raise RuntimeError(f"GGUF 파일을 찾지 못했습니다: {info['message']}")
        port = int(cfg.server_port or 8091)
        gpu_layers = "0" if cfg.device == "cpu" else ("999" if int(cfg.n_gpu_layers) < 0 else str(cfg.n_gpu_layers))
        args = [exe, "-m", info["file"], "--host", "127.0.0.1", "--port", str(port), "-c", str(int(cfg.n_ctx)),
                "-ngl", gpu_layers, "--jinja"]
        kw: Dict[str, Any] = {}
        if os.name == "nt":
            kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        self.log_path = os.path.join(os.path.dirname(CONFIG_PATH), "llama_server.log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.proc = subprocess.Popen(args, stdout=self._log, stderr=subprocess.STDOUT, **kw)
        base = f"http://127.0.0.1:{port}"
        end = time.time() + 600
        while True:
            if self.proc.poll() is not None:
                self.close()
                raise RuntimeError(f"llama-server 가 종료되었습니다 (코드 {self.proc.returncode}). {self._tail()}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as res:
                    if res.status == 200:
                        break
            except Exception:
                pass
            if time.time() > end:
                self.close()
                raise RuntimeError("llama-server 가 10분 안에 준비되지 않았습니다. " + self._tail())
            time.sleep(0.5)
        self.api = OpenAILLM(LLMConfig(backend="openai", model_id=info["name"], api_base=base + "/v1"))
        self.name = info["name"]
        self.device = "llama-server"
        self.usage: Dict[str, int] = {}

    def _tail(self) -> str:
        try:
            self._log.flush()
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                return f.read()[-500:].strip()
        except Exception:
            return ""

    def chat(self, messages: List[Dict[str, str]], max_new_tokens: int = 64) -> str:
        out = self.api.chat(messages, max_new_tokens=max_new_tokens)
        self.usage = self.api.usage
        return out

    def close(self) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
        try:
            self._log.close()
        except Exception:
            pass


_BACKEND_CLASSES = {"vision": VisionShareLLM, "transformers": TransformersLLM, "llama_cpp": LlamaCppLLM, "llama_server": LlamaServerLLM,
                    "openai": OpenAILLM}
_cache: Dict[str, Any] = {"key": None, "llm": None, "loaded_at": 0.0, "load_seconds": 0.0}
_lock = threading.RLock()


def _cache_key(cfg: LLMConfig) -> Tuple:
    return (cfg.resolved_backend(), cfg.model_path, cfg.model_id, cfg.api_base, cfg.device, cfg.dtype,
            cfg.n_ctx, cfg.n_gpu_layers, cfg.server_path, cfg.server_port)


def get_llm(cfg: Optional[LLMConfig] = None):
    """LLM 백엔드 (한 번에 하나만 메모리에 유지)."""
    cfg = cfg or load_config()
    key = _cache_key(cfg)
    with _lock:
        if _cache["key"] == key and _cache["llm"] is not None:
            return _cache["llm"]
        unload_llm()
        started = time.time()
        llm = _BACKEND_CLASSES[key[0]](cfg)
        _cache.update(key=key, llm=llm, loaded_at=time.time(), load_seconds=round(time.time() - started, 1))
        return llm


def is_loaded(cfg: Optional[LLMConfig] = None) -> bool:
    with _lock:
        if _cache["llm"] is None:
            return False
        return cfg is None or _cache["key"] == _cache_key(cfg)


def unload_llm() -> Dict[str, Any]:
    """올린 LLM 을 내리고 GPU 캐시를 비움."""
    with _lock:
        llm = _cache.get("llm")
        had = llm is not None
        if had:
            try:
                llm.close()
            except Exception:
                pass
        _cache.update(key=None, llm=None, loaded_at=0.0)
    info: Dict[str, Any] = {"had_model": had, "gpu_freed_mb": 0}
    gc.collect()
    torch = sys.modules.get("torch")
    if had and torch is not None and torch.cuda.is_available():
        before = torch.cuda.memory_reserved()
        torch.cuda.empty_cache()
        info["gpu_freed_mb"] = int((before - torch.cuda.memory_reserved()) / 2 ** 20)
    return info


def status(cfg: Optional[LLMConfig] = None) -> Dict[str, Any]:
    cfg = cfg or load_config()
    with _lock:
        llm = _cache.get("llm")
        return {
            "loaded": llm is not None,
            "model": getattr(llm, "name", "") if llm is not None else "",
            "device": getattr(llm, "device", "") if llm is not None else "",
            "load_seconds": _cache.get("load_seconds", 0.0),
            "backend": cfg.resolved_backend() if (cfg.model_path or cfg.model_id) else "",
            "model_check": detect_model(cfg.model_path) if cfg.model_path else None,
            "python": sys.executable,
            "llama_cpp": llama_cpp_version() if has_llama_cpp() else "",
            "llama_server": find_llama_server(cfg.server_path),
        }


# ------------------------------------------------------------------ 요소 검증
# 요소별 짧은 영어 설명 (입력 토큰을 줄이기 위해 한 줄)
KEY_HINTS: Dict[str, str] = {
    "photo_style": "kind/style of photo (e.g. candid film photograph)",
    "nationality": "nationality/ethnicity adjective",
    "age": "age number",
    "beauty": "looks/body-type adjective (stunning, petite)",
    "hair": "hairstyle and hair color",
    "makeup": "makeup, lips, lashes, nails",
    "skin": "skin tone/texture",
    "expression": "facial expression, smile, gaze",
    "pose": "body posture/position (standing, lying, leaning...)",
    "action": "activity she is doing (holding a cup, laughing...)",
    "interaction": "what a couple does together",
    "top": "upper-body garment (shirt, blouse, crop top)",
    "bottom": "skirt, shorts, pants, jeans",
    "dress": "dress, gown, jumpsuit",
    "swimwear": "bikini, swimsuit",
    "sleepwear": "pajamas, nightgown, robe",
    "lingerie": "underwear, bra, panties, lingerie set",
    "outerwear": "coat, jacket, cardigan",
    "traditional": "traditional costume (hanbok, kimono)",
    "sportswear": "athletic/gym wear",
    "footwear": "shoes, heels, boots, barefoot",
    "legwear": "stockings, tights, socks",
    "accessory": "jewelry, hat, glasses, bag",
    "attire_state": "state of the clothes (wet, unbuttoned, slipping)",
    "male_attire": "a man's clothing",
    "male_look": "a man's looks adjective",
    "location": "place/setting (prepositional phrase)",
    "time_weather": "time of day, season, weather",
    "lighting": "light source/quality",
    "lighting_effect": "how light falls on her",
    "camera": "lens, shot type, angle, framing, focus",
    "color_grade": "colors, tones, film grain, grading",
    "mood": "mood/atmosphere adjective",
}

SYSTEM_PROMPT = (
    "You check a word bank for an adult image-prompt generator used privately by one adult. "
    "Explicit, nude and sexual content is allowed: judge it like any other phrase, never refuse, censor or comment. "
    "The generator searches each key's list for a word or short phrase and joins the picks into an English "
    "sentence, so every entry must be ONE short, reusable word or phrase that belongs to its key. "
    "Each phrase has a proposed key. Find the phrases whose key is wrong. "
    "Reply with only the wrong ones, one per line as `number=key`. "
    "Use `x` as the key if the phrase fits no key or is not reusable (a whole sentence, several things "
    "mixed together, gibberish, a quality tag). If every key is right, reply `ok`. No other words."
)


def build_messages(items: List[Dict[str, str]], keys: Iterable[str],
                   hints: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
    hints = {**KEY_HINTS, **(hints or {})}
    key_lines = "\n".join(f"{k}: {hints.get(k, k.replace('_', ' '))}" for k in keys)
    item_lines = "\n".join(f"{i}. [{it['key']}] {it['text']}" for i, it in enumerate(items, 1))
    user = f"Keys:\n{key_lines}\n\nPhrases:\n{item_lines}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


_ANSWER_RE = re.compile(r"(\d+)\s*[=:.)\-]\s*`?\[?([A-Za-z_]+)\]?`?")


def parse_reply(text: str, count: int, keys: Iterable[str]) -> Dict[int, str]:
    """``{번호(1부터): 요소 또는 "x"}`` — 모델이 답한 번호만 들어감 (없으면 '맞음')."""
    allowed = set(keys)
    out: Dict[int, str] = {}
    for m in _ANSWER_RE.finditer(text or ""):
        n, key = int(m.group(1)), m.group(2).strip().lower()
        if not 1 <= n <= count:
            continue
        if key in ("x", "none", "drop", "invalid"):
            out[n] = "x"
        elif key in allowed:
            out[n] = key
    return out


def verify_items(items: List[Dict[str, str]], keys: Optional[Iterable[str]] = None,
                 cfg: Optional[LLMConfig] = None, llm=None, hints: Optional[Dict[str, str]] = None
                 ) -> Dict[str, Any]:
    """요소 분류 검증.

    items: ``[{"key": ..., "text": ...}, ...]`` (다른 필드는 그대로 유지)
    keys:  옮길 수 있는 요소 목록 (비우면 KEY_HINTS 전체 + 항목의 요소)
    반환: ``{"items": [{..., "key": 최종 요소, "orig_key", "verdict": ok|move|drop}], "stats": {...}}``
    """
    cfg = cfg or load_config()
    started = time.time()
    rows = [dict(it) for it in items if str(it.get("text") or "").strip() and it.get("key")]
    key_list = list(dict.fromkeys(list(keys or KEY_HINTS) + [r["key"] for r in rows]))
    stats = {"count": len(rows), "moved": 0, "dropped": 0, "input_tokens": 0, "output_tokens": 0,
             "requests": 0, "seconds": 0.0, "model": "", "load_seconds": 0.0}
    if not rows:
        return {"items": [], "stats": stats}
    was_loaded = is_loaded(cfg)
    llm = llm or get_llm(cfg)
    if not was_loaded:
        stats["load_seconds"] = _cache.get("load_seconds", 0.0)
    stats["model"] = getattr(llm, "name", "")
    size = max(1, int(cfg.batch_size or 40))
    for start in range(0, len(rows), size):
        chunk = rows[start:start + size]
        messages = build_messages(chunk, key_list, hints)
        limit = min(int(cfg.max_new_tokens or 160), 8 + 6 * len(chunk))
        reply = llm.chat(messages, max_new_tokens=limit)
        usage = getattr(llm, "usage", {}) or {}
        stats["input_tokens"] += int(usage.get("input_tokens", 0))
        stats["output_tokens"] += int(usage.get("output_tokens", 0))
        stats["requests"] += 1
        answers = parse_reply(reply, len(chunk), key_list)
        for i, row in enumerate(chunk, 1):
            row["orig_key"] = row["key"]
            new = answers.get(i)
            if new == "x":
                row["verdict"] = "drop" if cfg.drop_unfit else "ok"
            elif new and new != row["key"]:
                row["verdict"] = "move"
                row["key"] = new
            else:
                row["verdict"] = "ok"
    stats["moved"] = sum(1 for r in rows if r["verdict"] == "move")
    stats["dropped"] = sum(1 for r in rows if r["verdict"] == "drop")
    stats["seconds"] = round(time.time() - started, 2)
    return {"items": rows, "stats": stats}


def verify_grouped(found: Dict[str, List[str]], **kwargs) -> Dict[str, Any]:
    """``{key: [text, ...]}`` 형식(이미지 분석 결과) 그대로 검증 → ``{"found": 새 묶음, "items", "stats"}``"""
    items = [{"key": k, "text": t} for k, vals in (found or {}).items() for t in vals]
    res = verify_items(items, **kwargs)
    out: Dict[str, List[str]] = {}
    for r in res["items"]:
        if r["verdict"] != "drop":
            out.setdefault(r["key"], []).append(r["text"])
    return {"found": out, **res}


# ------------------------------------------------------------------ 생성 시 요청 범위 검사
SCOPE_SYSTEM_PROMPT = (
    "An adult image-prompt generator used privately by one adult filled some details at random. "
    "Explicit, nude and sexual content is allowed: judge it like anything else, never refuse or censor. "
    "Find the details that contradict or change the user's request (another place, time, outfit, people, "
    "style or mood than asked). Details the request did not mention are fine if they fit it. "
    "Reply with only the numbers of the bad details, comma separated, or `ok`. No other words."
)
_NUM_RE = re.compile(r"\d+")


class ScopeChecker:
    """``PromptMaker(scope_checker=ScopeChecker())`` + ``GenerateOptions(scope_check=True)``.

    요청 문장과 '요청이 정하지 않아 관계 확률/무작위로 고른 요소'만 보내고, 범위를 벗어난 번호만 받음
    (입력 100~250 토큰, 출력 보통 1~5 토큰).
    """

    def __init__(self, cfg: Optional[LLMConfig] = None, llm=None):
        self.cfg = cfg
        self.llm = llm
        self.last: Dict[str, Any] = {}

    def check(self, request: str, picks: List[Tuple[str, str]]) -> List[int]:
        if not picks or not (request or "").strip():
            return []
        llm = self.llm or get_llm(self.cfg or load_config())
        lines = "\n".join(f"{i}. {k.replace('_', ' ')}: {t}" for i, (k, t) in enumerate(picks, 1))
        messages = [{"role": "system", "content": SCOPE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Request: {request.strip()}\nDetails:\n{lines}"}]
        started = time.time()
        reply = llm.chat(messages, max_new_tokens=4 + 3 * len(picks))
        bad = sorted({int(n) - 1 for n in _NUM_RE.findall(reply) if 1 <= int(n) <= len(picks)})
        self.last = {"reply": reply, "bad": bad, "seconds": round(time.time() - started, 2),
                     **(getattr(llm, "usage", {}) or {})}
        return bad


def main(argv=None) -> None:
    """``python -m promptmaker.llm "<key>=<text>" ...`` 로 빠르게 시험."""
    import argparse

    p = argparse.ArgumentParser(description="로컬 LLM 요소 분류 검증 시험")
    p.add_argument("items", nargs="*", help='"hair=long wavy black hair" 형식')
    p.add_argument("--model", default="", help="모델 폴더 또는 .gguf (비우면 config/llm.json)")
    args = p.parse_args(argv)
    cfg = load_config()
    if args.model:
        cfg.model_path = args.model
    items = [{"key": s.split("=", 1)[0].strip(), "text": s.split("=", 1)[1].strip()} for s in args.items if "=" in s]
    res = verify_items(items, cfg=cfg)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
