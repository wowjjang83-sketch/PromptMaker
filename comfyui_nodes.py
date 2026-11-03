"""ComfyUI custom nodes for PromptMaker.

설치: 이 저장소 폴더를 ComfyUI/custom_nodes/PromptMaker 로 복사(또는 git clone)하세요.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from promptmaker import GenerateOptions, PromptMaker  # noqa: E402
from promptmaker.core import MODES  # noqa: E402
from promptmaker.data_loader import RATINGS  # noqa: E402

_USER_DATA = os.path.join(_HERE, "user_data")
_MAKER = PromptMaker(user_data_dir=_USER_DATA if os.path.isdir(_USER_DATA) else None)


def _template_choices():
    return ["random"] + sorted(_MAKER.templates)


class PromptMakerGenerate:
    """사용자 요청 + 옵션 -> 영문 자연어 프롬프트."""

    CATEGORY = "PromptMaker"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt", "info_json", "seed")
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "user_input": ("STRING", {"multiline": True, "default": ""}),
                "prompt_type": (["image", "video"],),
                "mode": (list(MODES), {"default": "random_by_type"}),
                "template": (_template_choices(),),
                "content_rating": (RATINGS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "translate": ("BOOLEAN", {"default": True}),
                "use_related_words": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "selections_json": ("STRING", {"multiline": True, "default": "{}"}),
                "random_elements": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"multiline": True, "default": ""}),
                "suffix": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def generate(self, user_input, prompt_type, mode, template, content_rating, seed,
                 translate, use_related_words, selections_json="{}", random_elements="",
                 prefix="", suffix=""):
        try:
            selections = json.loads(selections_json or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"selections_json 형식 오류: {exc}")
        opts = GenerateOptions.from_dict({
            "prompt_type": prompt_type, "mode": mode, "template": template,
            "content_rating": content_rating, "seed": int(seed), "translate": translate,
            "use_related_words": use_related_words, "selections": selections,
            "random_elements": random_elements,
        })
        result = _MAKER.generate(user_input, opts)
        prompt = " ".join(p for p in (prefix.strip(), result.prompt, suffix.strip()) if p)
        info = {k: v for k, v in result.to_dict().items() if k not in ("system_prompt", "model_example")}
        return (prompt, json.dumps(info, ensure_ascii=False, indent=2), result.seed)


class PromptMakerReload:
    """JSON/TXT 데이터 파일을 수정한 뒤 다시 읽어 들입니다."""

    CATEGORY = "PromptMaker"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "reload"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trigger": ("INT", {"default": 0, "min": 0, "max": 9999})}}

    def reload(self, trigger):
        _MAKER.reload()
        return (f"templates={len(_MAKER.templates)}, elements={len(_MAKER.elements)}",)


class _ComfyProgress:
    """Progress bar on the node + console line + ComfyUI's cancel button, for vision.ImageAnalyzer."""

    def __init__(self, batch: int):
        self.batch, self.index = batch, 0
        self._stage, self._printed = "", 0.0
        try:
            from comfy.utils import ProgressBar

            self.bar = ProgressBar(100 * batch)
        except ImportError:  # outside ComfyUI
            self.bar = None

    def __call__(self, p) -> None:
        import time

        if self.bar is not None:
            self.bar.update_absolute(int(100 * self.index + p.percent), 100 * self.batch)
        now = time.monotonic()
        if p.stage != self._stage or now - self._printed >= 2:  # console: stage changes + every 2 s
            print(f"[PromptMaker 이미지 분석 {self.index + 1}/{self.batch}] {p}", flush=True)
            self._stage, self._printed = p.stage, now

    def is_set(self) -> bool:  # used as the "cancel" flag
        try:
            import comfy.model_management as mm

            return mm.processing_interrupted()
        except ImportError:
            return False


class PromptMakerImageAnalyze:
    """이미지 -> 비전 모델(기본 Qwen3-VL-4B) -> 요소별 문구 -> 중복 확인 후 요소 txt 에 추가."""

    CATEGORY = "PromptMaker"
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("report", "elements_json", "added_count")
    FUNCTION = "analyze"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        from promptmaker import vision

        cfg = vision.load_config()
        return {
            "required": {
                "image": ("IMAGE",),
                "write_to_txt": ("BOOLEAN", {"default": True}),
                "content_rating": (RATINGS,),
                "backend": (list(vision.BACKENDS), {"default": cfg.backend}),
                "model_path": ("STRING", {"default": cfg.model_path,
                                          "tooltip": "모델 폴더 위치 (비우면 model_id 를 자동 다운로드)"}),
                "model_id": ("STRING", {"default": cfg.model_id}),
            },
            "optional": {
                "api_base": ("STRING", {"default": cfg.api_base}),
                "max_items_per_key": ("INT", {"default": cfg.max_items_per_key, "min": 1, "max": 10}),
                "save_as_default": ("BOOLEAN", {"default": False}),
            },
        }

    def analyze(self, image, write_to_txt, content_rating, backend, model_path, model_id,
                api_base="", max_items_per_key=3, save_as_default=False):
        import numpy as np
        from PIL import Image

        from promptmaker import vision

        cfg = vision.load_config()
        cfg.backend, cfg.model_path, cfg.model_id = backend, model_path.strip(), model_id.strip() or cfg.model_id
        cfg.api_base = api_base.strip() or cfg.api_base
        cfg.max_items_per_key = int(max_items_per_key)
        if save_as_default:
            vision.save_config(cfg)
        analyzer = vision.ImageAnalyzer(_MAKER, cfg)
        lines, merged, added = [], {}, 0
        batch = image.shape[0]
        reporter = _ComfyProgress(batch)
        for i in range(batch):  # ComfyUI IMAGE: [B, H, W, C] float 0..1
            arr = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            reporter.index = i
            try:
                result = analyzer.analyze(Image.fromarray(arr), progress=reporter, cancel=reporter)
            except vision.AnalysisCancelled:
                import comfy.model_management as mm

                mm.throw_exception_if_processing_interrupted()  # ComfyUI's own "interrupted" handling
                raise
            s = result["stats"]
            lines.append(f"== 이미지 {i + 1}/{batch}: 입력 {s['input_tokens']} 토큰, 출력 {s['output_tokens']} 토큰, "
                         f"{s['tokens_per_sec']:.1f} tok/s, {vision._fmt_time(s['elapsed'])} 소요")
            for key, rows in result["elements"].items():
                merged.setdefault(key, []).extend(rows)
                lines += [f"[{r['status']}] {key}: {r['text']}" for r in rows]
            if write_to_txt and result["new"]:
                rep = analyzer.apply(result["new"], content_rating)
                added += rep["added_count"]
                lines += [f"+ {k}: {t}" for k, v in rep["added"].items() for t in v]
        lines.append(f"추가된 항목: {added}개" if write_to_txt else "미리보기 모드 (txt 에 추가하지 않음)")
        return ("\n".join(lines), json.dumps(merged, ensure_ascii=False, indent=2), added)


NODE_CLASS_MAPPINGS = {
    "PromptMakerGenerate": PromptMakerGenerate,
    "PromptMakerReload": PromptMakerReload,
    "PromptMakerImageAnalyze": PromptMakerImageAnalyze,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptMakerGenerate": "PromptMaker 프롬프트 생성",
    "PromptMakerReload": "PromptMaker 데이터 다시 읽기",
    "PromptMakerImageAnalyze": "PromptMaker 이미지 분석 → 요소 추가",
}
