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


def _model_choices():
    # "default" = krea2 for image, minimax_h3 for video (see data/model_profiles/*.json)
    return ["default"] + sorted(_MAKER.model_profiles) + ["none"]


class PromptMakerGenerate:
    """사용자 요청 + 옵션 -> 영문 자연어 프롬프트."""

    CATEGORY = "PromptMaker"
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("prompt", "info_json", "seed", "negative_prompt")
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
                "model_profile": (_model_choices(), {"default": "default"}),
            },
            "optional": {
                "selections_json": ("STRING", {"multiline": True, "default": "{}"}),
                "random_elements": ("STRING", {"default": ""}),
                "prefix": ("STRING", {"multiline": True, "default": ""}),
                "suffix": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def generate(self, user_input, prompt_type, mode, template, content_rating, seed,
                 translate, use_related_words, model_profile="default", selections_json="{}", random_elements="",
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
            "model_profile": None if model_profile == "default" else model_profile,
        })
        result = _MAKER.generate(user_input, opts)
        prompt = " ".join(p for p in (prefix.strip(), result.prompt, suffix.strip()) if p)
        info = {k: v for k, v in result.to_dict().items() if k not in ("system_prompt", "model_example")}
        return (prompt, json.dumps(info, ensure_ascii=False, indent=2), result.seed, result.negative_prompt)


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


def _comfy_interrupted() -> bool:
    try:
        import comfy.model_management as mm

        return mm.processing_interrupted()
    except ImportError:  # outside ComfyUI
        return False


class _ComfyProgress:
    """Progress bar on the node + console line + ComfyUI's cancel button, for vision.AnalysisQueue."""

    def __init__(self, queue):
        self.queue = queue
        self._stage, self._printed = "", 0.0
        try:
            from comfy.utils import ProgressBar

            self.bar = ProgressBar(100 * len(queue.items))
        except ImportError:  # outside ComfyUI
            self.bar = None

    def __call__(self, item, p) -> None:
        import time

        if _comfy_interrupted():
            self.queue.stop()
        total = len(self.queue.items)
        index = self.queue.items.index(item)
        if self.bar is not None:
            self.bar.update_absolute(int(100 * index + p.percent), 100 * total)
        now = time.monotonic()
        if p.stage != self._stage or now - self._printed >= 2:  # console: stage changes + every 2 s
            print(f"[PromptMaker 이미지 분석 {index + 1}/{total} {item.name}] {p}", flush=True)
            self._stage, self._printed = p.stage, now


class PromptMakerImageAnalyze:
    """이미지(배치 또는 폴더) -> 비전 모델(기본 Qwen3-VL-4B) -> 요소별 문구 -> 중복 확인 후 요소 txt 에 추가.

    여러 장은 대기열로 차례로 분석하며, 풀 프롬프트(이미지를 묘사하는 영문 프롬프트)도 만들고 저장할 수 있습니다.
    """

    CATEGORY = "PromptMaker"
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("report", "elements_json", "added_count", "full_prompt")
    FUNCTION = "analyze"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        from promptmaker import vision

        cfg = vision.load_config()
        return {
            "required": {
                "write_to_txt": ("BOOLEAN", {"default": True}),
                "content_rating": (RATINGS,),
                "backend": (list(vision.BACKENDS), {"default": cfg.backend}),
                "model_path": ("STRING", {"default": cfg.model_path,
                                          "tooltip": "모델 폴더 위치 (비우면 model_id 를 자동 다운로드)"}),
                "model_id": ("STRING", {"default": cfg.model_id}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "여러 장(배치)이면 차례로 분석합니다"}),
                "image_folder": ("STRING", {"default": "", "tooltip": "폴더의 이미지를 모두 대기열에 넣어 차례로 분석 (image 와 함께 쓰면 둘 다)"}),
                "full_prompt": ("BOOLEAN", {"default": cfg.full_prompt, "tooltip": "이미지마다 풀 프롬프트도 생성"}),
                "save_full_prompt": ("BOOLEAN", {"default": False, "tooltip": "풀 프롬프트를 파일에 저장 (중복 제외)"}),
                "full_prompt_file": ("STRING", {"default": cfg.full_prompt_file, "tooltip": "비우면 output/full_prompts.txt"}),
                "api_base": ("STRING", {"default": cfg.api_base}),
                "max_items_per_key": ("INT", {"default": cfg.max_items_per_key, "min": 1, "max": 10}),
                "save_as_default": ("BOOLEAN", {"default": False}),
            },
        }

    def analyze(self, write_to_txt, content_rating, backend, model_path, model_id, image=None, image_folder="",
                full_prompt=False, save_full_prompt=False, full_prompt_file="", api_base="",
                max_items_per_key=3, save_as_default=False):
        import numpy as np
        from PIL import Image

        from promptmaker import vision

        cfg = vision.load_config()
        cfg.backend, cfg.model_path, cfg.model_id = backend, model_path.strip(), model_id.strip() or cfg.model_id
        cfg.api_base = api_base.strip() or cfg.api_base
        cfg.max_items_per_key = int(max_items_per_key)
        cfg.full_prompt = bool(full_prompt or save_full_prompt)
        cfg.full_prompt_file = full_prompt_file.strip()
        if save_as_default:
            vision.save_config(cfg)
        queue = vision.AnalysisQueue(vision.ImageAnalyzer(_MAKER, cfg), auto_apply=write_to_txt,
                                     rating=content_rating, auto_save_full_prompt=save_full_prompt)
        if image is not None:
            for i in range(image.shape[0]):  # ComfyUI IMAGE: [B, H, W, C] float 0..1
                arr = (image[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                queue.add(Image.fromarray(arr), f"image {i + 1}")
        if image_folder.strip():
            if not os.path.isdir(image_folder.strip()):
                raise ValueError(f"image_folder 를 찾을 수 없습니다: {image_folder}")
            queue.add_folder(image_folder.strip())
        if not queue.items:
            raise ValueError("image 입력이나 image_folder 중 하나는 있어야 합니다.")
        queue.run(progress=_ComfyProgress(queue))
        if _comfy_interrupted():
            import comfy.model_management as mm

            mm.throw_exception_if_processing_interrupted()  # ComfyUI's own "interrupted" handling

        lines, merged, added, prompts = [], {}, 0, []
        total = len(queue.items)
        for n, item in enumerate(queue.items, 1):
            if item.status != "done":
                lines.append(f"== {n}/{total} {item.name}: {item.status} {item.error}")
                continue
            result, s = item.result, item.result["stats"]
            lines.append(f"== {n}/{total} {item.name}: 입력 {s['input_tokens']} 토큰, 출력 {s['output_tokens']} 토큰, "
                         f"{s['tokens_per_sec']:.1f} tok/s, {vision._fmt_time(s['elapsed'])} 소요")
            for key, rows in result["elements"].items():
                merged.setdefault(key, []).extend(rows)
                lines += [f"[{r['status']}] {key}: {r['text']}" for r in rows]
            if item.report:
                added += item.report["added_count"]
                lines += [f"+ {k}: {t}" for k, v in item.report["added"].items() for t in v]
            if result.get("full_prompt"):
                prompts.append(result["full_prompt"])
                lines.append(f"풀 프롬프트: {result['full_prompt']}")
            if item.full_prompt_report and item.full_prompt_report["saved_count"]:
                lines.append(f"-> 풀 프롬프트 저장: {item.full_prompt_report['path']}")
        lines.append(f"추가된 항목: {added}개" if write_to_txt else "미리보기 모드 (txt 에 추가하지 않음)")
        return ("\n".join(lines), json.dumps(merged, ensure_ascii=False, indent=2), added, "\n\n".join(prompts))


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
