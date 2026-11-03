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


NODE_CLASS_MAPPINGS = {
    "PromptMakerGenerate": PromptMakerGenerate,
    "PromptMakerReload": PromptMakerReload,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptMakerGenerate": "PromptMaker 프롬프트 생성",
    "PromptMakerReload": "PromptMaker 데이터 다시 읽기",
}
