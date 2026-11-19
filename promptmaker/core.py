"""PromptMaker engine.

Pipeline::

    options -> user input -> (translate if not English) -> keyword search
    -> probability-based related words -> fill template with grammar rules -> output
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from . import body_parts as body_mod, framing as framing_mod, grammar, safety, undressing as undress_mod
from .data_loader import (PACKAGE_DATA_DIR, Element, Item, Template, load_elements,
                          load_json, load_model_profiles, load_templates, load_text_files,
                          merge_legacy_rating_files, SUBJECT_KEYS, SUBJECT_LEAD_RE)
from .intent import Intent, extract
from .restrictions import Restrictions
from .lexicon import LexiconResult, analyze
from .model_profiles import ModelProfile, format_prompt
from .relations import PLACE_TAGS, Relations, SceneContext
from .translator import DictionaryTranslator, GoogleTranslator, NoTranslator, Translator, is_english

MODES = ("manual", "random", "random_by_type", "random_by_element")
PROMPT_TYPES = ("image", "video")

_SLOT_RE = re.compile(r"\{([a-zA-Z_]\w*)(?::(\d+))?(\?)?\}")
_VARIANT_RE = re.compile(r"\{([^{}]*\|[^{}]*)\}")


class LLMRefiner(Protocol):
    """Optional future hook: polish the rule-based prompt with an LLM."""

    def refine(self, prompt: str, system_prompt: str, examples: str) -> str: ...


class ScopeChecker(Protocol):
    """Optional hook (0.5.0): finds picks that go beyond the user's request.

    ``check(request, picks)`` gets the request (original + English) and ``[(key, text), ...]`` of the
    elements the request did NOT fix (randomly / relation-picked), and returns the 0-based indexes of
    the picks that contradict or change the request. ``promptmaker.llm.ScopeChecker`` is a local-LLM one.
    """

    def check(self, request: str, picks: List[Tuple[str, str]]) -> List[int]: ...


@dataclass
class GenerateOptions:
    prompt_type: str = "image"                 # "image" | "video"
    template: Optional[str] = None             # template id, "random" or None
    mode: str = "random_by_type"               # see MODES
    selections: Dict[str, str] = field(default_factory=dict)  # element key -> fixed value
    random_elements: List[str] = field(default_factory=list)  # for mode "random_by_element"
    seed: Optional[int] = None
    translate: bool = True
    use_related_words: bool = True
    model_profile: Optional[str] = None        # target model: data/model_profiles/<id>.json (krea2, qwen2.1, moss, minimax_h3 ...)
    use_llm: bool = False                      # only used when an LLMRefiner is set
    scope_check: bool = False                  # only used when a ScopeChecker is set: re-pick elements
                                               # that go beyond the user's request (one LLM call)
    framing: str = "auto"                      # 샷 구도: auto(요청에서 찾음) | extreme_closeup | closeup | medium
                                               # | cowboy | full_body  (framing.FRAMINGS)
    body_part: str = ""                        # 신체 부위 클로즈업: eyes | lips | hair | hands | neck | shoulders | back
                                               # | chest | waist | hips | thighs | legs | feet (body_parts.PARTS)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerateOptions":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        opts = cls(**known)
        if isinstance(opts.random_elements, str):
            opts.random_elements = [s.strip() for s in opts.random_elements.split(",") if s.strip()]
        opts.selections = {k: v for k, v in (opts.selections or {}).items() if v not in (None, "")}
        return opts


@dataclass
class PromptResult:
    prompt: str
    prompt_type: str
    template_id: str
    template_name: str
    seed: int
    original_input: str = ""
    english_input: str = ""
    translated: bool = False
    picks: Dict[str, str] = field(default_factory=dict)
    lexicon: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    system_prompt: str = ""
    model_example: str = ""
    model_profile: str = ""
    negative_prompt: str = ""      # for models that take a separate negative prompt
    base_prompt: str = ""          # the prompt before the target-model formatting

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.prompt


class PromptMaker:
    """Rule/template based natural-language prompt generator."""

    def __init__(self, data_dir: Optional[str] = None, user_data_dir: Optional[str] = None,
                 translator: Optional[Translator] = None, llm: Optional[LLMRefiner] = None,
                 scope_checker: Optional[ScopeChecker] = None,
                 restrictions: Optional[Union[Restrictions, Dict[str, Any]]] = None):
        self.data_dirs = [data_dir or PACKAGE_DATA_DIR]
        user_dir = user_data_dir or os.environ.get("PROMPTMAKER_USER_DATA")
        self.user_dir = user_dir if user_dir and os.path.isdir(user_dir) else None
        # 국적 · 여성 나이 규칙 (기본 Korean/Japanese · 20~22). 사용자 폴더 restrictions.json 이 있으면 그것
        if isinstance(restrictions, Restrictions):
            self.restrictions = restrictions
        elif isinstance(restrictions, dict):
            self.restrictions = Restrictions.from_dict(restrictions)
        else:
            self.restrictions = Restrictions.load(self.user_dir)
        if user_dir and os.path.isdir(user_dir):
            self.data_dirs.append(user_dir)
            try:  # 예전 수위 파일(<key>.suggestive.txt 등)을 <key>.txt 로 합침 (0.4.0 수위 통합)
                merge_legacy_rating_files(user_dir)
            except OSError:
                pass  # 읽기 전용 폴더 등: 합치지 않아도 같은 요소로 읽힘
        self.llm = llm
        self.scope_checker = scope_checker
        self.reload()
        if translator is None:
            translator = GoogleTranslator(fallback=DictionaryTranslator(self.ko_en))
        self.translator = translator

    # ------------------------------------------------------------------ data
    def reload(self) -> None:
        """Re-read every JSON/TXT file (call after editing data files)."""
        self.elements: Dict[str, Element] = load_elements(self.data_dirs)
        self.templates: Dict[str, Template] = load_templates(self.data_dirs)
        self.related_db: Dict[str, dict] = load_json(self.data_dirs, "lexicon/related_words.json")
        self.ko_en: Dict[str, str] = load_json(self.data_dirs, "lexicon/ko_en.json")
        self.system_prompts: Dict[str, str] = load_text_files(self.data_dirs, "system")
        self.model_examples: Dict[str, str] = load_text_files(self.data_dirs, "model_examples")
        self.model_profiles: Dict[str, ModelProfile] = load_model_profiles(self.data_dirs)
        self.relations = Relations.load(load_json(self.data_dirs, "lexicon/relations.json"),
                                        load_json(self.data_dirs, "lexicon/tag_keywords.json"))
        self.relations.auto_tag(self.elements)

    def list_templates(self, prompt_type: Optional[str] = None) -> List[Template]:
        return [t for t in self.templates.values() if prompt_type in (None, t.type)]

    def list_elements(self) -> Dict[str, Element]:
        return self.elements

    def info(self) -> Dict[str, Any]:
        """Metadata used by the HTML page / ComfyUI node."""
        return {
            "prompt_types": list(PROMPT_TYPES),
            "modes": list(MODES),
            "templates": [
                {"id": t.id, "name": t.name, "type": t.type, "description": t.description,
                 "elements": [k for k in t.element_keys() if k in self.elements]}
                for t in self.templates.values()
            ],
            "elements": {
                k: {"label": e.label, "description": e.description,
                    "items": [{"text": i.text} for i in e.items]}
                for k, e in sorted(self.elements.items())
            },
            "model_profiles": sorted(set(self.model_examples) | set(self.model_profiles)),
            # 샷 구도 선택지 (auto = 요청에서 찾음, none = 구도를 넣지 않음)
            "framings": [{"id": "auto", "label": "자동 (요청에서 찾음)"}]
            + [{"id": f, "label": framing_mod.LABELS_KO[f]} for f in framing_mod.FRAMINGS]
            + [{"id": "none", "label": "넣지 않음"}],
            "body_parts": [{"id": "", "label": "없음"}]
            + [{"id": k, "label": body_mod.LABELS_KO[k]} for k in body_mod.PARTS],
            "model_profile_info": [p.to_info() for p in sorted(self.model_profiles.values(), key=lambda p: p.id)],
        }

    # -------------------------------------------------------------- pipeline
    def generate(self, user_input: str = "", options: Optional[GenerateOptions] = None,
                 **kwargs) -> PromptResult:
        if options is None:
            options = GenerateOptions.from_dict(kwargs)
        elif kwargs:
            options = GenerateOptions.from_dict({**asdict(options), **kwargs})
        if options.mode not in MODES:
            raise ValueError(f"알 수 없는 mode: {options.mode} (사용 가능: {', '.join(MODES)})")

        seed = options.seed if options.seed is not None and options.seed >= 0 \
            else random.randrange(0, 2**31 - 1)
        rng = random.Random(seed)
        warnings: List[str] = []

        # 1) user input + safety
        user_input = (user_input or "").strip()
        safety.check_text(user_input)
        for value in options.selections.values():
            safety.check_text(str(value))

        # 2) translation
        english = user_input
        translated = False
        if user_input and options.translate and not is_english(user_input):
            # '얼빡샷' 같은 구도 말은 번역기가 엉뚱하게 옮기므로 ('a quick shot') 먼저 영어로 바꿔 둠
            english = self.translator.translate(framing_mod.ko_to_english(user_input))
            translated = english != user_input
            err = getattr(self.translator, "last_error", None)
            if err:
                warnings.append(f"구글 번역 실패, 오프라인 사전으로 대체했습니다 ({err})")
            safety.check_text(english)
        elif user_input and not is_english(user_input):
            warnings.append("번역이 꺼져 있어 영어가 아닌 입력은 키워드 검색에 사용되지 않습니다.")
            english = ""

        # 3) keyword search + 4) related words
        lex = analyze(english, self.elements, self.related_db if options.use_related_words else {},
                      rng) if english else LexiconResult()

        # 4-1) hard constraints from the request (age, nationality, people, location ...)
        intent = extract(user_input, english, self.elements) if english or user_input else Intent()

        # 4-2) 샷 구도: 직접 고른 값 > 요청
        if options.framing and options.framing not in ("auto", "none", "", framing_mod.BODY_PART) \
                and options.framing not in framing_mod.FRAMINGS:
            raise ValueError(f"알 수 없는 framing: {options.framing} (사용 가능: auto, {', '.join(framing_mod.FRAMINGS)})")
        if options.body_part and options.body_part not in body_mod.PARTS:
            raise ValueError(f"알 수 없는 body_part: {options.body_part} (사용 가능: {', '.join(body_mod.PARTS)})")
        if len(set(intent.framings_found)) > 1 and options.framing in ("auto", "", None) \
                and not set(intent.framings_found) <= framing_mod.CLOSE:
            names = " + ".join(framing_mod.LABELS_KO[f] for f in intent.framings_found)
            warnings.append(f"샷 구도가 함께 쓸 수 없게 여러 개입니다 ({names}). "
                            f"'{framing_mod.LABELS_KO[intent.framing]}' 하나만 썼습니다.")

        # 5) template choice
        template = self._choose_template(options, rng, lex, intent)

        # 6) fill template
        literal = translated is False or getattr(self.translator, "last_backend", "") != "fallback"
        filler = _Filler(self, template, options, rng, lex, english, warnings, literal, intent)
        prompt = filler.render()

        # 6-1) scope guard: relation picks must stay inside what the user asked for
        scope: Dict[str, Any] = {}
        if options.scope_check and self.scope_checker is not None and (english or user_input):
            filler, prompt, scope = self._scope_guard(filler, prompt, template, options, seed, lex, english,
                                                      warnings, literal, intent, user_input)

        system_prompt = self.system_prompts.get("system_prompt", "")
        profile = options.model_profile or ""
        if not profile:  # default target model for the prompt type (krea2 / minimax_h3)
            profile = next((p.id for p in self.model_profiles.values() if p.default_for == template.type), "")
        if profile in ("none", "off"):
            profile = ""
        mprofile = self.model_profiles.get(profile) if profile else None
        if profile and mprofile is None and profile not in self.model_examples:
            warnings.append(f"모델 프로필 '{profile}' 을(를) 찾을 수 없어 기본 형식으로 출력합니다.")
        if mprofile and mprofile.type not in ("any", template.type):
            warnings.append(f"'{mprofile.label or mprofile.id}' 은(는) {mprofile.type} 모델인데 {template.type} 프롬프트에 적용했습니다.")
        model_example = self.model_examples.get(profile, "") if profile else ""
        if options.use_llm and self.llm is not None:
            try:
                prompt = self.llm.refine(prompt, system_prompt, model_example) or prompt
            except Exception as exc:
                warnings.append(f"LLM 보정 실패: {exc}")

        base_prompt = prompt
        formatted = format_prompt(prompt, filler.picks, filler.ctx.tags, mprofile, template.type)
        prompt = formatted.prompt

        return PromptResult(
            prompt=prompt, prompt_type=template.type, template_id=template.id,
            template_name=template.name, seed=seed, original_input=user_input,
            english_input=english, translated=translated, picks=filler.picks,
            lexicon={**lex.to_dict(), "intent": intent.to_dict(), "trace": filler.ctx.trace,
                     "scene_tags": sorted(filler.ctx.tags), "scope": scope}, warnings=warnings, system_prompt=system_prompt,
            model_example=model_example, model_profile=profile if (mprofile or profile in self.model_examples) else "",
            negative_prompt=formatted.negative, base_prompt=base_prompt,
        )

    def generate_batch(self, user_input: str = "", count: int = 4,
                       options: Optional[GenerateOptions] = None, **kwargs) -> List[PromptResult]:
        options = options or GenerateOptions.from_dict(kwargs)
        base = options.seed if options.seed is not None and options.seed >= 0 \
            else random.randrange(0, 2**31 - 1 - count)
        return [self.generate(user_input, options, seed=base + i) for i in range(count)]

    # -------------------------------------------------------------- scope guard
    SCOPE_SKIP = {"casting", "age", "outfit_type", "subject", "framing"}
    SCOPE_RETRIES = 2

    def _scope_targets(self, filler: "_Filler", intent: Intent, options: GenerateOptions) -> List[Tuple[str, str]]:
        """Picks the request did not fix (those are the ones relations / randomness chose)."""
        fixed = set(intent.pools) | set(intent.locks) | set(options.selections)
        out = []
        for key, text in filler.picks.items():
            if key in self.SCOPE_SKIP or key in fixed or not str(text).strip():
                continue
            if key == "attire" and fixed & (set(self.relations.garments) | {"attire"}):
                continue
            if intent.nude and key in ("attire", "attire_state"):  # 규칙으로 이미 비움
                continue
            if intent.undressing and key in (filler.undress_slot, "attire_state"):  # 요청한 탈의 동작
                continue
            out.append((key, str(text)))
        return out

    def _scope_guard(self, filler, prompt, template, options, seed, lex, english, warnings, literal, intent,
                     user_input):
        import time as _time

        request = user_input if not english or english == user_input else f"{user_input} ({english})"
        banned: Dict[str, set] = {}
        info: Dict[str, Any] = {"checked": 0, "flagged": [], "retries": 0, "seconds": 0.0}
        started = _time.time()
        for attempt in range(self.SCOPE_RETRIES + 1):
            targets = self._scope_targets(filler, intent, options)
            if not targets:
                break
            try:
                bad = self.scope_checker.check(request, targets) or []
            except Exception as exc:
                warnings.append(f"범위 검사 실패: {exc}")
                break
            info["checked"] += len(targets)
            bad = [targets[i] for i in sorted(set(bad)) if 0 <= i < len(targets)]
            if not bad:
                break
            info["flagged"].extend({"key": k, "text": t} for k, t in bad)
            if attempt == self.SCOPE_RETRIES:
                warnings.append("요청 범위를 벗어난 요소를 다시 골라도 남아 있습니다: "
                                + ", ".join(f"{k}={t}" for k, t in bad))
                break
            for key, text in bad:
                if key == "attire":  # composed outfit: ban the garments it was made of
                    for part in list(self.relations.garments) + ["outerwear", "footwear", "accessory"]:
                        for t in filler.cache.get(part, []):
                            banned.setdefault(part, set()).add(t.lower())
                else:
                    for t in filler.cache.get(key, [text]):
                        banned.setdefault(key, set()).add(t.lower())
            info["retries"] += 1
            filler = _Filler(self, template, options, random.Random(seed + 7919 * (attempt + 1)), lex, english,
                             warnings, literal, intent, banned=banned)
            prompt = filler.render()
        info["seconds"] = round(_time.time() - started, 2)
        return filler, prompt, info

    # -------------------------------------------------------------- helpers
    def _casting_items(self) -> List[Item]:
        el = self.elements.get("casting")
        return list(el.items) if el else []

    @staticmethod
    def _casting_allowed(t: Template, item: Item) -> bool:
        f = t.casting_filter
        fe, ma = int(item.meta.get("female", 1)), int(item.meta.get("male", 0))
        return (f.get("female_min", 0) <= fe <= f.get("female_max", 99)
                and f.get("male_min", 0) <= ma <= f.get("male_max", 99))

    def _choose_template(self, options: GenerateOptions, rng: random.Random,
                         lex: LexiconResult, intent: Optional[Intent] = None) -> Template:
        intent = intent or Intent()
        tid = options.template
        if tid and tid != "random":  # a template the user picked is used in every mode
            if tid not in self.templates:
                raise ValueError(f"템플릿 '{tid}' 을(를) 찾을 수 없습니다.")
            return self.templates[tid]
        # type: what the request says ("영상", "video" ...) wins, otherwise the selected type
        ptype = intent.prompt_type if options.mode == "random" and intent.prompt_type else options.prompt_type
        pool = self.list_templates(ptype)
        if not pool:
            raise ValueError("선택 가능한 템플릿이 없습니다. prompt_type 을 확인하세요.")
        # keep only templates that can show the people the request asked for
        if intent.female is not None or intent.male is not None:
            wanted = [i for i in self._casting_items()
                      if intent.casting_ok(int(i.meta.get("female", 1)), int(i.meta.get("male", 0)))]
            fit = [t for t in pool if any(self._casting_allowed(t, i) for i in wanted)]
            pool = fit or pool
        wanted_framing = options.framing if options.framing in framing_mod.FRAMINGS else intent.framing
        if options.body_part:
            wanted_framing = framing_mod.BODY_PART
        if options.framing == "none":
            wanted_framing = None
        # 부위 클로즈업 전용 템플릿 등 (framing_only) 은 그 구도일 때만
        pool = [t for t in pool if not t.raw.get("framing_only") or wanted_framing in t.raw.get("framings", ())] or pool
        if wanted_framing:  # 구도를 담을 수 있는 템플릿만 (전신샷에 '뷰티 클로즈업' 템플릿 X)
            fit = [t for t in pool if wanted_framing in t.raw.get("framings", framing_mod.FRAMINGS)]
            pool = fit or pool
            if wanted_framing in ("extreme_closeup", framing_mod.BODY_PART):  # 그 구도 전용 템플릿이 있으면 그것
                own = [t for t in pool if wanted_framing in t.raw.get("framings", ())]
                pool = own or pool
        if options.mode == "manual" and not tid:
            return pool[0]
        # prefer templates whose keywords / elements match the request
        kws = set(intent.template_keywords)

        def score(t: Template) -> int:
            tk = {k.lower() for k in t.raw.get("keywords", [])}
            keys = set(t.element_keys())
            if "attire" in keys:  # {attire} is composed from the garment elements
                keys |= set(self.relations.garments)
            wanted = set(intent.pools) | (set(intent.locks) - {"age", "nationality"})
            return 3 * len(kws & tk) + 2 * len(wanted & keys)

        # a requested garment must be visible: keep templates that have an {attire} slot
        if (set(intent.pools) | set(intent.locks)) & set(self.relations.garments):
            pool = [t for t in pool if "attire" in t.element_keys()] or pool
        best = max(score(t) for t in pool)
        if best > 0:
            pool = [t for t in pool if score(t) >= max(1, best - 1)]
        return rng.choice(pool)


STYLE_FIT = 6.0  # weight for photo/video styles that match the template's keywords
GARMENT_MENTION_KEYS = ("attire_state", "pose", "action", "motion", "undressing")
# garment words an attire_state / pose line may mention -> words that must be in the outfit
_GARMENT_WORDS = {
    "blouse": ("blouse",), "shirt": ("shirt", "blouse"), "skirt": ("skirt",), "dress": ("dress", "gown"),
    "jacket": ("jacket", "blazer", "coat"), "coat": ("coat", "trench"), "bikini": ("bikini",),
    "swimsuit": ("swimsuit", "monokini", "one-piece"), "lacing": ("corset", "bodice", "lace-up"),
    "corset": ("corset",), "stockings": ("stockings", "thigh-high"), "jeans": ("jeans",),
    "sweater": ("sweater", "knit", "cardigan"), "hoodie": ("hoodie",), "robe": ("robe",),
    "kimono": ("kimono", "yukata"), "hanbok": ("hanbok",), "bra": ("bra", "bralette", "bikini"),
    "shorts": ("shorts",), "tied at the neck": ("halter",), "halter": ("halter",),
    "unbuttoning": ("button", "shirt", "blouse", "cardigan", "coat", "jacket", "pajama", "pyjama", "nightshirt"),
    "unbuttoned": ("button", "shirt", "blouse", "cardigan", "coat", "jacket", "pajama", "pyjama", "nightshirt"),
    "unzipping": ("zip", "jacket", "hoodie", "dress", "skirt", "jeans", "pants", "shorts", "gown"),
    "unzipped": ("zip", "jacket", "hoodie", "dress", "skirt", "jeans", "pants", "shorts", "gown"),
    "one strap": ("strap", "cami", "slip", "bikini", "tank", "sundress", "bra"), "zipper": ("zip", "jacket", "hoodie", "dress"), "buttons": ("button", "shirt", "blouse", "cardigan"),
}


NUDE_ATTIRE = ("nothing at all", "nothing but bare skin")
NUDE_STATE = "fully nude"
_NUDE_ITEM_RE = re.compile(r"\b(?:nude|naked|bare-chested|topless|undressed|unclothed)\b", re.I)
_SUBJECT_LEAD_RE = SUBJECT_LEAD_RE  # 요소 txt 에 잘못 들어간 문장 조각 → 생성에 쓰지 않음
# 한 프롬프트 안에서 반복되면 어색한 특징 단어 (예: 카메라 'soft bokeh' + 장소 '... bokeh background')
SIGNATURE_WORDS = ("bokeh", "glow", "golden hour", "golden-hour", "haze", "hazy", "film grain", "grain", "backlit",
                   "backlight", "rim light", "neon", "candlelight", "moonlight", "sunlight", "lens flare", "silhouette",
                   "depth of field", "shallow focus", "soft focus", "desaturated", "sun-kissed", "dewy", "glossy",
                   "slowly")
_SIGNATURE_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in SIGNATURE_WORDS) + r")\b", re.I)


def _stems(text: str) -> set:
    """단어 앞 5글자 (unbuttoning / unbutton 을 같은 말로 봄)."""
    return {w[:5] for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 3}


def _garments_fit(text: str, outfit: str) -> bool:
    low = text.lower()
    for word, needs in _GARMENT_WORDS.items():
        if re.search(r"\b" + word + r"s?\b", low) and not any(n in outfit for n in needs):
            return False
    return True


SCENE_TAGS = {"beach", "pool", "bedroom", "bathroom", "cafe", "bar", "street", "studio", "nature",
              "urban", "indoor", "outdoor"}
PICK_ORDER = ["location", "time_weather", "photo_style", "video_style", "attire", "attire_state",
              "pose", "action", "motion", "interaction", "expression", "ambient_motion", "lighting",
              "lighting_effect", "camera", "camera_movement", "color_grade", "mood", "pacing",
              "male_attire"]

# 샷 구도에 따라 걸러 내는 요소 (구도에 안 담기는 몸 부위·구도를 말하는 줄)
FRAMED_KEYS = ("pose", "action", "motion", "attire_state", "camera", "camera_movement", "photo_style",
               "video_style", "interaction", "lighting", "lighting_effect", "ambient_motion")
UNDRESS_SLOTS = ("action", "motion", "pose")
# 부위 클로즈업에서 그 부위를 말하는 줄만 쓰는 요소 (없으면 넣지 않음)
BODY_STRICT_KEYS = ("pose", "action", "motion", "interaction")  # 탈의 동작을 넣을 자리 (템플릿에 있는 첫 번째)

INTENT_NOISE = {"taken", "shot", "wearing", "year", "old", "years", "year-old", "years-old",
                "model", "models", "idol", "idols", "actress", "solo", "alone", "twenties",
                "thirties", "early", "late", "mid", "gravure", "pictorial",
                "photoshoot", "editorial", "magazine", "japanese", "korean", "chinese"}


class _Filler:
    """Fills one template. Kept separate so that state (picks, rng) is per call."""

    def __init__(self, maker: PromptMaker, template: Template, options: GenerateOptions,
                 rng: random.Random, lex: LexiconResult, english: str, warnings: List[str],
                 literal_input: bool = True, intent: Optional[Intent] = None,
                 banned: Optional[Dict[str, set]] = None):
        self.literal_input = literal_input
        self.banned = banned or {}  # key -> lower-case texts the scope guard rejected
        self.intent = intent or Intent()
        # request locks < explicit user selections
        self.selections = {**self.intent.locks, **options.selections}
        self.maker = maker
        self.t = template
        self.o = options
        self.rng = rng
        self.lex = lex
        self.english = english
        self.warnings = warnings
        self.picks: Dict[str, str] = {}
        self.cache: Dict[str, List[str]] = {}
        self.ctx = SceneContext(maker.relations)
        self.framing: Optional[str] = None
        self.body_part = ""
        self.framing_source = ""
        self.framing_phrase = ""
        self.undress_slot: Optional[str] = None
        self.undress_text = ""
        self._otype = ""
        req_tags = maker.relations.tags_for_request(english) if english else set()
        self.ctx.add(req_tags, provider=bool(req_tags & SCENE_TAGS))
        self.ctx.add(template.raw.get("tags", []))
        self.cast = self._choose_casting()
        self.computed = self._computed_slots()
        # 샷 구도: 동작·의상과 따로 정하고, 구도에 안 담기는 포즈·의상은 고르지 않음
        self.framing, self.framing_source = self._init_framing()
        self.framing_phrase = self._framing_phrase() if self.framing and self.framing_source != "element" else ""
        if self.framing and self.framing_source != "element":
            # 요청의 구도 말('close-up', 'full body')로 걸린 카메라 후보는 버림 — 구도 문구를 따로 넣음
            cam = self.intent.pools.get("camera")
            if cam and all(framing_mod.framing_in_text(c) for c in cam):
                self.intent.pools.pop("camera", None)
        if self.framing_phrase:
            self.picks["framing"] = self.framing_phrase
            self.ctx.trace.append(f"framing: {self.framing_phrase}  [{self.framing_source}]")
        self._prepick()

    # ---------------------------------------------------------- element picks
    def _is_random(self, key: str) -> bool:
        if self.o.mode in ("random", "random_by_type"):
            return True
        if self.o.mode == "random_by_element":
            r = self.o.random_elements
            return key in r or "*" in r or ("attire" in r and key in self.maker.relations.garments)
        return False

    def _candidates(self, key: str) -> List[Item]:
        el = self.maker.elements.get(key)
        items = list(el.items) if el else []
        known = {i.text.lower() for i in items}
        for hint in self.lex.element_hints.get(key, []):
            if hint.lower() not in known and safety.is_safe_item(hint):
                items.append(Item(text=hint, weight=1.0))
        if key not in SUBJECT_KEYS:
            kept = [i for i in items if not _SUBJECT_LEAD_RE.match(i.text)]
            items = kept or items
        rules = self.maker.restrictions
        if key in ("nationality", "age") and rules.enabled:
            kept = [i for i in items if rules.allows(key, i.text)]
            if not kept and key == "nationality" and rules.nat_on:
                kept = [Item(text=n) for n in rules.nationalities]
            if not kept and key == "age" and rules.age_on:
                lo = rules.female_age_min if rules.female_age_min is not None else safety.MIN_AGE
                hi = rules.female_age_max if rules.female_age_max is not None else lo
                kept = [Item(text=str(a)) for a in range(lo, hi + 1)]
            items = kept or items
        ban = self.banned.get(key)
        if ban:
            kept = [i for i in items if i.text.lower() not in ban]
            items = kept or items  # never leave an element empty
        return items

    def _default_for(self, key: str, items: List[Item]) -> Optional[str]:
        if key in self.t.defaults:
            return self.t.defaults[key]
        el = self.maker.elements.get(key)
        if el and el.default:
            return el.default
        return items[0].text if items else None

    def _weighted_pick(self, key: str, items: List[Item], n: int,
                       deterministic: bool = False) -> List[Item]:
        """Pick n items using base weight x request boost x scene-context relations."""
        boosts = self._boosts(key)
        scored = self.ctx.weigh(items, boosts)
        if key in ("photo_style", "video_style"):  # styles that fit the chosen template
            words = [k.lower() for k in self.t.raw.get("keywords", []) if len(k) > 3]
            scored = [(it, w * (STYLE_FIT if any(re.search(r"\b" + re.escape(k), it.text, re.I) for k in words) else 1.0),
                       why) for it, w, why in scored]
        if key in GARMENT_MENTION_KEYS and "attire" in self.cache:
            outfit = self.cache["attire"][0].lower()
            scored = [(it, w if _garments_fit(it.text, outfit) else 0.0, why if _garments_fit(it.text, outfit)
                       else "의상과 안 맞음") for it, w, why in scored]
        if not self.intent.nude and "attire" in self.cache and key != "attire_state":
            # 옷을 입은 장면인데 '누드' 포즈/동작이 섞이지 않게
            scored = [(it, 0.0, "의상과 모순(누드)") if _NUDE_ITEM_RE.search(it.text) else (it, w, why)
                      for it, w, why in scored]
        if self.framing and key in FRAMED_KEYS:
            scored = [(it, w, why) if self._fits_frame(key, it.text)
                      else (it, 0.0, f"구도({framing_mod.LABELS_KO[self.framing]})에 안 담김") for it, w, why in scored]
        elif not self.framing and self.framing_source != "off" and key in ("camera", "photo_style", "video_style"):
            # 구도를 정하게 될 줄은 요청한 의상·포즈가 담기는 것만 ('미니스커트' 요청에 '허리 위 미디엄샷' X)
            need = self._requested_rank()
            if need:
                def _ok(text: str) -> bool:
                    fr = framing_mod.framing_in_text(text)
                    return not fr or framing_mod.MAX_RANK[fr] >= need
                scored = [(it, w, why) if _ok(it.text) else (it, 0.0, "요청한 의상·포즈가 구도에 안 담김")
                          for it, w, why in scored]
        shoes = [s for s in self.cache.get("footwear", []) if s and s.lower() != "barefoot"]
        if shoes and key in UNDRESS_SLOTS:  # 신발을 신었는데 '맨발로 걷는' 포즈 X
            scored = [(it, 0.0, "신발과 모순(맨발)") if re.search(r"\bbarefoot\b", it.text, re.I) else (it, w, why)
                      for it, w, why in scored]
        if self.intent.undressing and key in UNDRESS_SLOTS and key != self.undress_slot:
            # 탈의 동작과 함께: 손이 바쁘거나 몸 전체가 움직이는 포즈·동작은 빼기
            scored = [(it, 0.0, "탈의 동작과 함께 못 함") if undress_mod.BUSY_RE.search(it.text) else (it, w, why)
                      for it, w, why in scored]
        used_sig = self._used_signatures()
        if used_sig:
            scored = [(it, w * 0.1, (why + " " if why else "") + "특징 단어 반복")
                      if {m.lower() for m in _SIGNATURE_RE.findall(it.text)} & used_sig else (it, w, why)
                      for it, w, why in scored]
        valid = [(it, w, why) for it, w, why in scored if w > 0]
        if not valid and self.framing == framing_mod.BODY_PART and key in BODY_STRICT_KEYS:
            self.ctx.trace.append(f"{key}: (생략)  [{body_mod.LABELS_KO[self.body_part]}에 맞는 항목 없음]")
            return []  # 부위 클로즈업에 안 맞는 포즈·동작은 넣지 않음 (문장째 빠짐)
        if not valid:
            # nothing satisfies the relation rules: fall back to plain weights, but say so
            self.warnings.append(f"'{key}': 장면 규칙에 맞는 항목이 없어 규칙 없이 선택했습니다.")
            base = [it for it in items if self._fits_frame(key, it.text)] if self.framing and key in FRAMED_KEYS else []
            valid = [(it, it.weight * boosts.get(it.text.lower(), 1.0), "규칙 완화") for it in (base or items)]
        chosen: List[Item] = []
        pool = list(valid)
        for _ in range(min(n, len(pool))):
            if deterministic:
                idx = 0
            else:
                idx = self.rng.choices(range(len(pool)), weights=[w for _, w, _ in pool], k=1)[0]
            it, w, why = pool.pop(idx)
            chosen.append(it)
            self.ctx.trace.append(f"{key}: {it.text}" + (f"  [{why}]" if why else ""))
        return chosen

    def _used_signatures(self) -> set:
        out = set()
        for texts in self.cache.values():
            for t in texts:
                out |= {m.lower() for m in _SIGNATURE_RE.findall(t)}
        return out

    def _boosts(self, key: str) -> Dict[str, float]:
        """Request boosts, minus direct matches on words another element already used.

        e.g. "on the beach" fixed the location; it must not also force the pose
        "lying on a beach towel" just because that line contains "beach".
        """
        boosts = self.lex.boost_for(key)
        if not boosts or key in self.intent.pools:
            return boosts
        hints = {h.lower() for h in self.lex.element_hints.get(key, [])}
        fresh = [k for k in self.lex.keywords if k not in set(self.intent.consumed)]
        out = {}
        for text, b in boosts.items():
            if text in hints or any(re.search(r"\b" + re.escape(k), text, re.I) for k in fresh):
                out[text] = b
        return out

    def _commit(self, key: str, items: List[Item]) -> List[str]:
        provider = key in self.maker.relations.providers
        for it in items:
            # only the scene elements may set the place; flip-flops do not make a beach
            self.ctx.add(it.tags if provider else it.tags - PLACE_TAGS, provider=provider)
        texts = [it.text for it in items]
        self.cache[key] = texts
        if self.framing is None and self.framing_source != "off" and key in ("camera", "photo_style", "video_style") \
                and texts:
            fr = framing_mod.framing_in_text(texts[0])
            if fr:  # 고른 카메라/스타일이 구도를 말함 ('full-body shot'): 나머지를 그 구도에 맞춤
                self.framing, self.framing_source = fr, "element"
                self.ctx.trace.append(f"framing: {fr}  [{key} 에서]")
        return texts

    # ---------------------------------------------------------- framing
    def _init_framing(self) -> Tuple[Optional[str], str]:
        opt = (self.o.framing or "auto").strip()
        if opt == "none":
            return None, "off"
        if self.o.body_part in body_mod.PARTS:
            self.body_part = self.o.body_part
            return framing_mod.BODY_PART, "option"
        if opt in framing_mod.FRAMINGS:
            return opt, "option"
        sel_part = self.selections.get("body_closeup")
        if sel_part:  # 요소 목록에서 부위 클로즈업 문구를 골랐을 때
            el = self.maker.elements.get("body_closeup")
            item = next((i for i in (el.items if el else []) if i.text.lower() == str(sel_part).lower()), None)
            part = next((p for p in body_mod.PARTS if item is not None and p in item.tags), None) \
                or body_mod.part_in_text(str(sel_part))
            if part:
                self.body_part = part
                return framing_mod.BODY_PART, "selection"
        if self.intent.body_part and opt in ("auto", "", framing_mod.BODY_PART):
            self.body_part = self.intent.body_part
            return framing_mod.BODY_PART, "request"
        sel = self.selections.get("framing")
        if sel:
            el = self.maker.elements.get("framing")
            item = next((i for i in (el.items if el else []) if i.text.lower() == str(sel).lower()), None)
            fr = next((f for f in framing_mod.FRAMINGS if item is not None and f in item.tags), None) \
                or framing_mod.framing_in_text(str(sel))
            if fr:
                return fr, "selection"
        if self.intent.framing and self.intent.framing != framing_mod.BODY_PART:
            return self.intent.framing, "request"
        tf = [f for f in self.t.raw.get("framings", []) if f in framing_mod.FRAMINGS or f == framing_mod.BODY_PART]
        if tf:
            if tf[0] == framing_mod.BODY_PART:  # 부위 클로즈업 템플릿을 직접 골랐는데 부위가 없음: 무작위 부위
                self.body_part = self.rng.choice(list(body_mod.PARTS))
            return tf[0], "template"
        return None, ""

    def _framing_phrase(self) -> str:
        if self.framing == framing_mod.BODY_PART:
            if self.framing_source == "selection":
                text = str(self.selections["body_closeup"])
            else:
                el = self.maker.elements.get("body_closeup")
                items = [i for i in (el.items if el else [])
                         if self.body_part in i.tags or (not set(body_mod.PARTS) & i.tags
                                                         and body_mod.part_in_text(i.text) == self.body_part)]
                text = (self.rng.choices(items, weights=[max(i.weight, 0.01) for i in items], k=1)[0].text
                        if items else body_mod.DEFAULT_PHRASES[self.body_part])
            return self._resolve(text) or text
        if self.framing_source == "selection":
            text = str(self.selections["framing"])
        else:
            el = self.maker.elements.get("framing")
            items = [i for i in (el.items if el else []) if self.framing in i.tags]
            if not items:  # 태그 없이 추가된 줄 (직접 편집·이미지 분석): 문구로 구도 판단
                items = [i for i in (el.items if el else []) if not set(framing_mod.FRAMINGS) & i.tags
                         and framing_mod.framing_in_text(i.text) == self.framing]
            if items:
                text = self.rng.choices(items, weights=[max(i.weight, 0.01) for i in items], k=1)[0].text
            else:
                text = framing_mod.DEFAULT_PHRASES[self.framing]
        return self._resolve(text) or text

    def _requested(self, key: str) -> bool:
        return key in self.intent.pools or key in self.intent.locks or key in self.o.selections

    def _requested_rank(self) -> int:
        """요청(또는 직접 고른 값)한 의상·포즈를 담으려면 필요한 몸 범위."""
        need = max([framing_mod.GARMENT_RANK.get(p, 0) for p in framing_mod.GARMENT_RANK
                    if p != "accessory" and self._requested(p)] or [0])
        for key in ("pose", "action", "motion"):
            texts = list(self.intent.pools.get(key, [])) + [v for v in (self.intent.locks.get(key),
                                                                         self.o.selections.get(key)) if v]
            if texts:
                need = max(need, min(framing_mod.region_rank(t) for t in texts))
        return need

    def _fits_frame(self, key: str, text: str) -> bool:
        if not self.framing:
            return True
        if self.framing == framing_mod.BODY_PART:
            return self._fits_part(key, text)
        if key == "camera" and self.framing_source != "element" and framing_mod.framing_in_text(text):
            return False  # 구도 문구를 따로 넣으므로 카메라는 렌즈·각도만
        if key in ("photo_style", "video_style", "camera"):
            return framing_mod.compatible(framing_mod.framing_in_text(text), self.framing)
        return framing_mod.fits(text, self.framing)

    _FACE_RE = re.compile(r"\b(?:face|cheeks?|complexion|eyes?|lips?|smil\w*|gaze|expression)\b", re.I)

    def _fits_part(self, key: str, text: str) -> bool:
        """부위 클로즈업에서 이 줄을 쓸 수 있는지."""
        part = self.body_part
        if key in ("photo_style", "video_style", "camera"):
            fr = framing_mod.framing_in_text(text)
            other = body_mod.part_in_text(text)
            if key == "camera" and self.framing_source != "element" and (fr or other):
                return False
            return (not fr or fr in framing_mod.CLOSE) and (not other or other == part)
        if key in BODY_STRICT_KEYS:  # 포즈·동작은 그 부위를 말하고, 멀리 떨어진 부위는 말하지 않는 것만
            far = [p for p in body_mod.PARTS if p != part and p not in body_mod.NEIGHBORS.get(part, ())
                   and body_mod.mentions(text, p)]
            return body_mod.mentions(text, part) and not far
        if body_mod.mentions(text, part):
            return True
        # 의상 상태·조명 효과·카메라 움직임: 다른 부위·얼굴을 말하지 않으면 됨
        others = [p for p in body_mod.PARTS if p != part and body_mod.mentions(text, p)]
        face = part not in ("eyes", "lips") and self._FACE_RE.search(text)
        return not others and not face

    def _part_detail(self) -> str:
        """부위 클로즈업에서 그 부위를 꾸미는 한 문장 ('The focus is on her lips, with berry-stained lips')."""
        part = self.body_part
        if not part:
            return ""
        nouns = {"neck": "neck and collarbones", "chest": "neckline", "waist": "waist and midriff"}
        noun = nouns.get(part, part)
        detail = ""
        sources = {"eyes": ("makeup",), "lips": ("makeup",), "hair": ("hair",)}
        for key in sources.get(part, ()):
            items = [i for i in self._candidates(key) if body_mod.mentions(i.text, part)]
            if items:
                detail = self._weighted_pick(key, items, 1)[0].text if items else ""
                break
        if not detail and part == "hands":
            detail = "neatly manicured nails"
        if not detail and part not in ("eyes", "lips", "hair"):
            skin = self._one("skin")
            detail = f"{skin} skin" if skin else ""
        text = f"The focus is on {{her}} {noun}" + (f", with {detail}" if detail else "") + "."
        return self._resolve(text) or text

    # ---------------------------------------------------------- undressing
    GARMENT_PARTS = ("top", "bottom", "dress", "swimwear", "lingerie", "sleepwear", "sportswear", "traditional",
                     "outerwear")

    def _garment_names(self) -> Dict[str, str]:
        names: Dict[str, str] = {}
        for part in self.GARMENT_PARTS:
            vals = [v for v in self.cache.get(part, []) if v]
            if vals:
                noun = undress_mod.garment_noun(vals[0], part) or undress_mod.garment_noun(vals[0])
                if noun:
                    names[part] = noun
        asked = tuple(p for p in self.GARMENT_PARTS if self._requested(p))  # '블라우스를 벗는' → 블라우스가 대표
        main_parts = asked + self.OUTFIT_PARTS.get(self._otype, ()) + self.GARMENT_PARTS
        main = next((names[p] for p in main_parts if p in names), "")
        if not main and self.cache.get("attire") and self.cache["attire"][0]:
            main = undress_mod.garment_noun(self.cache["attire"][0])
        if main:
            names["garment"] = main
        return names

    def _undress_pick(self, key: str, source: str) -> List[str]:
        el = self.maker.elements.get(source)
        if not el:
            return []
        names = self._garment_names()
        items: List[Item] = []
        for it in el.items:
            types = it.requires & set(self.OUTFIT_PARTS)
            if types and self._otype not in types:
                continue
            text = undress_mod.fill_garments(it.text, names)
            if text and self._fits_frame(key if key in FRAMED_KEYS else "action", text):
                weight = it.weight
                parts = set(undress_mod.placeholders(it.text)) - {"garment"}
                asked = {p for p in self.GARMENT_PARTS if self._requested(p)}
                if asked and parts and not parts & asked:
                    weight *= 0.1  # 요청한 옷이 아닌 다른 옷을 벗는 줄 ('블라우스를 벗는' 에 '치마를 내리는')
                items.append(Item(text=text, weight=weight))
        # 요청이 말한 동작 ('단추를 푸는' → unbuttoning, '지퍼' → unzip) 이 들어간 줄을 우선
        req = _stems(self.english) - {"takin", "cloth", "undre", "woman", "women", "scene", "their", "while"}
        for item in items:
            item.weight *= 4.0 ** len(req & _stems(item.text))
        if not items:
            if source == "undressing":
                why = (f"{framing_mod.LABELS_KO[self.framing]}에는 탈의 동작이 담기지 않아"
                       if self.framing and (names or self.framing == "extreme_closeup") else "입은 옷에 맞는 탈의 동작이 없어")
                self.warnings.append(f"{why} 탈의 동작을 넣지 않았습니다.")
            return []
        chosen = self._weighted_pick(key, items, 1)
        texts = self._commit(key, chosen)
        if source == "undressing":
            self.undress_text = texts[0] if texts else ""
        return texts

    def pick(self, key: str, n: int = 1) -> List[str]:
        if key in self.cache and len(self.cache[key]) >= min(n, len(self.cache[key]) or n):
            return self.cache[key][:n]
        if key == "attire_state" and self.intent.nude and key not in self.selections:
            return self._commit(key, [Item(text=NUDE_STATE)])
        if self.intent.undressing and key not in self.selections:
            if key == self.undress_slot or key == "undressing":
                got = self._undress_pick(key, "undressing")
                if got or key == "undressing":
                    return got
            elif key == "attire_state" and "attire" in self.cache and self.cache["attire"][0]:
                got = self._undress_pick(key, "undressing_state")
                if got:
                    return got
        items = self._candidates(key)
        by_text = {i.text.lower(): i for i in items}
        if key in self.selections:
            value = str(self.selections[key])
            values = [v.strip() for v in value.split("|") if v.strip()] if "|" in value else [value]
            values = values[:max(n, 1)] if n > 1 else [values[0]]
            chosen = [by_text.get(v.lower(), Item(text=v)) for v in values]
            if not any(c.tags for c in chosen):  # user's own text: tag it on the fly
                rel = self.maker.relations
                for c in chosen:
                    c.tags = rel.tags_in_text(c.text) | rel.props_in_text(c.text)
            return self._commit(key, chosen)
        pool_texts = self.intent.pools.get(key)
        if pool_texts:  # the request mentioned this element: choose only among matches
            allowed = {t.lower() for t in pool_texts}
            restricted = [i for i in items if i.text.lower() in allowed]
            if restricted:
                return self._commit(key, self._weighted_pick(key, restricted, n))
        if not items:
            return []
        if not self._is_random(key):
            default = self._default_for(key, items)
            ordered = sorted(items, key=lambda i: 0 if default and i.text == default else 1)
            return self._commit(key, self._weighted_pick(key, ordered, n, deterministic=True))
        return self._commit(key, self._weighted_pick(key, items, n))

    # ---------------------------------------------------------- outfit
    OUTFIT_PARTS = {"top_bottom": ("top", "bottom"), "dress": ("dress",), "swimwear": ("swimwear",),
                    "sleepwear": ("sleepwear",), "traditional": ("traditional",),
                    "sportswear": ("sportswear",), "lingerie": ("lingerie",)}

    def _outfit(self) -> str:
        """Compose a coherent outfit: type -> garments -> layer -> shoes / accessories."""
        if "attire" in self.cache:
            return grammar.join_list(self.cache["attire"])
        if self.framing == framing_mod.BODY_PART and "attire" not in self.selections \
                and not set(body_mod.garments(self.body_part)) - {"accessory"}:
            # 눈·입술·머리카락·손 클로즈업: 옷은 보이지 않음 (손은 반지·팔찌만)
            acc = ""
            if "accessory" in body_mod.garments(self.body_part):
                fit = [i for i in self._candidates("accessory") if body_mod.mentions(i.text, self.body_part)]
                if fit and self.rng.random() < 0.6:
                    acc = self._commit("accessory", self._weighted_pick("accessory", fit, 1))[0]
            self.ctx.trace.append(f"attire: {acc or '(생략)'}  [{body_mod.LABELS_KO[self.body_part]}]")
            self.cache["attire"] = [acc]
            self.cache.setdefault("attire_state", [""])  # 반지·팔찌에 '구겨진' 같은 의상 상태는 붙이지 않음
            if acc:
                self.picks["attire"] = acc
            return acc
        if self.framing == "extreme_closeup" and "attire" not in self.selections:
            # 얼빡샷: 얼굴만 담기므로 의상·신발·액세서리를 넣지 않음
            asked = [p for p in list(self.GARMENT_PARTS) + ["footwear"]
                     if p in self.intent.pools or p in self.intent.locks]
            if asked:
                self.warnings.append("얼빡샷(얼굴만)이라 요청한 의상은 화면에 담기지 않아 넣지 않았습니다.")
            self.ctx.trace.append("attire: (생략)  [얼빡샷]")
            self.cache["attire"] = [""]
            return ""
        if self.intent.nude and "attire" not in self.selections:
            # 누드 요청: 의상·신발·액세서리를 고르지 않음
            text = self.rng.choice(NUDE_ATTIRE)
            self.ctx.add({"nude"})
            self.ctx.trace.append(f"attire: {text}  [누드 요청]")
            self.cache["attire"] = [text]
            self.picks["attire"] = text
            return text
        if "attire" in self.selections or "outfit_type" not in self.maker.elements:
            vals = self.pick("attire")
            return vals[0] if vals else ""
        forced = None
        for otype, parts in self.OUTFIT_PARTS.items():
            if any(p in self.intent.pools or p in self.selections for p in parts):
                forced = otype
                break
        if forced:
            otype = forced
            self.selections.setdefault("outfit_type", forced)
        otype = (self.pick("outfit_type") or ["top_bottom"])[0]
        self._otype = otype
        self.ctx.add({otype})
        parts = self.OUTFIT_PARTS.get(otype, ("top", "bottom"))
        body = self.framing == framing_mod.BODY_PART
        visible = set(body_mod.garments(self.body_part)) if body else set()
        if body:  # 부위 클로즈업: 그 부위에 보이는 옷만 (발 → 신발, 다리 → 하의·스타킹·신발)
            parts = tuple(p for p in parts if p in visible or self._requested(p))
        elif self.framing:  # 구도에 안 담기는 의상은 빼기 (미디엄샷: 하의 X). 요청한 의상은 남김
            parts = tuple(p for p in parts if framing_mod.garment_visible(p, self.framing)
                          or self._requested(p)) or parts[:1]
        garments = [g for g in (next(iter(self.pick(p)), "") for p in parts) if g]
        if not garments and not body:
            return ""
        main = grammar.join_list(garments)
        layered = otype in ("top_bottom", "dress", "traditional") and (not body or "outerwear" in visible) and main
        cold = bool(self.ctx.tags & {"winter", "rain", "snow"})
        if layered and "outerwear" in self.maker.elements and \
                self.rng.random() < (0.6 if cold else 0.15) and "summer" not in self.ctx.tags:
            outer = self.pick("outerwear")
            if outer:
                main += f" under {outer[0]}"
        extras, barefoot = [], False
        if body and "legwear" in visible and "legwear" in self.maker.elements and otype != "swimwear" \
                and self.rng.random() < 0.4:
            # 'bare legs' 는 옷이 아니므로 'wears' 뒤에 넣지 않음
            fit = [i for i in self._candidates("legwear") if not i.text.lower().startswith("bare")]
            leg = self._commit("legwear", self._weighted_pick("legwear", fit, 1)) if fit else []
            if leg:
                extras.append(leg[0])
        shoe_p = 0.5 if not body else (0.9 if self.body_part == "feet" else 0.5 if "footwear" in visible else 0.0)
        if otype != "sleepwear" and "footwear" in self.maker.elements and self.rng.random() < shoe_p \
                and (body or framing_mod.garment_visible("footwear", self.framing) or self._requested("footwear")):
            if otype == "swimwear":  # only beach-friendly shoes with swimwear
                self.intent.pools.setdefault("footwear", [i.text for i in self.maker.elements["footwear"].items
                                                          if "summer" in i.tags])
            shoe = self.pick("footwear")
            if shoe and shoe[0].lower() == "barefoot":
                barefoot = True
            elif shoe:
                extras.append(shoe[0])
        if "accessory" in self.maker.elements and self.rng.random() < (0.15 if " under " in main else 0.4):
            if body and not self._requested("accessory"):  # 그 부위의 액세서리만 (발 → 발찌)
                fit = [i for i in self._candidates("accessory") if body_mod.mentions(i.text, self.body_part)]
                acc = self._commit("accessory", self._weighted_pick("accessory", fit, 1)) if fit else []
            elif self.framing and not self._requested("accessory"):  # 구도에 담기는 액세서리만 (클로즈업에 핸드백 X)
                fit = [i for i in self._candidates("accessory") if framing_mod.fits(i.text, self.framing)]
                acc = self._commit("accessory", self._weighted_pick("accessory", fit, 1)) if fit else []
            else:
                acc = self.pick("accessory")
            if acc:
                extras.append(acc[0])
        joiner = ", paired with " if " with " in main else " with "  # avoid "with ... with ..."
        if main:
            text = main + (joiner + grammar.join_list(extras) if extras else "") + (", barefoot" if barefoot else "")
        else:  # 부위 클로즈업에서 옷 없이 신발·스타킹만
            text = grammar.join_list(extras)
            self.cache.setdefault("attire_state", [""])  # 신발·스타킹만이면 의상 상태 없음
        self.cache["attire"] = [text]
        self.picks["attire"] = text
        return text

    # ---------------------------------------------------------- order
    def _template_slots(self) -> Dict[str, int]:
        slots: Dict[str, int] = {}
        for s in self.t.sentences:
            for text in [s.get("text", "")] + list(s.get("options", [])):
                for m in _SLOT_RE.finditer(text):
                    k, n = m.group(1), int(m.group(2) or 1)
                    slots[k] = max(slots.get(k, 0), n)
        return slots

    def _prepick(self) -> None:
        """Choose elements in dependency order (scene first) so relations can apply."""
        slots = self._template_slots()
        ordered = [k for k in PICK_ORDER if k in slots] + [k for k in slots if k not in PICK_ORDER]
        if "camera" in ordered and not self.framing:  # 카메라가 구도를 정하면 나머지를 거기에 맞추도록 먼저
            ordered = ["camera"] + [k for k in ordered if k != "camera"]
        if self.intent.undressing:
            self.undress_slot = next((k for k in UNDRESS_SLOTS if k in slots), None)
            if "attire" not in slots:  # 탈의 동작이 말할 옷은 있어야 함
                ordered.insert(0, "attire")
            if self.undress_slot is None:
                ordered.append("undressing")
        for key in ordered:
            if key in self.computed or key in ("framing", "part_detail"):
                continue
            if key == "undressing":
                self.pick("undressing")
                continue
            if key == "attire":
                self._outfit()
            elif key in self.maker.elements or key in self.selections:
                self.pick(key, slots[key])

    def _choose_casting(self) -> Dict[str, Any]:
        el = self.maker.elements.get("casting")
        f = self.t.casting_filter
        items = []
        for i in (el.items if el else []):
            fe, ma = int(i.meta.get("female", 1)), int(i.meta.get("male", 0))
            if fe < f.get("female_min", 0) or fe > f.get("female_max", 99):
                continue
            if ma < f.get("male_min", 0) or ma > f.get("male_max", 99):
                continue
            items.append(i)
        wanted = [i for i in items if self.intent.casting_ok(int(i.meta.get("female", 1)),
                                                             int(i.meta.get("male", 0)))]
        if wanted:
            items = wanted
            if "casting" not in self.o.selections and len(wanted) == 1:
                chosen_fixed = wanted[0]
                self.picks["casting"] = chosen_fixed.text
                return {"female": int(chosen_fixed.meta.get("female", 1)),
                        "male": int(chosen_fixed.meta.get("male", 0)), "text": chosen_fixed.text}
        chosen: Optional[Item] = None
        sel = self.o.selections.get("casting")
        if sel:
            chosen = next((i for i in (el.items if el else []) if i.text == sel), None)
            if chosen is None:
                self.warnings.append(f"casting '{sel}' 을(를) 찾을 수 없어 무작위로 선택했습니다.")
        if chosen is None and items:
            if self._is_random("casting"):
                boosts = self.lex.boost_for("casting")
                weights = [i.weight * boosts.get(i.text.lower(), 1.0) for i in items]
                chosen = self.rng.choices(items, weights=weights, k=1)[0]
            else:
                default = self._default_for("casting", items)
                chosen = next((i for i in items if i.text == default), items[0])
        if chosen is None:
            chosen = Item(text="one woman", meta={"female": 1, "male": 0})
        self.picks["casting"] = chosen.text
        return {"female": int(chosen.meta.get("female", 1)), "male": int(chosen.meta.get("male", 0)),
                "text": chosen.text}

    # ------------------------------------------------------------- computed
    def _one(self, key: str, fallback: str = "") -> str:
        vals = self.pick(key)
        value = vals[0] if vals else fallback
        if key not in self.picks:
            self.picks[key] = value
        return value

    def _computed_slots(self) -> Dict[str, str]:
        fe, ma = self.cast["female"], self.cast["male"]
        age_raw = self._one("age", str(safety.MIN_AGE))
        age = safety.clamp_age(age_raw)
        if str(age) != str(age_raw).strip():
            self.warnings.append(f"나이 '{age_raw}' → {age}세로 조정했습니다 (최소 {safety.MIN_AGE}세).")
        rules = self.maker.restrictions
        if fe and rules.age_on and not rules.age_in_range(age):
            fixed = rules.clamp_age(age)
            self.warnings.append(f"나이 {age}세 → {fixed}세 (여성 나이 규칙 {rules.female_age_min}~{rules.female_age_max}세)")
            age = fixed
        self.picks["age"] = str(age)
        beauty = self._one("beauty")
        nationality = self._one("nationality")
        if fe and rules.nat_on:
            fixed = rules.nationality(nationality, self.rng)
            if nationality and fixed != nationality:
                self.warnings.append(f"국적 '{nationality}' → {fixed} (국적 규칙: {', '.join(rules.nationalities)})")
            nationality = fixed
            self.picks["nationality"] = nationality
        male_look = self._one("male_look") if ma else ""

        noun_s, noun_p = self.intent.female_noun or ("woman", "women")
        if noun_s != "woman" and noun_s in beauty.lower():  # avoid "model-like ... model"
            beauty = "stunning"
            self.picks["beauty"] = beauty
        female_core = " ".join(x for x in [beauty, f"{age}-year-old", nationality] if x)
        if fe == 1:
            female_subject = grammar.with_article(f"{female_core} {noun_s}")
        elif fe > 1:
            female_subject = f"{grammar.number_word(fe)} {female_core} {noun_p}"
        else:
            female_subject = ""
        if ma == 1:
            male_subject = grammar.with_article(f"{male_look} adult man".strip())
        elif ma > 1:
            male_subject = f"{grammar.number_word(ma)} {male_look} adult men".replace("  ", " ")
        else:
            male_subject = ""
        subject = grammar.join_list([female_subject, male_subject])

        plural = fe != 1
        c = {
            "subject": subject,
            "female_subject": female_subject,
            "male_subject": male_subject,
            "age": str(age),
            "she": "they" if plural else "she",
            "She": "They" if plural else "She",
            "her": "their" if plural else "her",
            "Her": "Their" if plural else "Her",
            "her_obj": "them" if plural else "her",
            "herself": "themselves" if plural else "herself",
            "woman": "women" if plural else "woman",
            "is": "are" if plural else "is",
            "has": "have" if plural else "has",
            "wears": "wear" if plural else "wears",
            "looks": "look" if plural else "looks",
            "s": "" if plural else "s",
            "partner": "the men" if ma > 1 else "the man",
            "Partner": "The men" if ma > 1 else "The man",
            "him": "them" if ma > 1 else "him",
            "his": "their" if ma > 1 else "his",
            "is_m": "are" if ma > 1 else "is",
            "user_details": self._details_sentence(),
        }
        return c

    def related_keys(self) -> List[str]:
        out = []
        db = self.maker.related_db
        for k in self.lex.keywords:
            if k in db or any(k in v.get("aliases", []) for v in db.values()):
                out.append(k)
        return out

    def _details_sentence(self, used_text: str = "") -> str:
        """Sentence carrying the user's own request + probabilistic related words.

        Words already expressed elsewhere in the prompt (``used_text``) are not repeated.
        """
        if not self.english:
            return ""
        adjs = self.lex.related.get("adjectives", [])
        nouns = self.lex.related.get("nouns", [])
        verbs = self.lex.related.get("verbs", [])
        text = self.english.strip().rstrip(".!?")
        text = re.sub(r"^(please\s+)?(make|create|generate|draw)\s+(me\s+)?(an?\s+)?"
                      r"(image|picture|photo|video|prompt)\s+(of\s+)?", "", text, flags=re.I)
        parts: List[str] = []
        from .lexicon import GENERIC_WORDS
        consumed = set(self.intent.consumed) | set(self.related_keys())
        leftover = [k for k in self.lex.keywords
                    if k not in consumed and k not in GENERIC_WORDS and k not in INTENT_NOISE
                    and not re.search(r"\b" + re.escape(k), used_text, re.I)]
        if leftover:
            parts.append(f"The scene also features {grammar.join_list(leftover)}")
        elif adjs or verbs or nouns:
            parts.append("The overall mood is")
        if adjs:
            if parts and parts[-1] == "The overall mood is":
                parts[-1] = f"The overall mood is {grammar.join_list(adjs)}"
            else:
                parts.append(f"with {grammar.with_article(grammar.join_list(adjs))} atmosphere")
        elif parts and parts[-1] == "The overall mood is":
            parts[-1] = "The scene includes"
        extras = verbs[:2] + [f"{n} in view" for n in nouns[:1]]
        if parts and parts[-1] == "The scene includes":
            if not extras:
                parts.pop()
            else:
                parts[-1] = f"The scene includes {grammar.join_list(extras)}"
                extras = []
        if extras:
            prefix = "with " if parts and parts[-1].startswith("The overall mood is") else ""
            parts.append(prefix + grammar.join_list(extras))
        if not parts:
            return ""
        return ", ".join(parts) + "."

    # ---------------------------------------------------------------- render
    def _condition_ok(self, cond: Optional[str]) -> bool:
        if not cond:
            return True
        env = {"female": self.cast["female"], "male": self.cast["male"],
               "video": self.t.type == "video", "image": self.t.type == "image",
               "has_input": bool(self.english)}
        try:
            return bool(eval(cond, {"__builtins__": {}}, env))  # simple expressions only
        except Exception:
            self.warnings.append(f"조건식 오류: {cond}")
            return False

    def _resolve(self, text: str, depth: int = 0) -> Optional[str]:
        """Resolve {a|b} variants and {slot} references. None = a required slot was empty."""
        if depth > 6:
            return text
        text = _VARIANT_RE.sub(lambda m: self.rng.choice(m.group(1).split("|")), text)
        missing = False

        def repl(m: re.Match) -> str:
            nonlocal missing
            key, num, optional = m.group(1), m.group(2), m.group(3)
            if key in self.computed:
                value = self.computed[key]
                if "{" in value:
                    value = self._resolve(value, depth + 1) or ""
            elif key == "framing":
                value = self.framing_phrase
            elif key == "part_detail":
                value = self._part_detail()
            elif key == "attire":
                value = self._outfit()
                value = self._resolve(value, depth + 1) or "" if "{" in value else value
            else:
                n = int(num) if num else 1
                vals = self.pick(key, n)
                vals = [v for v in (self._resolve(v, depth + 1) for v in vals) if v]
                value = grammar.join_list(vals)
                if value:
                    prev = self.picks.get(key)
                    self.picks[key] = value if not prev else prev if prev == value else f"{prev} / {value}"
            if not value and not optional:
                missing = True
            return value

        out = _SLOT_RE.sub(repl, text)
        return None if missing else out

    def render(self) -> str:
        sentences: List[str] = []
        for s in self.t.sentences:
            if not self._condition_ok(s.get("when")):
                continue
            prob = s.get("probability")
            if prob is not None and self.rng.random() >= float(prob):
                continue
            text = s.get("text") or ""
            if "options" in s:  # alternative phrasings
                text = self.rng.choice(s["options"])
            out = self._resolve(text)
            if out is None or not out.strip():
                continue
            sentences.append(grammar.ensure_sentence(grammar.capitalize_first(out.strip())))
        joined = " ".join(sentences).lower()
        if self.framing_phrase and self.framing_phrase.lower() not in joined:
            # 구도는 앞쪽에 (주제 문장 바로 뒤) — 템플릿에 {framing} 자리가 없을 때
            sentences.insert(min(1, len(sentences)),
                             grammar.ensure_sentence(grammar.capitalize_first(self.framing_phrase)))
        if self.undress_text:
            undress = self._resolve(self.undress_text) or ""
            if undress and undress.lower() not in joined:
                sent = self._resolve("{She} {is} " + undress) or ""
                outfit = (self.cache.get("attire") or [""])[0]
                at = next((i + 1 for i, s in enumerate(sentences) if outfit and outfit[:25].lower() in s.lower()),
                          min(2, len(sentences)))
                sentences.insert(at, grammar.ensure_sentence(grammar.capitalize_first(sent)))
        uses_details = any("{user_details}" in (s.get("text", "") + " ".join(s.get("options", [])))
                           for s in self.t.sentences)
        if not uses_details and self.computed["user_details"]:
            details = self._details_sentence(" ".join(sentences))
            if details:
                sentences.append(self._resolve(details) or "")
        return grammar.tidy(" ".join(sentences))
