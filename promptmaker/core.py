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
from typing import Any, Dict, List, Optional, Protocol

from . import grammar, safety
from .data_loader import (PACKAGE_DATA_DIR, RATINGS, Element, Item, Template, load_elements,
                          load_json, load_templates, load_text_files)
from .lexicon import LexiconResult, analyze
from .translator import DictionaryTranslator, GoogleTranslator, NoTranslator, Translator, is_english

MODES = ("manual", "random", "random_by_type", "random_by_element")
PROMPT_TYPES = ("image", "video")

_SLOT_RE = re.compile(r"\{([a-zA-Z_]\w*)(?::(\d+))?(\?)?\}")
_VARIANT_RE = re.compile(r"\{([^{}]*\|[^{}]*)\}")


class LLMRefiner(Protocol):
    """Optional future hook: polish the rule-based prompt with an LLM."""

    def refine(self, prompt: str, system_prompt: str, examples: str) -> str: ...


@dataclass
class GenerateOptions:
    prompt_type: str = "image"                 # "image" | "video"
    template: Optional[str] = None             # template id, "random" or None
    mode: str = "random_by_type"               # see MODES
    selections: Dict[str, str] = field(default_factory=dict)  # element key -> fixed value
    random_elements: List[str] = field(default_factory=list)  # for mode "random_by_element"
    content_rating: str = "sfw"                # "sfw" | "suggestive" | "explicit"
    seed: Optional[int] = None
    translate: bool = True
    use_related_words: bool = True
    model_profile: Optional[str] = None        # name of a file in data/model_examples
    use_llm: bool = False                      # only used when an LLMRefiner is set

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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.prompt


class PromptMaker:
    """Rule/template based natural-language prompt generator."""

    def __init__(self, data_dir: Optional[str] = None, user_data_dir: Optional[str] = None,
                 translator: Optional[Translator] = None, llm: Optional[LLMRefiner] = None):
        self.data_dirs = [data_dir or PACKAGE_DATA_DIR]
        user_dir = user_data_dir or os.environ.get("PROMPTMAKER_USER_DATA")
        if user_dir and os.path.isdir(user_dir):
            self.data_dirs.append(user_dir)
        self.llm = llm
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

    def list_templates(self, prompt_type: Optional[str] = None) -> List[Template]:
        return [t for t in self.templates.values() if prompt_type in (None, t.type)]

    def list_elements(self) -> Dict[str, Element]:
        return self.elements

    def info(self) -> Dict[str, Any]:
        """Metadata used by the HTML page / ComfyUI node."""
        return {
            "prompt_types": list(PROMPT_TYPES),
            "modes": list(MODES),
            "ratings": RATINGS,
            "templates": [
                {"id": t.id, "name": t.name, "type": t.type, "description": t.description,
                 "rating": t.rating,
                 "elements": [k for k in t.element_keys() if k in self.elements]}
                for t in self.templates.values()
            ],
            "elements": {
                k: {"label": e.label, "description": e.description,
                    "items": [{"text": i.text, "rating": i.rating} for i in e.items]}
                for k, e in sorted(self.elements.items())
            },
            "model_profiles": sorted(self.model_examples),
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
        if options.content_rating not in RATINGS:
            raise ValueError(f"알 수 없는 content_rating: {options.content_rating}")

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
            english = self.translator.translate(user_input)
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

        # 5) template choice
        template = self._choose_template(options, rng, lex)

        # 6) fill template
        literal = translated is False or getattr(self.translator, "last_backend", "") != "fallback"
        filler = _Filler(self, template, options, rng, lex, english, warnings, literal)
        prompt = filler.render()

        system_prompt = self.system_prompts.get("system_prompt", "")
        profile = options.model_profile or ""
        model_example = self.model_examples.get(profile, "") if profile else ""
        if options.use_llm and self.llm is not None:
            try:
                prompt = self.llm.refine(prompt, system_prompt, model_example) or prompt
            except Exception as exc:
                warnings.append(f"LLM 보정 실패: {exc}")

        return PromptResult(
            prompt=prompt, prompt_type=template.type, template_id=template.id,
            template_name=template.name, seed=seed, original_input=user_input,
            english_input=english, translated=translated, picks=filler.picks,
            lexicon=lex.to_dict(), warnings=warnings, system_prompt=system_prompt,
            model_example=model_example,
        )

    def generate_batch(self, user_input: str = "", count: int = 4,
                       options: Optional[GenerateOptions] = None, **kwargs) -> List[PromptResult]:
        options = options or GenerateOptions.from_dict(kwargs)
        base = options.seed if options.seed is not None and options.seed >= 0 \
            else random.randrange(0, 2**31 - 1 - count)
        return [self.generate(user_input, options, seed=base + i) for i in range(count)]

    # -------------------------------------------------------------- helpers
    def _choose_template(self, options: GenerateOptions, rng: random.Random,
                         lex: LexiconResult) -> Template:
        limit = RATINGS.index(options.content_rating)
        if options.mode == "random":
            pool = list(self.templates.values())
        else:
            pool = self.list_templates(options.prompt_type)
        pool = [t for t in pool if RATINGS.index(t.rating if t.rating in RATINGS else "sfw") <= limit]
        if not pool:
            raise ValueError("선택 가능한 템플릿이 없습니다. prompt_type / content_rating 을 확인하세요.")
        tid = options.template
        if options.mode in ("manual", "random_by_element"):
            if tid and tid != "random":
                if tid not in self.templates:
                    raise ValueError(f"템플릿 '{tid}' 을(를) 찾을 수 없습니다.")
                return self.templates[tid]
            if not tid and options.mode == "manual":
                return pool[0]
        return rng.choice(pool)


class _Filler:
    """Fills one template. Kept separate so that state (picks, rng) is per call."""

    def __init__(self, maker: PromptMaker, template: Template, options: GenerateOptions,
                 rng: random.Random, lex: LexiconResult, english: str, warnings: List[str],
                 literal_input: bool = True):
        self.literal_input = literal_input
        self.maker = maker
        self.t = template
        self.o = options
        self.rng = rng
        self.lex = lex
        self.english = english
        self.warnings = warnings
        self.picks: Dict[str, str] = {}
        self.cast = self._choose_casting()
        self.computed = self._computed_slots()

    # ---------------------------------------------------------- element picks
    def _is_random(self, key: str) -> bool:
        if self.o.mode in ("random", "random_by_type"):
            return True
        if self.o.mode == "random_by_element":
            return key in self.o.random_elements or "*" in self.o.random_elements
        return False

    def _candidates(self, key: str) -> List[Item]:
        el = self.maker.elements.get(key)
        items = list(el.items_for(self.o.content_rating)) if el else []
        known = {i.text.lower() for i in items}
        for hint in self.lex.element_hints.get(key, []):
            if hint.lower() not in known and safety.is_safe_item(hint):
                items.append(Item(text=hint, weight=1.0))
        return items

    def _default_for(self, key: str, items: List[Item]) -> Optional[str]:
        if key in self.t.defaults:
            return self.t.defaults[key]
        el = self.maker.elements.get(key)
        if el and el.default:
            return el.default
        return items[0].text if items else None

    def pick(self, key: str, n: int = 1) -> List[str]:
        if key in self.o.selections:
            value = str(self.o.selections[key])
            values = [v.strip() for v in value.split("|") if v.strip()] if "|" in value else [value]
            return values[:max(n, 1)] if n > 1 else [values[0]]
        items = self._candidates(key)
        if not items:
            return []
        if not self._is_random(key):
            default = self._default_for(key, items)
            if n == 1 or default is None:
                return [default] if default else []
        boosts = self.lex.boost_for(key)
        pool = list(items)
        chosen: List[str] = []
        for _ in range(min(n, len(pool))):
            weights = [i.weight * boosts.get(i.text.lower(), 1.0) for i in pool]
            idx = self.rng.choices(range(len(pool)), weights=weights, k=1)[0]
            chosen.append(pool.pop(idx).text)
        return chosen

    def _choose_casting(self) -> Dict[str, Any]:
        el = self.maker.elements.get("casting")
        f = self.t.casting_filter
        items = []
        for i in (el.items_for(self.o.content_rating) if el else []):
            fe, ma = int(i.meta.get("female", 1)), int(i.meta.get("male", 0))
            if fe < f.get("female_min", 0) or fe > f.get("female_max", 99):
                continue
            if ma < f.get("male_min", 0) or ma > f.get("male_max", 99):
                continue
            items.append(i)
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
        self.picks["age"] = str(age)
        beauty = self._one("beauty")
        nationality = self._one("nationality")
        male_look = self._one("male_look") if ma else ""

        female_core = " ".join(x for x in [beauty, f"{age}-year-old", nationality] if x)
        if fe == 1:
            female_subject = grammar.with_article(f"{female_core} woman")
        elif fe > 1:
            female_subject = f"{grammar.number_word(fe)} {female_core} women"
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

    def _details_sentence(self) -> str:
        """Sentence carrying the user's own request + probabilistic related words."""
        if not self.english:
            return ""
        adjs = self.lex.related.get("adjectives", [])
        nouns = self.lex.related.get("nouns", [])
        verbs = self.lex.related.get("verbs", [])
        text = self.english.strip().rstrip(".!?")
        text = re.sub(r"^(please\s+)?(make|create|generate|draw)\s+(me\s+)?(an?\s+)?"
                      r"(image|picture|photo|video|prompt)\s+(of\s+)?", "", text, flags=re.I)
        parts: List[str] = []
        if self.literal_input and text:
            if len(text.split()) <= 3 and not re.match(r"(?i)^(an?|the)\s", text):
                parts.append(f"The theme is {text}")
            else:
                parts.append(f"The scene captures {text}")
        else:
            from .lexicon import GENERIC_WORDS
            kws = [k for k in self.lex.keywords if k not in GENERIC_WORDS]
            if kws:
                parts.append(f"The scene emphasizes {grammar.join_list(kws)}")
        if adjs:
            parts.append(f"with {grammar.with_article(grammar.join_list(adjs))} atmosphere")
        extras = verbs[:2] + [f"{n} in view" for n in nouns[:1]]
        if extras:
            parts.append(grammar.join_list(extras))
        if not parts:
            return ""
        return ", ".join(parts) + "."

    # ---------------------------------------------------------------- render
    def _condition_ok(self, cond: Optional[str]) -> bool:
        if not cond:
            return True
        env = {"female": self.cast["female"], "male": self.cast["male"],
               "video": self.t.type == "video", "image": self.t.type == "image",
               "rating": RATINGS.index(self.o.content_rating),
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
            min_rating = s.get("rating")
            if min_rating and RATINGS.index(min_rating) > RATINGS.index(self.o.content_rating):
                continue
            text = s.get("text") or ""
            if "options" in s:  # alternative phrasings
                text = self.rng.choice(s["options"])
            out = self._resolve(text)
            if out is None or not out.strip():
                continue
            sentences.append(grammar.ensure_sentence(grammar.capitalize_first(out.strip())))
        uses_details = any("{user_details}" in (s.get("text", "") + " ".join(s.get("options", [])))
                           for s in self.t.sentences)
        if not uses_details and self.computed["user_details"]:
            sentences.append(self._resolve(self.computed["user_details"]) or "")
        return grammar.tidy(" ".join(sentences))
