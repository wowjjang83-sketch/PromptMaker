"""[0.18.0] 프롬프트 커스터마이징 (캐릭터 커스터마이징 화면용).

게임에서 캐릭터를 꾸미듯 단계별로 요소를 골라 영문 자연어 프롬프트를 만듭니다.

    피사체 1 · 피사체 2  →  장소/배경  →  조명 세팅  →  앵글(카메라)

각 카테고리는 ``auto``(요소 txt 에서 자동 선택) · ``pick``(고른 카드 사용) · ``off``(넣지 않음) 중 하나입니다.
자동 선택은 PromptMaker 요소 txt 의 가중치와 장면 태그(relations.json)로 고르므로, 이미 고른 장소·의상과
어울리는 항목이 더 잘 뽑힙니다. 국적·여성 나이 규칙(restrictions)도 그대로 지킵니다.

사용::

    from promptmaker import PromptMaker
    from promptmaker.customize import schema, build

    pm = PromptMaker()
    info = schema(pm)                         # 화면에 그릴 단계 · 카테고리 · 카드(요소 항목)
    res = build(pm, {
        "subjects": [{"enabled": True, "gender": "female",
                      "sel": {"hair": {"mode": "pick", "values": ["long wavy chestnut hair"]}}}],
        "scene": {"location": {"mode": "auto"}, "view": {"direction": "front_3q", "height": "high"}},
    })
    print(res["prompt"])

spec 형식 (모든 키는 생략 가능, 생략 = auto)::

    {"subjects": [{"enabled": bool, "gender": "female"|"male"|"auto",
                   "sel": {카테고리: {"mode": "auto"|"pick"|"off", "values": [텍스트, ...]}}}, ...최대 2],
     "scene": {카테고리: {...}, "view": {"direction", "height", "dutch", "facing"} | null},
     "model_profile": "krea2" | ...}
"""
from __future__ import annotations

import random
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import grammar, safety
from . import view as view_mod
from .data_loader import Item
from .relations import SceneContext

MODES = ("auto", "pick", "off")
GENDERS = ("female", "male")
MAX_SUBJECTS = 2

# 여성 의상 요소 (카드 목록 · 자동 선택 대상). 남성은 male_attire 하나
FEMALE_GARMENTS: Tuple[str, ...] = ("top", "bottom", "dress", "outerwear", "swimwear", "lingerie", "sleepwear",
                                    "sportswear", "traditional")
GARMENT_LABELS: Dict[str, str] = {
    "top": "상의", "bottom": "하의", "dress": "원피스·드레스", "outerwear": "아우터", "swimwear": "수영복",
    "lingerie": "란제리", "sleepwear": "잠옷", "sportswear": "운동복", "traditional": "전통의상",
    "male_attire": "남성 의상",
}
# outfit_type.txt 의 값 → 그 구성에 쓰는 요소
OUTFIT_TYPES: Dict[str, Tuple[str, ...]] = {
    "top_bottom": ("top", "bottom"), "dress": ("dress",), "swimwear": ("swimwear",), "sleepwear": ("sleepwear",),
    "traditional": ("traditional",), "sportswear": ("sportswear",), "lingerie": ("lingerie",),
}
# 의상 구성에 붙는 장면 태그 (attire_state 의 !swim · +top_bottom 같은 조건에 씀)
OUTFIT_TAGS: Dict[str, Tuple[str, ...]] = {
    "top": ("top_bottom",), "bottom": ("top_bottom",), "dress": ("dress", "hem"), "swimwear": ("swim",),
    "sleepwear": ("sleep",), "sportswear": ("sport",), "lingerie": ("lingerie",), "traditional": ("traditional",),
}
OUTERWEAR_RATE = 0.2
MATERIAL_RE = re.compile(r"\b(silk|satin|denim|leather|lace|knit|linen|cotton|velvet|chiffon|mesh|wool|tweed|latex|"
                         r"sheer|fur|suede|sequin\w*|corduroy|nylon|spandex|lycra)\b", re.I)
MALE_AGES: Tuple[str, ...] = ("23", "25", "27", "30", "33", "35")

# ------------------------------------------------------------------ 화면 구성
# kind: element(요소 txt 카드) | choice(고정 선택지) | garment(여러 요소 묶음, 여러 장 선택) | pose | view(카메라 다이얼)
STEPS: List[Dict[str, Any]] = [
    {"id": "subject", "label": "피사체", "icon": "🧑", "per_subject": True, "groups": [
        {"id": "basic", "label": "기본", "categories": ["gender", "nationality", "age", "beauty"]},
        {"id": "hair", "label": "헤어", "categories": ["hair", "hair_length", "makeup"]},
        {"id": "outfit", "label": "의상", "categories": ["outfit", "outfit_design", "outfit_material", "attire_state"]},
        {"id": "pose", "label": "포즈", "categories": ["pose"]},
        {"id": "action", "label": "동작", "categories": ["action", "expression"]},
    ]},
    {"id": "place", "label": "장소/배경", "icon": "🏙", "groups": [
        {"id": "place", "label": "장소", "categories": ["location", "time_weather"]},
    ]},
    {"id": "light", "label": "조명 세팅", "icon": "💡", "groups": [
        {"id": "light", "label": "조명", "categories": ["lighting", "lighting_effect", "color_grade", "photo_style"]},
    ]},
    {"id": "camera", "label": "앵글", "icon": "🎥", "groups": [
        {"id": "camera", "label": "카메라", "categories": ["view", "framing", "camera"]},
    ]},
]

CATEGORIES: Dict[str, Dict[str, Any]] = {
    # 피사체
    "gender": {"label": "성별", "kind": "choice", "scope": "subject",
               "choices": [{"value": "female", "label": "여성"}, {"value": "male", "label": "남성"}]},
    "nationality": {"label": "국적", "kind": "element", "scope": "subject", "elements": ["nationality"]},
    "age": {"label": "나이", "kind": "element", "scope": "subject", "elements": ["age"]},
    "beauty": {"label": "외모 수준", "kind": "element", "scope": "subject", "elements": ["beauty"],
               "male_elements": ["male_look"]},
    "hair": {"label": "헤어스타일", "kind": "element", "scope": "subject", "elements": ["hair"],
             "male_elements": ["male_hair"]},
    "hair_length": {"label": "머리 길이", "kind": "element", "scope": "subject", "elements": ["hair_length"]},
    "makeup": {"label": "메이크업", "kind": "element", "scope": "subject", "elements": ["makeup"], "female_only": True,
               "auto_rate": 0.7},
    "outfit": {"label": "의상", "kind": "garment", "scope": "subject", "elements": list(FEMALE_GARMENTS),
               "male_elements": ["male_attire"], "multi": True},
    "outfit_design": {"label": "디자인", "kind": "element", "scope": "subject", "elements": ["outfit_design"],
                      "auto_rate": 0.25, "female_only_auto": True},
    "outfit_material": {"label": "소재", "kind": "element", "scope": "subject", "elements": ["outfit_material"],
                        "auto_rate": 0.25, "female_only_auto": True},
    "attire_state": {"label": "착용 상태", "kind": "element", "scope": "subject", "elements": ["attire_state"],
                     "auto_rate": 0.5, "female_only_auto": True},
    "pose": {"label": "포즈", "kind": "pose", "scope": "subject", "elements": ["pose"]},
    "action": {"label": "동작·제스처", "kind": "element", "scope": "subject", "elements": ["action"],
               "auto_rate": 0.7},
    "expression": {"label": "표정", "kind": "element", "scope": "subject", "elements": ["expression"]},
    # 장소 / 조명 / 카메라
    "location": {"label": "장소", "kind": "element", "scope": "scene", "elements": ["location"]},
    "time_weather": {"label": "시간·날씨", "kind": "element", "scope": "scene", "elements": ["time_weather"],
                     "auto_rate": 0.6},
    "lighting": {"label": "조명", "kind": "element", "scope": "scene", "elements": ["lighting"]},
    "lighting_effect": {"label": "조명 효과", "kind": "element", "scope": "scene", "elements": ["lighting_effect"],
                        "auto_rate": 0.6},
    "color_grade": {"label": "색감", "kind": "element", "scope": "scene", "elements": ["color_grade"],
                    "auto_rate": 0.6},
    "photo_style": {"label": "사진 스타일", "kind": "element", "scope": "scene", "elements": ["photo_style"]},
    "view": {"label": "카메라 앵글", "kind": "view", "scope": "scene"},
    "framing": {"label": "샷 구도", "kind": "element", "scope": "scene", "elements": ["framing"]},
    "camera": {"label": "렌즈·촬영", "kind": "element", "scope": "scene", "elements": ["camera"], "default_mode": "off"},
}

# 기본 사진 스타일 (photo_style 을 '사용 안 함' 으로 했을 때)
DEFAULT_PHOTO = "photorealistic photograph"


def _pronouns(gender: str, plural: bool = False) -> Dict[str, str]:
    if plural:
        return {"she": "they", "She": "They", "her": "their", "her_obj": "them", "his": "their", "him": "them",
                "is": "are", "has": "have", "wears": "wear", "s": "", "is_m": "are"}
    if gender == "male":
        return {"she": "he", "She": "He", "her": "his", "her_obj": "him", "his": "his", "him": "him",
                "is": "is", "has": "has", "wears": "wears", "s": "s", "is_m": "is"}
    return {"she": "she", "She": "She", "her": "her", "her_obj": "her", "his": "his", "him": "him",
            "is": "is", "has": "has", "wears": "wears", "s": "s", "is_m": "is"}


_SLOT_RE = re.compile(r"\{([a-zA-Z_]\w*)\}")
_VARIANT_RE = re.compile(r"\{([^{}]*\|[^{}]*)\}")


def resolve_text(text: str, gender: str = "female", plural: bool = False, rng: Optional[random.Random] = None) -> str:
    """요소 항목의 {her} · {a|b} 같은 자리를 채움 (모르는 자리는 지움)."""
    rng = rng or random.Random(text)
    text = _VARIANT_RE.sub(lambda m: rng.choice(m.group(1).split("|")), str(text or ""))
    pro = _pronouns(gender, plural)
    out = _SLOT_RE.sub(lambda m: pro.get(m.group(1), pro.get(m.group(1).lower(), "")), text)
    return " ".join(out.split()).strip(" ,")


def _elements_for(cat: Dict[str, Any], gender: str) -> List[str]:
    if gender == "male" and cat.get("male_elements"):
        return list(cat["male_elements"])
    return list(cat.get("elements") or [])


def category_items(maker, key: str, gender: str = "female") -> List[Dict[str, Any]]:
    """카테고리의 카드(요소 항목) 목록: [{"element", "text", "label", "weight", "tags"}]."""
    cat = CATEGORIES.get(key)
    if not cat:
        return []
    if cat["kind"] == "choice":
        return [{"element": key, "text": c["value"], "label": c["label"], "weight": 1.0, "tags": []}
                for c in cat["choices"]]
    if key == "age" and gender == "male":
        return [{"element": "age", "text": a, "label": f"{a}세", "weight": 1.0, "tags": []} for a in MALE_AGES]
    out: List[Dict[str, Any]] = []
    rules = maker.restrictions
    seen = set()
    for ekey in _elements_for(cat, gender):
        el = maker.elements.get(ekey)
        for it in (el.items if el else []):
            if ekey in ("nationality", "age") and rules.enabled and not rules.allows(ekey, it.text):
                continue
            if ekey == "age":
                m = re.match(r"\s*(\d+)", it.text)
                if not m or int(m.group(1)) < safety.MIN_AGE:
                    continue
            sig = (ekey, it.text.lower())
            if sig in seen:
                continue
            seen.add(sig)
            label = resolve_text(it.text, gender)
            if ekey == "age":
                label = f"{label}세"
            out.append({"element": ekey, "text": it.text, "label": label, "weight": it.weight,
                        "tags": sorted(it.tags), "group": GARMENT_LABELS.get(ekey, "") if cat["kind"] == "garment" else ""})
    if key == "nationality" and not out and rules.nat_on:
        out = [{"element": "nationality", "text": n, "label": n, "weight": 1.0, "tags": []} for n in rules.nationalities]
    return out


def schema(maker, with_items: bool = True) -> Dict[str, Any]:
    """화면에 그릴 단계 · 카테고리 · 카드. 카드 목록은 성별(female/male)마다 따로 줍니다."""
    cats: Dict[str, Any] = {}
    for key, cat in CATEGORIES.items():
        info = {k: v for k, v in cat.items() if k not in ("elements", "male_elements")}
        info["key"] = key
        info["elements"] = list(cat.get("elements") or [])
        info["male_elements"] = list(cat.get("male_elements") or [])
        info["default_mode"] = cat.get("default_mode", "auto")
        if with_items and cat["kind"] != "view":
            info["items"] = {"female": category_items(maker, key, "female")}
            if cat["scope"] == "subject":
                info["items"]["male"] = category_items(maker, key, "male")
        cats[key] = info
    return {
        "steps": STEPS, "categories": cats, "modes": list(MODES), "max_subjects": MAX_SUBJECTS,
        "garment_labels": GARMENT_LABELS,
        "view": {"directions": list(view_mod.DIRECTIONS), "heights": list(view_mod.HEIGHTS),
                 "labels": view_mod.LABELS_KO},
        "restrictions": maker.restrictions.to_dict() if hasattr(maker.restrictions, "to_dict") else {},
    }


# ------------------------------------------------------------------ 프롬프트 만들기
def _sel(sel: Dict[str, Any], key: str) -> Tuple[str, List[str]]:
    raw = (sel or {}).get(key)
    default = CATEGORIES.get(key, {}).get("default_mode", "auto")
    if raw is None:
        return default, []
    if isinstance(raw, str):
        return ("pick", [raw]) if raw.strip() else (default, [])
    mode = str(raw.get("mode") or default)
    vals = raw.get("values")
    if vals is None and raw.get("value"):
        vals = [raw.get("value")]
    vals = [str(v).strip() for v in (vals or []) if str(v or "").strip()]
    if mode not in MODES:
        mode = "auto"
    if mode == "pick" and not vals:
        mode = "auto"
    return mode, vals


class _Builder:
    def __init__(self, maker, spec: Dict[str, Any], seed: Optional[int] = None):
        self.maker = maker
        self.spec = spec or {}
        self.seed = seed if seed is not None else random.randrange(1, 2 ** 31)
        self.rng = random.Random(self.seed)
        self.ctx = SceneContext(maker.relations)
        self.picks: Dict[str, str] = {}
        self.warnings: List[str] = []
        self.used: set = set()
        self.plural = False
        self.view_dir = ""

    # 요소 항목 찾기 / 장면 태그 반영
    def _item(self, ekey: str, text: str) -> Optional[Item]:
        el = self.maker.elements.get(ekey)
        low = text.lower()
        return next((i for i in (el.items if el else []) if i.text.lower() == low), None)

    def _note(self, ekey: str, text: str, extra: Iterable[str] = ()) -> None:
        it = self._item(ekey, text)
        tags = set(extra)
        if it:
            tags |= it.tags
        tags |= self.maker.relations.tags_in_text(text)
        if tags:
            self.ctx.add(tags, provider=ekey in self.maker.relations.providers)
        self.used.add(text.lower())

    def _auto(self, ekeys: List[str], gender: str = "female") -> Optional[Tuple[str, str]]:
        items: List[Tuple[str, Item]] = []
        for ek in ekeys:
            el = self.maker.elements.get(ek)
            items += [(ek, i) for i in (el.items if el else []) if i.text.lower() not in self.used]
        if not items:
            return None
        scored = self.ctx.weigh([i for _, i in items])
        weights = [max(w, 0.0) for _, w, _ in scored]
        if not any(weights):  # 조건이 다 안 맞으면 조건 없는 항목 중에서
            weights = [0.0 if (i.requires or i.excludes) else max(i.weight, 0.01) for _, i in items]
        if not any(weights):
            weights = [1.0] * len(items)
        ek, it = items[self.rng.choices(range(len(items)), weights=weights, k=1)[0]]
        return ek, it.text

    def _one(self, sel: Dict[str, Any], key: str, gender: str = "female") -> Optional[str]:
        """카테고리 하나 → 원문 텍스트 (None = 넣지 않음)."""
        mode, vals = _sel(sel, key)
        if mode == "off":
            return None
        cat = CATEGORIES[key]
        ekeys = _elements_for(cat, gender)
        if mode == "pick":
            text = self.rng.choice(vals) if len(vals) > 1 else vals[0]
            self._note(ekeys[0] if ekeys else key, text)
            return text
        rate = float(cat.get("auto_rate", 1.0))
        if cat.get("female_only_auto") and gender == "male":
            rate = 0.0
        if rate < 1.0 and self.rng.random() >= rate:
            return None  # 자동: 가끔만 넣는 카테고리 (소재 · 디자인 · 조명 효과 ...)
        got = self._auto(ekeys, gender)
        if not got:
            return None
        self._note(*got)
        return got[1]

    # ---------------------------------------------------------------- 피사체
    def _age(self, sel: Dict[str, Any], gender: str) -> str:
        mode, vals = _sel(sel, "age")
        rules = self.maker.restrictions
        if mode == "off":
            return ""
        if mode == "pick":
            raw = self.rng.choice(vals)
        elif gender == "male":
            raw = self.rng.choice(MALE_AGES)
        else:
            raw = self._auto(["age"]) or ("age", "21")
            raw = raw[1]
        m = re.match(r"\s*(\d+)", raw)
        age = int(m.group(1)) if m else 21
        age = max(age, safety.MIN_AGE)
        if gender == "female" and rules.enabled:
            age = rules.clamp_age(age)
        return str(age)

    def _nationality(self, sel: Dict[str, Any], gender: str) -> str:
        mode, vals = _sel(sel, "nationality")
        if mode == "off":
            return ""
        rules = self.maker.restrictions
        if mode == "pick":
            val = self.rng.choice(vals)
        else:
            got = self._auto(["nationality"])
            val = got[1] if got else ""
        return rules.nationality(val, self.rng) if rules.nat_on else val

    def _outfit(self, sel: Dict[str, Any], gender: str) -> Tuple[str, List[str]]:
        """의상 → (문구, 쓰인 요소 키)."""
        mode, vals = _sel(sel, "outfit")
        if mode == "off":
            return "", []
        if mode == "pick":
            ekeys = []
            for v in vals:
                ek = next((k for k in _elements_for(CATEGORIES["outfit"], gender) if self._item(k, v)), "")
                ekeys.append(ek)
                self._note(ek or "attire", v, OUTFIT_TAGS.get(ek, ()))
            return grammar.join_list(vals), [k for k in ekeys if k]
        if gender == "male":
            got = self._auto(["male_attire"])
            if not got:
                return "", []
            self._note(*got)
            return got[1], ["male_attire"]
        # 구성 고르기 (outfit_type.txt) → 요소별 항목
        el = self.maker.elements.get("outfit_type")
        types = [i for i in (el.items if el else []) if i.text in OUTFIT_TYPES]
        if types:
            scored = self.ctx.weigh(types)
            weights = [max(w, 0.0) for _, w, _ in scored]
            if not any(weights):
                weights = [max(i.weight, 0.01) for i in types]
            otype = types[self.rng.choices(range(len(types)), weights=weights, k=1)[0]].text
        else:
            otype = "top_bottom"
        parts, used = [], []
        for ek in OUTFIT_TYPES[otype]:
            self.ctx.add(OUTFIT_TAGS.get(ek, ()))
            got = self._auto([ek])
            if got:
                self._note(*got)
                parts.append(got[1])
                used.append(ek)
        if otype in ("top_bottom", "dress") and self.rng.random() < OUTERWEAR_RATE:
            got = self._auto(["outerwear"])
            if got:
                self._note(*got)
                parts.append(got[1])
                used.append("outerwear")
        return grammar.join_list(parts), used

    def _pose(self, sel: Dict[str, Any], gender: str) -> Optional[str]:
        return self._one(sel, "pose", gender)

    def subject(self, sub: Dict[str, Any], idx: int, plural: bool) -> Dict[str, Any]:
        sel = dict(sub.get("sel") or {})
        gender = str(sub.get("gender") or "").lower()
        gmode, gvals = _sel(sel, "gender")
        if gmode == "pick" and gvals and gvals[0] in GENDERS:
            gender = gvals[0]
        if gender not in GENDERS:
            gender = "female" if idx == 0 else self.rng.choice(GENDERS)
        d: Dict[str, Any] = {"gender": gender}
        d["nationality"] = self._nationality(sel, gender)
        d["age"] = self._age(sel, gender)
        d["beauty"] = self._one(sel, "beauty", gender) or ""
        d["hair"] = self._one(sel, "hair", gender) or ""
        d["hair_length"] = self._one(sel, "hair_length", gender) or ""
        d["makeup"] = (self._one(sel, "makeup", gender) or "") if gender == "female" else ""
        d["outfit"], d["outfit_keys"] = self._outfit(sel, gender)
        d["outfit_design"] = (self._one(sel, "outfit_design", gender) or "") if d["outfit"] else ""
        has_material = bool(MATERIAL_RE.search(d["outfit"]))
        d["outfit_material"] = ((self._one(sel, "outfit_material", gender) or "")
                                if d["outfit"] and (not has_material or _sel(sel, "outfit_material")[0] == "pick") else "")
        d["attire_state"] = (self._one(sel, "attire_state", gender) or "") if d["outfit"] else ""
        d["pose"] = self._pose(sel, gender) or ""
        d["action"] = self._one(sel, "action", gender) or ""
        emode, _ = _sel(sel, "expression")
        if self.view_dir in view_mod.FACE_HIDDEN and emode == "auto":
            d["expression"] = ""  # 뒷모습: 얼굴이 안 보이므로 표정은 자동으로 넣지 않음
        else:
            d["expression"] = self._one(sel, "expression", gender) or ""
        return d

    # ---------------------------------------------------------------- 문장
    def _r(self, text: str, gender: str = "female", plural: bool = False) -> str:
        return resolve_text(text, gender, plural, self.rng)

    @staticmethod
    def _noun(gender: str) -> str:
        return "man" if gender == "male" else "woman"

    def _subject_phrase(self, d: Dict[str, Any]) -> str:
        g = d["gender"]
        words = [self._r(d["beauty"], g)] if d["beauty"] else []
        if d["age"]:
            words.append(f"{d['age']}-year-old")
        if d["nationality"]:
            words.append(d["nationality"])
        words.append(self._noun(g))
        return grammar.with_article(" ".join(w for w in words if w))

    def _hair_phrase(self, d: Dict[str, Any]) -> str:
        g = d["gender"]
        hair = self._r(d["hair"], g) if d["hair"] else ""
        length = self._r(d["hair_length"], g) if d["hair_length"] else ""
        if length and hair:
            if re.search(r"\b(long|short|shoulder|bob|lob|pixie|waist|chin|cropped|buzz)\w*", hair, re.I):
                return hair  # 헤어스타일에 길이가 이미 있음
            if re.match(r"^(a|an|the)\s", hair, re.I):
                return f"{length} hair styled in {hair}"
            return f"{length} {hair}"
        if length:
            return f"{length} hair"
        return hair

    def _outfit_phrase(self, d: Dict[str, Any]) -> str:
        g = d["gender"]
        out = self._r(d["outfit"], g)
        if not out:
            return ""
        if d["outfit_material"]:
            out += " " + self._r(d["outfit_material"], g)
        if d["outfit_design"]:
            out += " " + self._r(d["outfit_design"], g)
        return out

    def _person_lines(self, d: Dict[str, Any], ref: str) -> List[str]:
        """한 사람의 문장들. ref = 문장 주어 ('She' / 'The woman on the left')."""
        g = d["gender"]
        pro = _pronouns(g)
        lines = []
        hair = self._hair_phrase(d)
        makeup = self._r(d["makeup"], g) if d["makeup"] else ""
        if hair or makeup:
            body = grammar.join_list([p for p in (hair, makeup) if p])
            lines.append(f"{ref} {pro['has']} {body}.")
        outfit = self._outfit_phrase(d)
        if outfit:
            state = self._r(d["attire_state"], g) if d["attire_state"] else ""
            lines.append(f"{ref if not lines else pro['She']} {pro['wears']} {outfit}" + (f", {state}" if state else "") + ".")
        acts = [self._r(t, g) for t in (d["pose"], d["action"]) if t]
        expr = self._r(d["expression"], g) if d["expression"] else ""
        if acts or expr:
            subj = ref if not lines else pro["She"]
            if acts:
                s = f"{subj} {pro['is']} " + ", ".join(acts)
                if expr:
                    s += f", with {expr}"
            else:
                s = f"{subj} {pro['has']} {expr}"
            lines.append(s + ".")
        return lines

    def build(self) -> Dict[str, Any]:
        scene = dict(self.spec.get("scene") or {})
        subs = [s for s in (self.spec.get("subjects") or []) if isinstance(s, dict) and s.get("enabled", True)]
        subs = subs[:MAX_SUBJECTS] or [{"enabled": True, "gender": "female"}]
        plural = len(subs) > 1
        self.plural = plural
        raw_view = scene.get("view") if isinstance(scene.get("view"), dict) else {}
        self.view_dir = str(raw_view.get("direction") or "")

        # 장소 → 조명 → 사람 순으로 정해야 장면 태그가 의상 · 포즈 자동 선택에 반영됨
        location = self._one(scene, "location")
        time_weather = self._one(scene, "time_weather")
        lighting = self._one(scene, "lighting")
        people = [self.subject(s, i, plural) for i, s in enumerate(subs)]
        g0 = people[0]["gender"]
        lighting_effect = self._one(scene, "lighting_effect")
        color_grade = self._one(scene, "color_grade")
        photo_style = self._one(scene, "photo_style")
        framing = self._framing(scene, plural)
        camera = self._one(scene, "camera")
        view = self._view(scene.get("view"))

        lines: List[str] = []
        style = self._r(photo_style, g0) if photo_style else DEFAULT_PHOTO
        if not plural:
            lines.append(f"{grammar.with_article(style)} of {self._subject_phrase(people[0])}.")
            lines += self._person_lines(people[0], _pronouns(g0)["She"])
        else:
            a, b = people
            lines.append(f"{grammar.with_article(style)} of two people: {self._subject_phrase(a)} on the left "
                         f"and {self._subject_phrase(b)} on the right.")
            same = a["gender"] == b["gender"]
            for d, side in ((a, "left"), (b, "right")):
                ref = f"The {self._noun(d['gender'])}" + (f" on the {side}" if same else "")
                lines += self._person_lines(d, ref)
        if location or time_weather:
            place = " ".join(self._r(t, g0, plural) for t in (location, time_weather) if t)
            lines.append(f"The scene takes place {place}.")
        light = [self._r(t, g0, plural) for t in (lighting, lighting_effect) if t]
        if light:
            lines.append(grammar.capitalize_first(", ".join(light)) + ".")
        cam = [self._r(t, g0, plural) for t in (framing, view, camera, color_grade) if t]
        if plural:  # 두 사람: 'they faces' → 'they face', 'their entire body' → 'both of them fully'
            cam = [re.sub(r"\bthey (\w+?)(e?s)\b", lambda m: "they " + (m.group(1) + ("e" if m.group(2) == "es" and not
                          re.search(r"(sh|ch|x|ss)$", m.group(1)) else "")), c.replace("their entire body", "both of them fully"))
                   for c in cam]
        if cam:
            lines.append(grammar.capitalize_first(", ".join(cam)) + ".")
        if self.view_dir in view_mod.BACK_TURNED:  # 뒤 · 뒤 사선: '카메라 쪽으로 걸어오는' → '카메라에서 멀어지는'
            lines = [view_mod.turn_away(ln) for ln in lines]
        prompt = grammar.tidy(" ".join(lines))
        if self.maker.restrictions.enabled:
            prompt = self.maker.restrictions.apply_to_text(prompt, self.rng)

        self.picks = {k: v for k, v in {
            "location": location, "time_weather": time_weather, "lighting": lighting,
            "lighting_effect": lighting_effect, "color_grade": color_grade, "photo_style": photo_style,
            "framing": framing, "camera": camera, "view": view,
        }.items() if v}
        for i, d in enumerate(people, 1):
            for k, v in d.items():
                if v and k != "outfit_keys":
                    self.picks[f"subject{i}.{k}"] = v
        return {"prompt": prompt, "seed": self.seed, "picks": self.picks, "people": people,
                "warnings": self.warnings}

    def _framing(self, scene: Dict[str, Any], plural: bool) -> Optional[str]:
        """샷 구도. 두 사람을 자동으로 고를 때는 얼굴 클로즈업 구도를 빼고 (둘 다 담기게)."""
        mode, _ = _sel(scene, "framing")
        if mode != "auto" or not plural:
            return self._one(scene, "framing")
        el = self.maker.elements.get("framing")
        items = [i for i in (el.items if el else []) if i.tags & {"medium", "cowboy", "full_body"}]
        if not items:
            return self._one(scene, "framing")
        it = self.rng.choices(items, weights=[max(i.weight, 0.01) for i in items], k=1)[0]
        self._note("framing", it.text)
        return it.text

    def _view(self, raw: Any) -> str:
        """카메라 다이얼 값 {direction, height, dutch, facing} → 시점 문구 (view.txt 의 태그 줄 우선)."""
        if not isinstance(raw, dict):
            return ""
        el = self.maker.elements.get("view")
        parts = []
        for axis in ("direction", "height", "dutch", "facing"):
            tag = raw.get(axis)
            if not tag or (axis == "height" and tag == "eye_level" and not raw.get("explicit_eye_level")):
                continue
            if tag not in view_mod.DEFAULT_PHRASES:
                continue
            items = [i for i in (el.items if el else []) if tag in i.tags]
            if self.plural:
                items = []
            text = (self.rng.choices(items, weights=[max(i.weight, 0.01) for i in items], k=1)[0].text
                    if items else view_mod.DEFAULT_PHRASES[tag])
            parts.append(re.sub(r"\b(at|toward|towards) \{her\}(?!\w)", r"\1 {her_obj}", text))
        return ", ".join(parts)


def build(maker, spec: Dict[str, Any], seed: Optional[int] = None,
          model_profile: Optional[str] = None) -> Dict[str, Any]:
    """spec → {"prompt", "base_prompt", "negative_prompt", "seed", "picks", "people", "warnings"}."""
    b = _Builder(maker, spec, seed)
    res = b.build()
    base = res["prompt"]
    profile_id = model_profile or (spec or {}).get("model_profile") or ""
    res["base_prompt"] = base
    res["negative_prompt"] = ""
    res["model_profile"] = ""
    prof = maker.model_profiles.get(profile_id) if profile_id else None
    if prof is not None:
        from .model_profiles import format_prompt
        flat = {k.split(".", 1)[-1]: v for k, v in res["picks"].items() if not k.startswith("subject2.")}
        formatted = format_prompt(base, flat, b.ctx.tags, prof, "image")
        res["prompt"] = formatted.prompt
        res["negative_prompt"] = getattr(formatted, "negative", "") or ""
        res["model_profile"] = profile_id
    return res


def build_batch(maker, spec: Dict[str, Any], count: int = 1, seed: Optional[int] = None,
                model_profile: Optional[str] = None) -> List[Dict[str, Any]]:
    """같은 설정으로 여러 장 (자동 선택 항목은 장마다 새로 고름)."""
    base = seed if seed is not None else random.randrange(1, 2 ** 31)
    return [build(maker, spec, base + i, model_profile) for i in range(max(1, int(count)))]
