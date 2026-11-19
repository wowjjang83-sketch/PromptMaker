"""신체 부위별 클로즈업 — 요청·요소 문구에서 부위를 찾고, 그 부위에 맞는 요소만 남깁니다.

'다리 클로즈업', '입술 접사', '손만 나오게', 'close-up of her legs' 처럼 부위와 클로즈업 말이 붙어 있으면
부위 클로즈업(샷 구도 ``body_part``)으로 봅니다. 그러면

* 구도 문구는 ``elements/body_closeup.txt`` 에서 ``#<부위>`` 태그로 고르고 (태그가 없는 줄은 문구로 부위를 찾음),
* 의상은 그 부위에 보이는 것만 (다리 → 하의·스타킹·신발, 손 → 반지·팔찌, 입술 → 의상 없음),
* 포즈·동작·의상 상태는 그 부위를 말하는 것만 씁니다 ('sitting with legs crossed' ○, 'smiling' ✕).

이미지 분석 결과의 분리 (:func:`route_phrase`): 카메라·구도로 나온 문구 중 부위 클로즈업은
``body_closeup`` 에 ``#legs`` 같은 태그를 붙여, 일반 구도는 ``framing`` 에 ``#medium`` 같은 태그를 붙여 저장합니다.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# 부위 id -> (한국어 이름, 한국어 말, 영어 말(정규식), 관련 단어(정규식), 보이는 의상 요소)
_UPPER = ("top", "dress", "swimwear", "lingerie", "sleepwear", "sportswear", "traditional", "outerwear")
PARTS: Dict[str, Dict[str, object]] = {
    "eyes": {"ko": "눈", "ko_terms": ("눈동자", "눈매", "속눈썹", "눈"),
             "en": r"eyes?|eyelash(?:es)?|lashes",
             "words": r"eyes?|eyelash(?:es)?|lash(?:es)?|brows?|eyebrows?|eyeliner|eyeshadow|mascara|gaze|gazing|"
                      r"glanc\w*|wink\w*|blink\w*|look(?:s|ing)?|stare|staring|tears?",
             "garments": ()},
    "lips": {"ko": "입술", "ko_terms": ("입술", "입술만", "입"),
             "en": r"lips?|mouth",
             "words": r"lips?|mouth|smil\w*|kiss\w*|biting|bites|pout\w*|lipstick|gloss\w*|teeth|tongue|grin\w*",
             "garments": ()},
    "hair": {"ko": "머리카락", "ko_terms": ("머리카락", "헤어"),
             "en": r"hair",
             "words": r"hair|strands?|ponytail|braids?|curls?|bun|bangs|locks",
             "garments": ()},
    "hands": {"ko": "손", "ko_terms": ("손가락", "손톱", "네일", "손"),
              "en": r"hands?|fingers?|nails|fingertips",
              "words": r"hands?|fingers?|fingertips?|nails?|manicure\w*|palms?|wrists?|holding|holds|touch\w*|"
                       r"grip\w*|rings?|bracelets?|watch|bangles?",
              "garments": ("accessory",)},
    "neck": {"ko": "목선", "ko_terms": ("목선", "쇄골", "목덜미", "목"),
             "en": r"neck|necklines?|collarbones?|clavicles?|nape|throat",
             "words": r"neck\w*|collarbones?|clavicles?|nape|throat|chokers?|necklaces?|pendants?|earrings?",
             "garments": _UPPER + ("accessory",)},
    "shoulders": {"ko": "어깨", "ko_terms": ("어깨선", "어깨"),
                  "en": r"shoulders?",
                  "words": r"shoulders?|straps?|off-the-shoulder|collarbones?",
                  "garments": _UPPER + ("accessory",)},
    "back": {"ko": "등", "ko_terms": ("뒤태", "등라인", "등"),
             "en": r"back|spine|shoulder blades?",
             "words": r"(?<!on her )(?<!on their )(?<!on \{her\} )(?<!on \{their\} )(?<!looking )(?<!look )(?<!leaning )back|backless|open-back|spine|"
                      r"shoulder blades?",
             "garments": _UPPER},
    "chest": {"ko": "가슴", "ko_terms": ("가슴골", "데콜테", "가슴"),
              "en": r"chest|cleavage|d[eé]colletage|bust",
              "words": r"chest|cleavage|d[eé]colletage|bust|necklines?|plunging",
              "garments": _UPPER + ("accessory",)},
    "waist": {"ko": "허리", "ko_terms": ("잘록한 허리", "개미허리", "허리라인", "배꼽", "복근", "허리"),
              "en": r"waist|waistline|midriff|navel|belly|abs|stomach",
              "words": r"waist\w*|midriff|navel|belly|abs|stomach|crop\w*|hips?",
              "garments": ("top", "bottom", "dress", "swimwear", "lingerie", "sportswear", "sleepwear")},
    "hips": {"ko": "엉덩이", "ko_terms": ("엉덩이", "힙라인", "골반", "힙"),
             "en": r"hips?|buttocks?|butt|derriere",
             "words": r"hips?|buttocks?|butt|derriere|curves?|waistband|hem|bottoms?",
             "garments": ("bottom", "dress", "swimwear", "lingerie", "sportswear", "sleepwear")},
    "thighs": {"ko": "허벅지", "ko_terms": ("허벅지",),
               "en": r"thighs?",
               "words": r"thighs?|stockings?|garters?|lap|hem|thigh-highs?",
               "garments": ("bottom", "dress", "swimwear", "lingerie", "legwear")},
    "legs": {"ko": "다리", "ko_terms": ("각선미", "다리라인", "다리"),
             "en": r"legs?|calves",
             "words": r"legs?|thighs?|knees?|calves|ankles?|crossed|cross-legged|stockings?|tights|heels|kneel\w*|"
                      r"stretch\w*|walk\w*|stride|striding|sitting",
             "garments": ("bottom", "dress", "swimwear", "legwear", "footwear")},
    "feet": {"ko": "발", "ko_terms": ("발가락", "발목", "맨발", "발"),
             "en": r"feet|foot|toes?|ankles?",
             "words": r"feet|foot|toes?|tiptoes?|ankles?|barefoot|heels|sandals|shoes|soles?|pedicure\w*|anklets?|"
                      r"stepping|standing|walking",
             "garments": ("footwear", "legwear")},
}
# 함께 보여도 되는 가까운 부위 (포즈·동작이 이 부위를 말해도 됨)
NEIGHBORS = {
    "eyes": ("lips", "hair"), "lips": ("eyes", "hands"), "hair": ("eyes", "lips", "neck", "shoulders", "back", "hands"),
    "hands": ("lips", "hair", "neck", "waist", "hips", "thighs", "legs", "chest"), "neck": ("shoulders", "chest", "hair"),
    "shoulders": ("neck", "back", "chest", "hair"), "back": ("shoulders", "waist", "hair", "hips"),
    "chest": ("neck", "shoulders", "waist"), "waist": ("chest", "hips", "hands", "back"),
    "hips": ("waist", "thighs", "hands", "back"), "thighs": ("hips", "legs", "hands"),
    "legs": ("thighs", "feet", "hips", "hands"), "feet": ("legs",),
}
LABELS_KO = {k: f"{v['ko']} 클로즈업" for k, v in PARTS.items()}
DEFAULT_PHRASES = {
    "eyes": "macro close-up of {her} eyes and lashes",
    "lips": "close-up shot of {her} lips, cropped tightly around {her} mouth",
    "hair": "close-up of {her} hair filling the frame",
    "hands": "close-up of {her} hands, framed from the wrists to the fingertips",
    "neck": "close-up of {her} neck and collarbones",
    "shoulders": "close-up of {her} bare shoulders",
    "back": "close-up of {her} back, framed from the shoulder blades to the waist",
    "chest": "close-up of {her} chest and neckline",
    "waist": "close-up of {her} waist and midriff",
    "hips": "close-up of {her} hips",
    "thighs": "close-up of {her} thighs",
    "legs": "close-up of {her} legs, framed from the hips to the ankles",
    "feet": "close-up of {her} feet and ankles",
}
_WORDS = {k: re.compile(r"\b(?:" + str(v["words"]) + r")\b", re.I) for k, v in PARTS.items()}
_EN_PART = {k: str(v["en"]) for k, v in PARTS.items()}
_CLOSE_EN = r"(?:extreme |tight |macro |detail(?:ed)? )?(?:close[- ]?ups?|macro(?: shots?)?|detail shots?)(?: shots?)?"
_KO_CLOSE = r"(?:클로즈업|클로즈 업|접사|확대|근접|디테일|만\s*나오|만\s*보이|위주)"


def _ko_terms():
    out = []
    for k, v in PARTS.items():
        for t in v["ko_terms"]:  # type: ignore[union-attr]
            out.append((t, k))
    return sorted(out, key=lambda x: -len(x[0]))  # 긴 말 먼저 ('발가락' 이 '발' 보다)


_KO_PART_RE = [(re.compile(re.escape(t) + r"\s*(?:을|를|의|이|가|만|쪽)?\s*" + _KO_CLOSE), k) for t, k in _ko_terms()]
_KO_PART_RE_REV = [(re.compile(r"(?:클로즈업|접사|확대)\s*(?:된|한|샷|으로|된\s*채)?\s*(?:의\s*)?" + re.escape(t) + r"(?![가-힣])"), k)
                   for t, k in _ko_terms()]
_EN_PART_RE = [(re.compile(r"\b" + _CLOSE_EN + r"\s+(?:of|on)\s+(?:(?:her|their|the|a|an|his)\s+)?(?:[a-z-]+\s+){0,2}?(?:"
                           + p + r")\b", re.I), k) for k, p in _EN_PART.items()] \
    + [(re.compile(r"\b(?:" + p + r")\s+" + _CLOSE_EN + r"\b", re.I), k) for k, p in _EN_PART.items()]


def detect(original: str = "", english: str = "") -> Tuple[Optional[str], str]:
    """요청에서 부위 클로즈업을 찾습니다. ``(부위 id 또는 None, 찾은 말)``."""
    best: Optional[Tuple[int, str, str]] = None
    for text, rules in ((original, _KO_PART_RE + _KO_PART_RE_REV), (english, _EN_PART_RE)):
        if not text:
            continue
        for rx, part in rules:
            m = rx.search(text)
            if m and (best is None or m.start() < best[0] or (m.start() == best[0] and len(m.group(0)) > len(best[2]))):
                best = (m.start(), part, m.group(0))
        if best:
            break
    return (best[1], best[2]) if best else (None, "")


def part_in_text(text: str) -> Optional[str]:
    """요소 한 줄이 부위 클로즈업을 말하면 그 부위 ('close-up of her legs' → legs)."""
    for rx, part in _EN_PART_RE:
        if rx.search(text or ""):
            return part
    return None


def mentions(text: str, part: str) -> bool:
    """이 문구가 그 부위를 말하는지 (부위 클로즈업에서 쓸 수 있는 포즈·동작인지)."""
    return bool(_WORDS[part].search(text or ""))


def garments(part: str) -> Tuple[str, ...]:
    return tuple(PARTS[part]["garments"])  # type: ignore[arg-type]


def route_phrase(text: str) -> Optional[Tuple[str, str]]:
    """분석 문구가 샷 구도면 ``(저장할 요소, 태그 붙인 문구)``. 아니면 None.

    'close-up of her legs' → ('body_closeup', 'close-up of her legs #legs'),
    'medium shot from the waist up' → ('framing', 'medium shot from the waist up #medium').
    렌즈·각도가 함께 있으면 ('85mm close-up') 카메라에 그대로 둡니다.
    """
    from . import framing as framing_mod
    t = (text or "").strip()
    if not t or re.search(r"\b\d{2,3}\s?mm\b|\bf/\d|\blens\b|\bangle\b", t, re.I):
        return None
    part = part_in_text(t)
    if part:
        return "body_closeup", f"{t} #{part}"
    fr = framing_mod.framing_in_text(t)
    if fr and re.search(r"\b(?:shot|close[- ]?up|framing|framed|portrait|view|crop(?:ped)?|headshot)\b", t, re.I):
        return "framing", f"{t} #{fr}"
    return None
