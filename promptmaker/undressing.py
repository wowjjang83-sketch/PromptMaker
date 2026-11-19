"""옷을 벗는 중 / 옷을 벗는 장면 — '누드(이미 벗은 상태)' 와 다른 '벗고 있는 동작'.

* 요청에서 찾기: :func:`detect` ('옷을 벗는 중', '벗는 장면', '탈의', 'undressing', 'taking off her top' …).
  '탈의실' 처럼 장소 이름은 제외합니다.
* 탈의 동작은 ``elements/undressing.txt``, 그때의 의상 상태는 ``elements/undressing_state.txt``
  (사용자 편집 가능). 줄 안의 ``{top}`` ``{bottom}`` ``{dress}`` ``{swimwear}`` ``{lingerie}``
  ``{sleepwear}`` ``{sportswear}`` ``{traditional}`` ``{outerwear}`` ``{garment}`` 은 지금 입은 옷의 이름으로
  바뀌고 ('a white satin blouse' → 'white blouse'), 그 옷을 입지 않았으면 그 줄은 쓰지 않습니다.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

GARMENT_SLOTS = ("top", "bottom", "dress", "swimwear", "lingerie", "sleepwear", "sportswear", "traditional",
                 "outerwear", "garment")
_PLACEHOLDER_RE = re.compile(r"\{(" + "|".join(GARMENT_SLOTS) + r")\}")

# '벗는 중' 계열은 그것만으로, '푸는 / 내리는' 계열은 옷 이름이 앞에 있을 때만 ('머리를 푸는' 은 아님)
_KO_STRONG = re.compile(r"벗는|벗으며|벗으면서|벗으려|벗기\s*시작|벗고\s*있|벗어\s*내리|탈의(?!실)")
_KO_WEAK = re.compile(r"(?:옷|상의|하의|치마|스커트|바지|블라우스|셔츠|브라|비키니|원피스|드레스|속옷|잠옷|가운|수영복|"
                      r"팬티|단추|지퍼|끈|어깨끈)\s*(?:을|를)?\s*(?:푸는|풀고\s*있|풀어\s*헤치|풀며|내리는|내리고\s*있|"
                      r"내리며|내리면서|벗겨)")
_EN_RE = re.compile(
    r"\b(?:undress(?:ing|es)?|disrob(?:ing|es)|strip(?:ping|s)?(?: off| down)?|"
    r"(?:taking|takes|take|pulling|pulls|peeling|peels|slipping|slips|sliding|slides) (?:off|out of|down)\b|"
    r"(?:taking|takes|pulling|pulls|peeling|slipping|sliding) (?:her|their|the) [a-z -]{0,30}?\b(?:off|down)\b|"
    r"remov(?:ing|es) (?:her|their) (?:clothes|clothing|top|shirt|blouse|dress|skirt|bra|bikini|outfit)|"
    r"unbutton(?:ing|s)|unzip(?:ping|s)|unhook(?:ing|s)|unty(?:ing)|unties|"
    r"(?:undoing|undoes|loosening|loosens) (?:her|their|the) [a-z -]{0,30}?\b(?:straps?|strings?|buttons?|zipper|"
    r"ties?|sash|belt|bra|bikini|top|robe)|getting undressed)\b", re.I)
# 쓴 영어 단어 (키워드 검색·'The scene also features' 에 다시 나오지 않게)
CONSUMED = ("undressing", "undress", "undresses", "disrobing", "stripping", "strip", "taking", "takes", "take",
            "off", "clothes", "clothing", "pulling", "peeling", "slipping", "sliding", "removing", "unbuttoning",
            "unzipping", "getting", "undressed", "scene", "out", "undoing", "untying", "unhooking", "loosening",
            "strap", "straps", "string", "strings", "button", "buttons", "zipper", "zip", "sash", "down", "halfway")


def detect(original: str = "", english: str = "") -> bool:
    """요청이 '옷을 벗는 동작/장면' 인지."""
    if original and (_KO_STRONG.search(original) or _KO_WEAK.search(original)):
        return True
    return bool(english and _EN_RE.search(english))


# ------------------------------------------------------------------ 옷 이름
def _garment_terms() -> Dict[str, Tuple[str, ...]]:
    from .element_check import GARMENT_TERMS
    return GARMENT_TERMS


_COLOR_MATERIAL = {"white", "black", "red", "pink", "blue", "navy", "green", "yellow", "purple", "beige", "brown",
                   "grey", "gray", "silver", "gold", "golden", "ivory", "cream", "pastel", "denim", "leather",
                   "lace", "satin", "silk", "sheer", "pleated", "floral", "striped", "knit", "velvet", "tweed",
                   "linen", "cotton", "wet", "oversized", "cropped", "mini", "tight", "ruffled", "sequined"}


def garment_noun(text: str, part: str = "") -> str:
    """'a School-style tennis micro mini pleated skirt' → 'pleated skirt' (수식어 하나 + 중심 명사)."""
    low = (text or "").lower()
    terms = _garment_terms()
    pool = list(terms.get(part, ())) if part in terms else [t for v in terms.values() for t in v]
    best: Optional[Tuple[int, int, str]] = None
    for term in pool:
        for m in re.finditer(r"\b" + re.escape(term) + r"s?\b", low):
            cand = (m.end(), len(term), m.group(0))
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is None:
        return ""
    end, _, noun = best
    before = low[:end - len(noun)].split()
    if before:
        w = re.sub(r"[^a-z-]", "", before[-1])
        if w in _COLOR_MATERIAL and w not in noun:
            return f"{w} {noun}"
    return noun


def fill_garments(text: str, garments: Dict[str, str]) -> Optional[str]:
    """줄 안의 {top} 등을 옷 이름으로. 입지 않은 옷을 말하는 줄이면 None."""
    missing = False

    def repl(m: "re.Match[str]") -> str:
        nonlocal missing
        name = garments.get(m.group(1), "")
        if not name:
            missing = True
        return name

    out = _PLACEHOLDER_RE.sub(repl, text)
    return None if missing else out


def placeholders(text: str) -> List[str]:
    return _PLACEHOLDER_RE.findall(text or "")


# 탈의 동작과 함께 쓰기 어려운 포즈·동작 (손이 바쁘거나 몸 전체가 움직이는 것)
BUSY_RE = re.compile(r"\b(?:holding|holds|hold|sipping|sips|reading|writing|pens|playing|plays|carrying|carries|"
                     r"eating|applying|painting|selfie|umbrella|bouquet|book|cup|glass|piano|camera|fan|teacup|"
                     r"dancing|dances|twirling|twirls|spinning|balancing|walking|walks|strolling|running|jumping|"
                     r"hugging|crossed|crossing|stretching|swinging|riding|blowing a kiss|both hands|"
                     r"hands? in (?:her|their) hair|hands? on (?:her|their) hips?)\b", re.I)
