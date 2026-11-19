"""샷 구도(프레이밍) — 동작·의상과 따로 정하고, 서로 맞지 않는 조합을 막습니다.

구도는 '카메라가 몸의 어디까지 담는가' 하나로 정리합니다.

=================  ==============================  =========================
구도 id             요청 예                           보이는 범위 (rank)
=================  ==============================  =========================
extreme_closeup    얼빡샷, 얼굴만, extreme close-up   얼굴만 (0)
closeup            클로즈업, 헤드샷, close-up          머리 ~ 어깨 (1)
medium             미디엄샷, 상반신, waist up           허리 위 (3)
cowboy             카우보이샷, cowboy shot            허벅지 중간 위 (4)
full_body          전신샷, 풀샷, full body            머리 ~ 발끝 (5)
=================  ==============================  =========================

* 요청에서 구도를 찾고 (:func:`detect`), 구도가 둘 이상이면 하나만 고릅니다 (클로즈업 + 전신 같은 충돌).
* 포즈·동작·의상 상태 한 줄이 몸의 어디까지 보여야 하는지 (:func:`region_rank`) 를 보고, 구도에 안 담기는
  줄은 고르지 않습니다 (얼빡샷에 '다리를 꼬고 앉은', 미디엄샷에 '하이힐').
* 구도 문구는 ``elements/framing.txt`` (사용자 편집 가능) 에서 ``#<구도 id>`` 태그로 찾습니다.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

FRAMINGS: Tuple[str, ...] = ("extreme_closeup", "closeup", "medium", "cowboy", "full_body")
# 구도별로 담기는 몸 범위 (region_rank 가 이 값 이하인 줄만 쓸 수 있음)
# 신체 부위 클로즈업 (0.11.0): 부위는 body_parts.PARTS. 담기는 범위는 부위가 정함 (body_parts 참고)
BODY_PART = "body_part"
MAX_RANK: Dict[str, int] = {"extreme_closeup": 0, "closeup": 1, "medium": 3, "cowboy": 4, "full_body": 5,
                            BODY_PART: 5}
LABELS_KO: Dict[str, str] = {"extreme_closeup": "얼빡샷(초근접 얼굴)", "closeup": "클로즈업", "medium": "미디엄샷",
                             "cowboy": "카우보이샷", "full_body": "전신샷", BODY_PART: "신체 부위 클로즈업"}
# framing.txt 가 없거나 비었을 때 쓰는 기본 문구
DEFAULT_PHRASES: Dict[str, str] = {
    "extreme_closeup": "extreme close-up of {her} face only, filling the entire frame from forehead to chin",
    "closeup": "close-up shot of {her} face and shoulders",
    "medium": "medium shot framed from the waist up",
    "cowboy": "cowboy shot framed from mid-thigh up",
    "full_body": "full-body shot showing {her} from head to toe",
}
CLOSE = frozenset(("extreme_closeup", "closeup"))

# ------------------------------------------------------------------ 요청에서 찾기
# 긴 말부터 찾고, 찾은 자리는 지워서 짧은 말이 다시 걸리지 않게 함 ('extreme close-up' 안의 'close-up')
_KO_TERMS: Tuple[Tuple[str, str], ...] = (
    ("익스트림 클로즈업", "extreme_closeup"), ("익스트림클로즈업", "extreme_closeup"), ("얼빡샷", "extreme_closeup"),
    ("얼빡", "extreme_closeup"), ("얼굴만", "extreme_closeup"), ("초근접", "extreme_closeup"),
    ("초 근접", "extreme_closeup"), ("초접사", "extreme_closeup"),
    ("클로즈업", "closeup"), ("클로즈 업", "closeup"), ("근접샷", "closeup"), ("근접 샷", "closeup"),
    ("헤드샷", "closeup"), ("헤드 샷", "closeup"),
    ("미디엄샷", "medium"), ("미디엄 샷", "medium"), ("미디움샷", "medium"), ("상반신", "medium"),
    ("허리 위", "medium"), ("허리위", "medium"), ("바스트샷", "medium"), ("바스트 샷", "medium"), ("바스트업", "medium"),
    ("웨이스트샷", "medium"),
    ("카우보이샷", "cowboy"), ("카우보이 샷", "cowboy"), ("무릎 위", "cowboy"), ("무릎위", "cowboy"),
    ("허벅지 위", "cowboy"), ("허벅지위", "cowboy"),
    ("전신샷", "full_body"), ("전신 샷", "full_body"), ("전신", "full_body"), ("풀샷", "full_body"),
    ("풀 샷", "full_body"), ("풀바디", "full_body"), ("머리부터 발끝", "full_body"),
)
_EN_TERMS: Tuple[Tuple[str, str], ...] = (
    (r"extreme[- ]?close[- ]?ups?", "extreme_closeup"), (r"ecu", "extreme_closeup"),
    (r"face[- ]only", "extreme_closeup"), (r"only (?:her|the|their) faces?", "extreme_closeup"),
    (r"tight face shot", "extreme_closeup"), (r"face filling the (?:entire |whole )?frame", "extreme_closeup"),
    (r"close[- ]?ups?", "closeup"), (r"head ?shots?", "closeup"), (r"head[- ]and[- ]shoulders?", "closeup"),
    (r"medium(?: close)? shots?", "medium"), (r"mid[- ]shots?", "medium"), (r"waist[- ]up", "medium"),
    (r"upper[- ]body", "medium"), (r"half[- ]body", "medium"), (r"bust[- ](?:shot|up)s?", "medium"),
    (r"cowboy[- ]shots?", "cowboy"), (r"american shots?", "cowboy"), (r"knee[- ]shots?", "cowboy"),
    (r"(?:mid[- ])?thighs?[- ]up", "cowboy"), (r"three[- ]quarter shots?", "cowboy"),
    (r"full[- ]body(?: shots?)?", "full_body"), (r"full[- ]length(?: shots?)?", "full_body"),
    (r"full shots?", "full_body"), (r"head to toe", "full_body"), (r"(?:entire|whole) body", "full_body"),
    (r"long shots?", "full_body"),
)
_EN_RES = [(re.compile(r"\b" + p + r"\b", re.I), f) for p, f in _EN_TERMS]


def _find(text: str, english: bool) -> List[Tuple[int, str, str]]:
    """[(위치, 구도, 찾은 말), ...] — 긴 말 먼저, 찾은 자리는 가림."""
    found: List[Tuple[int, str, str]] = []
    if not text:
        return found
    buf = text
    if english:
        for rx, fr in _EN_RES:
            for m in rx.finditer(buf):
                found.append((m.start(), fr, m.group(0)))
            buf = rx.sub(lambda m: " " * len(m.group(0)), buf)
    else:
        for term, fr in _KO_TERMS:
            start = 0
            while True:
                i = buf.find(term, start)
                if i < 0:
                    break
                found.append((i, fr, term))
                buf = buf[:i] + " " * len(term) + buf[i + len(term):]
                start = i + len(term)
    return sorted(found)


def detect(original: str = "", english: str = "") -> Tuple[Optional[str], List[str], List[str]]:
    """요청에서 샷 구도를 찾습니다.

    반환: ``(구도 id 또는 None, 찾은 구도 목록(중복 제거, 나온 순서), 영어 요청에서 찾은 구도 말)``.
    구도가 둘 이상이면 하나만 고릅니다.

    * 얼빡샷 + 클로즈업처럼 둘 다 '가까운' 구도면 더 가까운 쪽 (얼빡샷)
    * 클로즈업 + 전신샷처럼 함께 쓸 수 없으면 마지막에 말한 쪽 ('클로즈업 전신샷' → 전신샷).
      한국어·영어 모두 수식어가 앞, 중심어가 뒤에 오기 때문입니다.
    """
    hits = _find(original, english=False) if original else []
    hits += [(10_000 + p, f, w) for p, f, w in _find(original, english=True)] if original else []
    if not hits and english:
        hits = _find(english, english=True)
    en_hits = _find(english, english=True) if english else []
    words = [h[2] for h in en_hits]
    if not hits:
        return None, [], words
    order: List[str] = []
    for _, fr, _ in sorted(hits):
        if fr in order:
            order.remove(fr)
        order.append(fr)
    if set(order) <= CLOSE:
        return ("extreme_closeup" if "extreme_closeup" in order else "closeup"), order, words
    return order[-1], order, words


def ko_to_english(text: str) -> str:
    """번역 전에 한국어 구도 말을 영어로 바꿉니다 ('얼빡샷' 이 'a quick shot' 으로 번역되는 것 방지)."""
    if not text:
        return text
    en = {"extreme_closeup": "extreme close-up face shot", "closeup": "close-up shot", "medium": "medium shot",
          "cowboy": "cowboy shot", "full_body": "full-body shot"}
    out = text
    for term, fr in _KO_TERMS:
        out = out.replace(term, f" {en[fr]} ")
    return re.sub(r"\s{2,}", " ", out).strip()


# ------------------------------------------------------------------ 문구가 말하는 구도
def framing_in_text(text: str) -> Optional[str]:
    """요소 한 줄이 스스로 구도를 말하면 그 구도 ('full-body shot' → full_body, 'headshot' → closeup)."""
    hits = _find(text or "", english=True)
    extra = re.search(r"\b(?:wide[- ]angle environmental|environmental portrait|wide shot|lookbook)\b", text or "", re.I)
    if not hits:
        return "full_body" if extra else None
    return hits[0][1]


def compatible(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b or a == b:
        return True
    return a in CLOSE and b in CLOSE


# ------------------------------------------------------------------ 몸 범위
_HAND_NEAR_FACE = re.compile(r"\b(?:chin|cheeks?|face|lips?|mouth|ears?|hair|strand|eyes?|forehead|temple)\b", re.I)
_REGIONS: Tuple[Tuple[int, "re.Pattern[str]"], ...] = (
    (5, re.compile(r"\b(?:anklets?|feet|foot|toes?|tiptoes?|barefoot|ankles?|heels|high heels|stilettos|shoes|sneakers|boots|"
                   r"sandals|loafers|knees?|kneeling|kneels|cross-legged|crouching|crouches|squatting|squats|"
                   r"legs|leg|full[- ]length|head to toe|whole (?:body|scene)|entire body|full figure|"
                   r"reveal the whole)\b", re.I)),
    (4, re.compile(r"\b(?:thighs?|hips?|lap|hem|stockings|garters?|buttocks|butt|bottoms?|skirt|shorts|jeans|"
                   r"pants|trousers)\b", re.I)),
    (3, re.compile(r"\b(?:waist|waistline|midriff|belly|navel|stomach|torso|abs|belt)\b", re.I)),
    (2, re.compile(r"\b(?:dancing|dances|twirling|twirls|spinning|spins|walking|walks|strolling|running|"
                   r"jumping|swaying|balancing|chest|cleavage|breasts?|bust|arms?|hands?|fingers?|elbows?|wrists?|holding|holds|"
                   r"sipping|reading|writing|cup|glass|book|bouquet|umbrella|phone|selfie|bag|"
                   r"stretching|hugging|crossed|crossing)\b", re.I)),
    (1, re.compile(r"\b(?:shoulders?|collarbones?|neck|nape|straps?|neckline|back)\b", re.I)),
)


def region_rank(text: str) -> int:
    """이 문구가 보여 주려면 몸의 어디까지 화면에 담겨야 하는지 (0 = 얼굴만 … 5 = 발끝까지)."""
    low = (text or "").lower()
    for rank, rx in _REGIONS:
        m = rx.search(low)
        if not m:
            continue
        if rank == 2 and _HAND_NEAR_FACE.search(low) and \
                re.search(r"\b(?:hands?|fingers?|chin|cheek)\b", low) and not re.search(
                    r"\b(?:holding|holds|sipping|reading|writing|cup|glass|book|bouquet|umbrella|phone|selfie|"
                    r"camera|bag|crossed|crossing|chest|cleavage|arms?)\b", low):
            return 1  # 얼굴에 댄 손 ('chin in her hand', 'fingers on her lips')
        return rank
    return 0


def fits(text: str, framing: Optional[str]) -> bool:
    """이 문구를 이 구도 안에서 보여 줄 수 있는지."""
    if not framing:
        return True
    if not compatible(framing_in_text(text), framing):
        return False
    return region_rank(text) <= MAX_RANK[framing]


# 의상 부위별로 담기는 최소 구도 rank (이보다 좁은 구도면 의상 조합에서 뺌)
GARMENT_RANK: Dict[str, int] = {"footwear": 5, "legwear": 5, "bottom": 4, "accessory": 0,
                                "top": 1, "dress": 1, "swimwear": 1, "lingerie": 1, "sleepwear": 1,
                                "sportswear": 1, "traditional": 1, "outerwear": 1}


def garment_visible(part: str, framing: Optional[str]) -> bool:
    if not framing:
        return True
    return GARMENT_RANK.get(part, 0) <= MAX_RANK[framing] and not (framing == "extreme_closeup" and part != "accessory")
