"""피사체 거리 (0.14.0) — 카메라가 피사체에서 얼마나 떨어져 찍는가.

샷 구도(:mod:`promptmaker.framing`)가 '몸의 어디까지 담는가' 라면, 피사체 거리는 '얼마나 떨어져서 찍는가' 입니다.
멀리서 찍으면 인물이 화면에서 작아지고 주변 배경이 넓게 담기며, 가까이서 찍으면 인물이 화면을 채웁니다.

=========  ===============================  ======================  ===================
거리 id     요청 예                            어울리는 구도              화면 느낌
=========  ===============================  ======================  ===================
close      가까이서, 근거리, 코앞에서, up close   얼빡샷·클로즈업·미디엄       인물이 화면을 채움
medium     중거리, 몇 걸음 떨어져서              클로즈업 ~ 전신              인물 + 주변 조금
far        멀리서, 원거리, from a distance      카우보이 · 전신              인물 + 배경 넓게
very_far   아주 멀리서, 원경, tiny figure        전신                        풍경 속 작은 인물
=========  ===============================  ======================  ===================

* 요청에서 거리를 찾고 (:func:`detect`), 둘 이상이면 마지막에 말한 쪽을 씁니다.
* 구도와 맞지 않으면 (얼빡샷 + 멀리서) 구도를 따르고 거리를 구도에 맞는 쪽으로 바꿉니다 (:func:`fit_to_framing`).
* 구도를 말하지 않았으면 거리로 구도를 정합니다 (멀리서 → 전신샷, :data:`DEFAULT_FRAMING`).
* 문구는 ``elements/distance.txt`` (사용자 편집 가능) 에서 ``#<거리 id>`` 태그로 찾습니다.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

DISTANCES: Tuple[str, ...] = ("close", "medium", "far", "very_far")
LABELS_KO: Dict[str, str] = {"close": "근거리 (가까이서)", "medium": "중거리", "far": "원거리 (멀리서)",
                             "very_far": "아주 먼 거리 (원경)"}
RANK: Dict[str, int] = {d: i for i, d in enumerate(DISTANCES)}
# 거리별로 어울리는 샷 구도 (framing.FRAMINGS + body_part)
FRAMINGS_FOR: Dict[str, Tuple[str, ...]] = {
    "close": ("extreme_closeup", "closeup", "medium", "body_part"),
    "medium": ("closeup", "medium", "cowboy", "full_body"),
    "far": ("cowboy", "full_body"),
    "very_far": ("full_body",),
}
# 구도를 말하지 않았을 때 거리로 정하는 구도
DEFAULT_FRAMING: Dict[str, Optional[str]] = {"close": "medium", "medium": None, "far": "full_body", "very_far": "full_body"}
# 구도에 가장 잘 맞는 거리 (구도와 충돌할 때 이 쪽으로 바꿈)
BEST_FOR_FRAMING: Dict[str, str] = {"extreme_closeup": "close", "closeup": "close", "medium": "medium",
                                    "cowboy": "medium", "full_body": "far", "body_part": "close"}
# distance.txt 가 없거나 비었을 때 쓰는 기본 문구
DEFAULT_PHRASES: Dict[str, str] = {
    "close": "shot up close from just an arm's length away",
    "medium": "shot from a few steps away",
    "far": "shot from a distance, with {her} figure set against the wide surroundings",
    "very_far": "shot from far away, {her} small figure in a vast scene",
}

# ------------------------------------------------------------------ 요청에서 찾기 (긴 말 먼저)
_KO_TERMS: Tuple[Tuple[str, str], ...] = (
    ("아주 멀리서", "very_far"), ("아주멀리서", "very_far"), ("아주 먼 거리", "very_far"), ("까마득히", "very_far"),
    ("초원거리", "very_far"), ("초 원거리", "very_far"), ("원경", "very_far"), ("익스트림 롱샷", "very_far"),
    ("익스트림롱샷", "very_far"), ("풍경 속 작은", "very_far"), ("작게 보이", "very_far"),
    ("멀리서", "far"), ("멀리 떨어져", "far"), ("멀찍이", "far"), ("원거리", "far"), ("먼 거리", "far"),
    ("먼거리", "far"), ("롱샷", "far"), ("롱 샷", "far"), ("떨어져서", "far"),
    ("중거리", "medium"), ("적당한 거리", "medium"), ("몇 걸음 떨어", "medium"), ("몇걸음 떨어", "medium"),
    ("가까이서", "close"), ("가까이에서", "close"), ("가까운 거리", "close"), ("근거리", "close"),
    ("코앞", "close"), ("바로 앞에서", "close"), ("밀착", "close"), ("가까이", "close"),
)
_EN_TERMS: Tuple[Tuple[str, str], ...] = (
    (r"extreme (?:long|wide) shots?", "very_far"), (r"from (?:very|really|so) far(?: away)?", "very_far"),
    (r"tiny (?:distant )?figure", "very_far"), (r"small figure in (?:a|the) vast", "very_far"),
    (r"(?:very|extremely) distant", "very_far"), (r"from (?:a )?great distance", "very_far"),
    (r"from (?:a )?distance", "far"), (r"from afar", "far"), (r"from far(?: away)?", "far"),
    (r"far away", "far"), (r"long[- ]distance", "far"), (r"distant (?:shot|view)", "far"),
    (r"from across the (?:room|street|road|park|beach|field)", "far"), (r"long[- ]range", "far"),
    (r"(?:medium|mid)[- ]distance", "medium"), (r"from a few (?:steps|feet|meters|metres|paces) away", "medium"),
    (r"a few (?:steps|paces) away", "medium"),
    (r"up close", "close"), (r"close[- ]range", "close"), (r"close distance", "close"),
    (r"at arm'?s length", "close"), (r"from (?:very )?close(?: by)?", "close"), (r"inches away", "close"),
)
_EN_RES = [(re.compile(r"\b" + p + r"\b", re.I), d) for p, d in _EN_TERMS]
# '가까이' 가 '가까이 다가가는'(동작) 처럼 거리가 아닌 뜻일 때는 빼기
_KO_NOT_DISTANCE = re.compile(r"가까이\s*(?:다가|붙|와|오|가서|간다|가는)")


def _find(text: str, english: bool) -> List[Tuple[int, str, str]]:
    found: List[Tuple[int, str, str]] = []
    if not text:
        return found
    buf = text
    if english:
        for rx, d in _EN_RES:
            for m in rx.finditer(buf):
                found.append((m.start(), d, m.group(0)))
            buf = rx.sub(lambda m: " " * len(m.group(0)), buf)
    else:
        buf = _KO_NOT_DISTANCE.sub(lambda m: " " * len(m.group(0)), buf)
        for term, d in _KO_TERMS:
            start = 0
            while True:
                i = buf.find(term, start)
                if i < 0:
                    break
                found.append((i, d, term))
                buf = buf[:i] + " " * len(term) + buf[i + len(term):]
                start = i + len(term)
    return sorted(found)


def detect(original: str = "", english: str = "") -> Tuple[Optional[str], List[str], List[str]]:
    """요청에서 피사체 거리를 찾습니다.

    반환: ``(거리 id 또는 None, 찾은 거리 목록(나온 순서), 영어 요청에서 찾은 거리 말)``.
    둘 이상이면 마지막에 말한 쪽을 씁니다 (수식어가 앞, 중심어가 뒤).
    """
    hits = _find(original, english=False) if original else []
    hits += [(10_000 + p, d, w) for p, d, w in _find(original, english=True)] if original else []
    en_hits = _find(english, english=True) if english else []
    if not hits:
        hits = en_hits
    words = [h[2] for h in en_hits]
    if not hits:
        return None, [], words
    order: List[str] = []
    for _, d, _ in sorted(hits):
        if d in order:
            order.remove(d)
        order.append(d)
    return order[-1], order, words


def ko_to_english(text: str) -> str:
    """번역 전에 한국어 거리 말을 영어로 바꿉니다 ('원경' 처럼 번역기가 엉뚱하게 옮기는 말 방지)."""
    if not text:
        return text
    en = {"close": " up close ", "medium": " from a few steps away ", "far": " from a distance ",
          "very_far": " from very far away "}
    keep = [m.group(0) for m in _KO_NOT_DISTANCE.finditer(text)]   # '가까이 다가가는' 같은 동작 말은 그대로
    out = _KO_NOT_DISTANCE.sub("\x00", text)
    for term, d in _KO_TERMS:
        out = out.replace(term, en[d])
    for k in keep:
        out = out.replace("\x00", k, 1)
    return re.sub(r"\s{2,}", " ", out).strip()


def distance_in_text(text: str) -> Optional[str]:
    """요소 한 줄이 말하는 거리 ('shot from a distance' → far)."""
    hits = _find(text or "", english=True)
    return hits[0][1] if hits else None


def compatible(distance: Optional[str], framing: Optional[str]) -> bool:
    if not distance or not framing:
        return True
    return framing in FRAMINGS_FOR.get(distance, ())


def fit_to_framing(distance: Optional[str], framing: Optional[str]) -> Optional[str]:
    """구도와 맞지 않는 거리를 구도에 맞는 거리로 바꿉니다 (맞으면 그대로)."""
    if not distance or not framing or compatible(distance, framing):
        return distance
    return BEST_FOR_FRAMING.get(framing, distance)


def route_phrase(text: str) -> Optional[Tuple[str, str]]:
    """분석 문구가 피사체 거리만 말하면 ``('distance', '문구 #거리')``. 구도·렌즈가 함께 있으면 None."""
    from . import framing as framing_mod
    t = (text or "").strip()
    if not t or framing_mod.framing_in_text(t) or re.search(r"\b\d{2,3}\s?mm\b|\bf/\d|\blens\b", t, re.I):
        return None
    d = distance_in_text(t)
    if d and len(t.split()) <= 14:
        return "distance", f"{t} #{d}"
    return None
