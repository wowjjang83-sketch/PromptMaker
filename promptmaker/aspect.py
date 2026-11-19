"""이미지 비율 추천 (0.14.0) — 완성된 프롬프트의 핵심 피사체로 화면 비율을 정합니다.

프롬프트(영문)에서 다음을 읽어 세로 · 정사각형 · 가로 점수를 매깁니다.

* **자세 · 포즈 · 행동**: 서 있기 · 걷기 · 기대 서기 → 세로 / 눕기 · 엎드리기 · 옆으로 누운 → 가로 /
  앉기 · 무릎 꿇기 → 약한 세로 / 양반다리 · 웅크리기 → 정사각형 쪽
* **샷 구도** (:mod:`promptmaker.framing`): 얼빡샷 → 정사각형 / 클로즈업 · 미디엄 → 세로 / 전신 → 세로
* **피사체 거리** (:mod:`promptmaker.distance`): 원거리 · 원경 → 가로 (배경이 넓게)
* **장면**: 와이드 · 파노라마 · 풍경 → 가로, 두 사람 나란히 → 가로

결과 비율 키는 ``9:16`` · ``1:1`` · ``16:9`` · ``2:1`` (아주 넓은 가로) · ``9:18`` (아주 긴 세로).

    from promptmaker.aspect import suggest_aspect
    suggest_aspect("A woman lying on her side on a bed ...")
    # {'ratio': '16:9', 'orientation': 'landscape', 'reasons': ['누운 자세 → 가로'], 'scores': {...}}
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from . import distance as distance_mod
from . import framing as framing_mod

RATIOS = ("9:16", "1:1", "16:9", "2:1", "9:18")

# (패턴, 세로, 정사각형, 가로, 이유) — 먼저 걸린 자세가 핵심 자세 (프롬프트는 피사체 문장이 앞쪽)
_POSE_RULES: Tuple[Tuple[str, float, float, float, str], ...] = (
    (r"lying (?:on (?:her|his|their) (?:side|stomach|front|belly)|face[- ]down|prone)|lies on (?:her|his) (?:side|stomach)|"
     r"on (?:her|his) side|sprawl\w*|stretched out|reclin\w*|lounging|lounges|lying down|lies down|"
     r"lying (?:on|in|across|back)|lies (?:on|in|across|back)|laid (?:out|back)|supine|napping|sleeping|asleep",
     0, 0.5, 4, "누운 · 기댄 자세 → 가로"),
    (r"on all fours|crawl\w*|on (?:her|his) hands and knees", 0, 1, 3, "엎드린 자세 → 가로"),
    (r"stand\w*|walk\w*|strid\w*|stroll\w*|leaning against|leans against|posing upright|contrapposto|"
     r"tiptoe\w*|running|jumping|dancing|twirl\w*", 3, 0, 0, "서 있기 · 걷기 → 세로"),
    (r"kneel\w*|on (?:her|his) knees", 2, 1, 0, "무릎 꿇은 자세 → 세로"),
    (r"cross[- ]legged|squat\w*|crouch\w*|hugging (?:her|his) knees|curled up", 0.5, 2, 0.5,
     "웅크리거나 양반다리 → 정사각형"),
    (r"sitting|seated|sits|perched", 1.5, 1, 0.5, "앉은 자세 → 세로"),
)
_POSE_RES = [(re.compile(r"\b(?:" + p + r")\b", re.I), v, s, h, why) for p, v, s, h, why in _POSE_RULES]

_WIDE_SCENE = re.compile(r"\b(?:panoram\w*|wide[- ]angle (?:landscape|vista)|sweeping (?:view|vista|landscape)|"
                         r"vast (?:landscape|field|ocean|desert|sky)|expansive (?:landscape|view)|cityscape|skyline|"
                         r"horizon|wide establishing shot|wide shot|landscape (?:shot|photograph))\b", re.I)
_TWO_PEOPLE = re.compile(r"\b(?:side by side|two (?:women|girls|people|friends)|couple|together with (?:a|her) "
                         r"(?:man|boyfriend|partner)|group of)\b", re.I)
_TALL = re.compile(r"\b(?:head to toe|full[- ]length|entire body|whole body)\b", re.I)
# 핵심 피사체 문장: 'she/her/woman' 이 나오는 문장 (배경 묘사의 'a man walking by' 같은 것 제외)
_SUBJECT = re.compile(r"\b(?:she|her|woman|girl|model|lady)\b", re.I)


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def suggest_aspect(prompt: str) -> Dict[str, Any]:
    """프롬프트로 화면 비율을 추천합니다. 반환: ``{ratio, orientation, reasons, scores, framing, distance}``."""
    text = " ".join((prompt or "").split())
    scores = {"portrait": 0.0, "square": 0.0, "landscape": 0.0}
    found: List[Tuple[str, str]] = []   # (이유, 그 이유가 미는 방향)

    def add(v: float, s: float, h: float, why: str) -> None:
        scores["portrait"] += v
        scores["square"] += s
        scores["landscape"] += h
        way = max((("portrait", v), ("square", s), ("landscape", h)), key=lambda x: x[1])[0]
        if why and why not in [w for w, _ in found]:
            found.append((why, way))

    subj = [s for s in _sentences(text) if _SUBJECT.search(s)] or _sentences(text)
    subj_text = " ".join(subj)

    # 1) 자세 · 포즈 · 행동 — 피사체 문장에서 처음 걸린 자세를 크게, 나머지는 작게
    first = True
    for rx, v, s, h, why in sorted(_POSE_RES, key=lambda r: (r[0].search(subj_text).start()
                                                              if r[0].search(subj_text) else 10 ** 9)):
        if not rx.search(subj_text):
            continue
        k = 1.0 if first else 0.35
        add(v * k, s * k, h * k, why)
        first = False

    # 2) 샷 구도
    fr = None
    hits = framing_mod._find(text, english=True)
    if hits:
        fr = hits[0][1]
    elif _TALL.search(text):
        fr = "full_body"
    if fr == "extreme_closeup":
        add(0.5, 4, 0, "얼빡샷 → 정사각형")
    elif fr == "closeup":
        add(2, 2, 0, "클로즈업 → 세로 · 정사각형")
    elif fr == "medium":
        add(2, 0.5, 0, "미디엄샷 → 세로")
    elif fr == "cowboy":
        add(2, 0, 0, "카우보이샷 → 세로")
    elif fr == "full_body":
        add(2, 0, 0.5, "전신샷 → 세로")

    # 3) 피사체 거리
    dist = None
    dhits = distance_mod._find(text, english=True)
    if dhits:
        dist = dhits[-1][1]
    if dist == "very_far":
        add(0, 0, 4, "아주 먼 거리(원경) → 넓은 가로")
    elif dist == "far":
        add(0, 0, 2.5, "원거리 → 가로")
    elif dist == "close":
        add(0.5, 1, 0, "근거리 → 정사각형 쪽")

    # 4) 장면
    if _WIDE_SCENE.search(text):
        add(0, 0, 2, "와이드 · 풍경 장면 → 가로")
    if _TWO_PEOPLE.search(text):
        add(0, 0.5, 1.5, "두 사람 이상 → 가로")

    if not any(scores.values()):
        return {"ratio": "9:16", "orientation": "portrait", "reasons": ["판단할 자세 · 구도가 없어 기본 세로"],
                "scores": scores, "framing": fr, "distance": dist}

    orientation = max(("portrait", "square", "landscape"), key=lambda o: (scores[o], o == "portrait"))
    if orientation == "landscape":
        wide = dist == "very_far" or (_WIDE_SCENE.search(text) and dist == "far")
        ratio = "2:1" if wide else "16:9"
    elif orientation == "square":
        ratio = "1:1"
    else:
        lying = scores["landscape"] >= 3
        ratio = "9:16" if lying or not (fr == "full_body" and scores["portrait"] >= 6 and not dist) else "9:18"
    # 고른 방향을 민 이유를 앞에
    reasons = [w for w, way in found if way == orientation] + [w for w, way in found if way != orientation]
    return {"ratio": ratio, "orientation": orientation, "reasons": reasons, "scores": scores,
            "framing": fr, "distance": dist}
