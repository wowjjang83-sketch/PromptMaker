"""카메라 시점·앵글 (0.16.0) — 카메라가 피사체를 '어느 쪽에서, 어느 높이에서' 보는가.

샷 구도(:mod:`promptmaker.framing`)가 '몸의 어디까지', 피사체 거리(:mod:`promptmaker.distance`)가 '얼마나 떨어져서' 라면,
시점은 '어느 방향·높이에서' 입니다. 실사 인물 카메라 컨트롤 사례(42개 시점 학습 LoRA 캡션)에서 효과가 확인된 방식을 따릅니다.

* **방향** (좌우 회전): 정면 · 정면 사선(3/4) · 측면 · 뒤 사선 · 뒷모습
* **높이** (위아래): 눈높이 · 하이앵글(내려다봄) · 탑뷰(바로 위) · 로우앵글(올려다봄) · 웜즈아이(바닥 가까이)
* **더치 앵글** (기울기), **화면 방향** (인물이 화면 왼쪽/오른쪽을 향함)

적용한 팁

1. 시점 이름만 쓰지 말고 **몸통이 어떻게 보이는지** 함께 적습니다 ('front three-quarter view, with the front and one
   side of her torso visible'). 정면 사선은 이름만 쓰면 측면처럼 나오는 경우가 많았습니다.
2. **뒷모습 · 뒤 사선**에서는 얼굴이 보여야 하는 표현(미소 · 눈 맞춤 · 윙크 · 눈 화장)을 빼고, 표정 자리는
   '얼굴을 돌린' 같은 말로 바꿉니다 (요청에 표정을 직접 말했으면 그대로 둠).
3. 완전히 바로 아래에서 올려다보는 시점(−90°)은 인물 형태가 무너지므로, 웜즈아이는 '바닥 가까이에서 올려다보는' 으로 씁니다.
4. 무작위 시점은 극단 시점(뒷모습 · 바로 위 · 바닥 · 기울기)을 피하고 정면 사선 · 측면 · 약한 높낮이만 씁니다.

문구는 ``elements/view.txt`` (사용자 편집 가능) 에서 ``#<시점 id>`` 태그로 찾습니다.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

DIRECTIONS: Tuple[str, ...] = ("front", "front_3q", "side", "rear_3q", "rear")
HEIGHTS: Tuple[str, ...] = ("eye_level", "high", "overhead", "low", "worm")
LABELS_KO: Dict[str, str] = {
    "front": "정면", "front_3q": "정면 사선 (3/4)", "side": "측면 (옆모습)", "rear_3q": "뒤 사선 (뒤돌아봄)",
    "rear": "뒷모습",
    "eye_level": "눈높이", "high": "하이앵글 (내려다봄)", "overhead": "탑뷰 (바로 위)", "low": "로우앵글 (올려다봄)",
    "worm": "웜즈아이 (바닥 가까이)", "dutch": "더치 앵글 (기울기)", "left": "화면 왼쪽을 향함", "right": "화면 오른쪽을 향함",
}
# 얼굴이 안 보이는 방향 — 표정 자리를 '얼굴을 돌린' 말로 바꾸고 눈 맞춤 줄을 뺌
FACE_HIDDEN: Tuple[str, ...] = ("rear",)
# 몸이 카메라 반대쪽을 향하는 방향 — '카메라 쪽으로 걸어오는' 같은 줄을 뺌 (뒤 사선은 돌아보는 얼굴은 허용)
BACK_TURNED: Tuple[str, ...] = ("rear", "rear_3q")
# view.txt 가 없거나 비었을 때 쓰는 기본 문구
DEFAULT_PHRASES: Dict[str, str] = {
    "front": "straight-on front view, with the front of {her} body facing the camera",
    "front_3q": "front three-quarter view, {her} body turned about 45 degrees, with the front and one side of {her} torso visible",
    "side": "side profile view, the camera looking directly at the side of {her} body",
    "rear_3q": "rear three-quarter view, with {her} back and one side of {her} body visible",
    "rear": "straight-on rear view, {her} back to the camera as {she} faces away",
    "eye_level": "shot at eye level",
    "high": "shot from a high angle, looking down at {her}",
    "overhead": "overhead shot looking down at {her} from directly above",
    "low": "shot from a low angle, looking up at {her}",
    "worm": "very low angle from near the ground, looking up at {her}",
    "dutch": "dutch angle with the frame slightly tilted",
    "left": "{her} body turned toward the left side of the frame",
    "right": "{her} body turned toward the right side of the frame",
}
# 뒷모습에서 표정 자리에 쓰는 말 (얼굴이 안 보이므로)
REAR_EXPRESSIONS: Dict[str, Tuple[str, ...]] = {
    "rear": ("{her} face turned away from the camera", "{her} head tilted gently to one side",
             "{her} head slightly bowed"),
}
# 얼굴이 보여야만 하는 표현 (뒷모습에서 뺌)
FACE_FRONT_RE = re.compile(
    r"\b(?:(?:at|into) the (?:camera|viewer|lens)|eye contact|winks?|winking)\b", re.I)
TOWARD_CAMERA_RE = re.compile(
    r"\b(?:(?:walk|mov|lean|turn|step|run|crawl|strid|approach)\w*|fac\w*|squar\w*)\s+(?:\w+\s+)?"
    r"(?:toward|towards|to|into|at) the (?:camera|viewer|lens)\b|\bfacing the (?:camera|viewer)\b", re.I)
TOWARD_WORDS_RE = re.compile(r"\btowards? the (?:camera|viewer|lens)\b", re.I)


def turn_away(text: str) -> str:
    """'walking toward the camera' → 'walking away from the camera' (뒷모습에 맞게)."""
    return TOWARD_WORDS_RE.sub("away from the camera", text)


EYE_MAKEUP_RE = re.compile(r"\b(?:eye|eyes|eyeliner|eyeshadow|eye shadow|lashes|eyelashes|mascara|brows?|eyebrows?)\b", re.I)

# 무작위 모드에서 요청에 시점이 없을 때 (극단 시점은 쓰지 않음)
RANDOM_DIRECTIONS: Tuple[Tuple[str, float], ...] = (("front_3q", 3.0), ("side", 1.0))
RANDOM_HEIGHTS: Tuple[Tuple[str, float], ...] = (("high", 1.0), ("low", 1.0))

# ------------------------------------------------------------------ 요청에서 찾기 (축, 값)
# 긴 말부터 찾고, 찾은 자리는 지워서 짧은 말이 다시 걸리지 않게 함 ('뒤 사선' 안의 '사선')
_KO_TERMS: Tuple[Tuple[str, str, str], ...] = tuple(sorted((
    # 방향
    ("뒤 사선", "direction", "rear_3q"), ("뒤사선", "direction", "rear_3q"), ("뒤 대각선", "direction", "rear_3q"),
    ("뒤대각선", "direction", "rear_3q"), ("후면 사선", "direction", "rear_3q"), ("뒤쪽 사선", "direction", "rear_3q"),
    ("뒤 3/4", "direction", "rear_3q"), ("뒤돌아보", "direction", "rear_3q"), ("어깨 너머로 돌아보", "direction", "rear_3q"),
    ("뒷모습", "direction", "rear"), ("뒤태", "direction", "rear"), ("뒤에서 찍", "direction", "rear"),
    ("뒤에서 본", "direction", "rear"), ("후면", "direction", "rear"), ("등지고", "direction", "rear"),
    ("등진", "direction", "rear"), ("백샷", "direction", "rear"), ("백 샷", "direction", "rear"),
    ("옆모습", "direction", "side"), ("옆 모습", "direction", "side"), ("측면", "direction", "side"),
    ("옆에서 찍", "direction", "side"), ("옆에서 본", "direction", "side"), ("옆얼굴", "direction", "side"),
    ("사이드뷰", "direction", "side"), ("사이드 뷰", "direction", "side"),
    ("정면 사선", "direction", "front_3q"), ("정면사선", "direction", "front_3q"), ("앞 사선", "direction", "front_3q"),
    ("반측면", "direction", "front_3q"), ("3/4 각도", "direction", "front_3q"), ("3/4 시점", "direction", "front_3q"),
    ("3/4 뷰", "direction", "front_3q"), ("사선", "direction", "front_3q"), ("대각선에서", "direction", "front_3q"),
    ("대각선 방향", "direction", "front_3q"), ("비스듬히", "direction", "front_3q"), ("45도", "direction", "front_3q"),
    ("정면", "direction", "front"), ("앞모습", "direction", "front"), ("앞에서 찍", "direction", "front"),
    # 높이
    ("웜즈아이", "height", "worm"), ("웜아이", "height", "worm"), ("바닥에서 올려다", "height", "worm"), ("바닥에서 올려다보는", "height", "worm"),
    ("바닥 높이에서", "height", "worm"), ("개미 시점", "height", "worm"), ("극단적인 로우앵글", "height", "worm"),
    ("로우앵글", "height", "low"), ("로우 앵글", "height", "low"), ("올려다보는 시점", "height", "low"),
    ("올려다보는 앵글", "height", "low"), ("올려다보는 구도", "height", "low"), ("올려다보는 각도", "height", "low"),
    ("아래에서 올려다", "height", "low"), ("아래에서 올려다보는", "height", "low"), ("아래에서 위로", "height", "low"), ("밑에서 찍", "height", "low"),
    ("아래에서 찍", "height", "low"), ("앙각", "height", "low"),
    ("탑뷰", "height", "overhead"), ("탑 뷰", "height", "overhead"), ("톱뷰", "height", "overhead"),
    ("바로 위에서", "height", "overhead"), ("바로 위에서 내려다", "height", "overhead"), ("바로 위에서 내려다보는", "height", "overhead"), ("버드아이", "height", "overhead"),
    ("버즈아이", "height", "overhead"), ("조감", "height", "overhead"), ("수직 부감", "height", "overhead"),
    ("하이앵글", "height", "high"), ("하이 앵글", "height", "high"), ("내려다보는 시점", "height", "high"),
    ("내려다보는 앵글", "height", "high"), ("내려다보는 구도", "height", "high"), ("내려다보는 각도", "height", "high"),
    ("위에서 내려다", "height", "high"), ("위에서 내려다보는", "height", "high"), ("위에서 찍", "height", "high"), ("부감", "height", "high"),
    ("눈높이", "height", "eye_level"), ("아이레벨", "height", "eye_level"), ("아이 레벨", "height", "eye_level"),
    # 기울기 · 화면 방향
    ("더치앵글", "dutch", "dutch"), ("더치 앵글", "dutch", "dutch"), ("기울어진 구도", "dutch", "dutch"),
    ("기울어진 앵글", "dutch", "dutch"), ("틸트 앵글", "dutch", "dutch"),
    ("화면 왼쪽을", "facing", "left"), ("화면 왼쪽으로", "facing", "left"), ("화면 오른쪽을", "facing", "right"),
    ("화면 오른쪽으로", "facing", "right"),
), key=lambda t: -len(t[0])))

_EN_TERMS: Tuple[Tuple[str, str, str], ...] = (
    # 방향 (뒤 사선 → 뒷모습 → 측면 → 정면 사선 → 정면 순: 'rear three-quarter view' 가 정면 사선으로 걸리지 않게)
    (r"(?:rear|back) three[- ]quarters? (?:view|angle|profile)", "direction", "rear_3q"),
    (r"(?:looking|glancing|peeking) back over (?:her|his|their|the|one) shoulders?", "direction", "rear_3q"),
    (r"from behind", "direction", "rear"), (r"(?:rear|back) view", "direction", "rear"),
    (r"from the back", "direction", "rear"), (r"back shots?", "direction", "rear"),
    (r"(?:with )?(?:her|his|their) back (?:to|turned to|facing) the (?:camera|viewer)", "direction", "rear"),
    (r"facing away from the (?:camera|viewer)", "direction", "rear"),
    (r"side[- ]profile(?: view| shot)?", "direction", "side"), (r"in profile", "direction", "side"),
    (r"profile view", "direction", "side"), (r"side view", "direction", "side"),
    (r"from the side", "direction", "side"), (r"side angle", "direction", "side"),
    (r"(?:front )?three[- ]quarters? (?:view|angle|profile)", "direction", "front_3q"),
    (r"3/4 (?:view|angle|profile)", "direction", "front_3q"), (r"45[- ]degree angle", "direction", "front_3q"),
    (r"diagonal angle", "direction", "front_3q"),
    (r"front view", "direction", "front"), (r"from the front", "direction", "front"),
    (r"straight[- ]on (?:front )?(?:view|shot|angle)", "direction", "front"), (r"frontal view", "direction", "front"),
    (r"head[- ]on view", "direction", "front"),
    # 높이 (웜즈아이 · 탑뷰를 먼저: 'very low angle' 이 로우앵글로 걸리지 않게)
    (r"worm'?s?[- ]eye(?: view| angle)?", "height", "worm"), (r"ground[- ]level (?:shot|angle|view)", "height", "worm"),
    (r"from (?:near )?the ground looking up", "height", "worm"), (r"(?:very|extreme(?:ly)?) low angle", "height", "worm"),
    (r"overhead (?:view|shot|angle)", "height", "overhead"), (r"top[- ]down(?: view| shot| angle)?", "height", "overhead"),
    (r"bird'?s?[- ]eye(?: view)?", "height", "overhead"), (r"from directly above", "height", "overhead"),
    (r"low[- ](?:camera[- ])?angle(?: shot| view)?", "height", "low"), (r"from below", "height", "low"),
    (r"looking up at (?:her|him|them|the subject)", "height", "low"), (r"from beneath", "height", "low"),
    (r"high[- ](?:camera[- ])?angle(?: shot| view)?", "height", "high"), (r"from above", "height", "high"),
    (r"looking down at (?:her|him|them|the subject)", "height", "high"),
    (r"eye[- ]level(?: shot| view| angle| framing)?", "height", "eye_level"),
    (r"dutch (?:angle|tilt)", "dutch", "dutch"), (r"canted angle", "dutch", "dutch"),
    (r"tilted (?:camera|frame|angle|horizon)", "dutch", "dutch"),
    (r"facing (?:the )?(?:screen|frame) left", "facing", "left"), (r"facing (?:the )?(?:screen|frame) right", "facing", "right"),
    (r"(?:facing|turned toward) the left (?:side )?of the frame", "facing", "left"),
    (r"(?:facing|turned toward) the right (?:side )?of the frame", "facing", "right"),
)
_EN_RES = [(re.compile(r"\b" + p + r"(?![\w'])", re.I), axis, v) for p, axis, v in _EN_TERMS]
# '뒤에서 안는' · 'hugging her from behind' 처럼 상호작용이지 시점이 아닌 말
_EN_NOT_VIEW = re.compile(r"\b(?:(?:hug\w*|embrac\w*|holding|held|grab\w*|taken|penetrat\w*|kiss\w*|fuck\w*|"
                          r"spoon\w*|approach\w*|push\w*)\s+(?:her\s+|him\s+|them\s+)?from behind|"
                          r"(?:light\w*|lit|sun\w*|glow\w*|shin\w*|illuminat\w*|beams?|rays?)\s+(?:\w+\s+)?"
                          r"from (?:above|below|behind|beneath|the side))\b", re.I)


def _find(text: str, english: bool) -> List[Tuple[int, str, str, str]]:
    """[(위치, 축, 값, 찾은 말), ...]"""
    found: List[Tuple[int, str, str, str]] = []
    if not text:
        return found
    buf = text
    if english:
        buf = _EN_NOT_VIEW.sub(lambda m: " " * len(m.group(0)), buf)
        for rx, axis, v in _EN_RES:
            for m in rx.finditer(buf):
                found.append((m.start(), axis, v, m.group(0)))
            buf = rx.sub(lambda m: " " * len(m.group(0)), buf)
    else:
        for term, axis, v in _KO_TERMS:
            start = 0
            while True:
                i = buf.find(term, start)
                if i < 0:
                    break
                found.append((i, axis, v, term))
                buf = buf[:i] + " " * len(term) + buf[i + len(term):]
                start = i + len(term)
    return sorted(found)


def detect(original: str = "", english: str = "") -> Tuple[Dict[str, Optional[str]], List[str]]:
    """요청에서 시점을 찾습니다.

    반환: ``({direction, height, dutch, facing}, 영어 요청에서 찾은 시점 말)``. 같은 축이 둘 이상이면 마지막에 말한 쪽.
    """
    hits = _find(original, english=False) if original else []
    hits += [(10_000 + p, a, v, w) for p, a, v, w in _find(original, english=True)] if original else []
    en_hits = _find(english, english=True) if english else []
    if not hits:
        hits = en_hits
    out: Dict[str, Optional[str]] = {"direction": None, "height": None, "dutch": None, "facing": None}
    for _, axis, v, _ in sorted(hits):
        out[axis] = v
    return out, [h[3] for h in en_hits]


_KO_KEEP = frozenset(("뒤돌아보", "어깨 너머로 돌아보"))


def ko_to_english(text: str) -> str:
    """번역 전에 한국어 시점 말을 영어로 바꿉니다 ('뒤 사선' · '부감' 처럼 번역기가 엉뚱하게 옮기는 말 방지)."""
    if not text:
        return text
    en = {"front": " front view ", "front_3q": " three-quarter view ", "side": " side profile view ",
          "rear_3q": " rear three-quarter view ", "rear": " rear view ", "eye_level": " eye-level shot ",
          "high": " high-angle shot ", "overhead": " overhead view ", "low": " low-angle shot ",
          "worm": " worm's-eye view ", "dutch": " dutch angle ", "left": " facing screen left ",
          "right": " facing screen right "}
    out = text
    for term, _, v in _KO_TERMS:
        if term in _KO_KEEP:  # '뒤돌아보는' 은 번역기가 'looking back' 으로 잘 옮김 (동작 말이 사라지지 않게)
            continue
        out = out.replace(term, en[v])
    return re.sub(r"\s{2,}", " ", out).strip()


def view_in_text(text: str) -> Dict[str, Optional[str]]:
    """요소 한 줄이 말하는 시점 ('low-angle shot' → {'height': 'low'})."""
    out: Dict[str, Optional[str]] = {}
    for _, axis, v, _ in _find(text or "", english=True):
        out.setdefault(axis, v)
    return out


def states_view(text: str) -> bool:
    """카메라 줄이 시점(방향·높이·기울기)을 말하는지 — 시점 문구를 따로 넣을 때 그런 카메라 줄은 뺌."""
    found = view_in_text(text)
    return bool(found) or bool(re.search(r"\bover[- ]the[- ]shoulder\b", text or "", re.I))


def route_phrase(text: str) -> Optional[Tuple[str, str]]:
    """분석 문구가 시점만 말하면 ``('view', '문구 #시점')``. 렌즈·구도가 함께 있으면 None."""
    from . import framing as framing_mod
    t = (text or "").strip()
    if not t or framing_mod.framing_in_text(t) or re.search(r"\b\d{2,3}\s?mm\b|\bf/\d|\blens\b", t, re.I):
        return None
    found = view_in_text(t)
    tag = found.get("direction") or found.get("height") or found.get("dutch")
    if tag and len(t.split()) <= 14:
        return "view", f"{t} #{tag}"
    return None
