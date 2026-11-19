"""자위 장면 (0.13.0) — 성인 여성 혼자 하는 자위 동작.

* 요청에서 찾기: :func:`detect` ('자위', '스스로 만지는', 'masturbating', 'touching herself' …).
  '가슴' / '클리토리스' / '손가락·질' 같은 말이 있으면 그 종류의 동작을 우선합니다 (:func:`line_kinds`).
* 동작 문구는 ``elements/masturbation.txt`` (사용자 편집 가능, 현재분사구, ``{her}`` 사용).
* 자위 장면에는 서 있거나 걷는 포즈를 쓰지 않습니다 (:data:`POSE_BAD_RE`). 맞는 포즈가 없으면
  :data:`FALLBACK_POSES` 에서 고릅니다. 표정은 쾌감 표정을 우선합니다 (:data:`EXPR_GOOD_RE`).
"""
from __future__ import annotations

import re
from typing import List, Set, Tuple

KINDS = ("breast", "clit", "finger")
LABELS_KO = {"breast": "가슴 주무르기", "clit": "클리토리스 만지기", "finger": "질에 손가락 넣기"}

_KO_RE = re.compile(r"자위|스스로\s*(?:를)?\s*(?:만지|애무|위로)|혼자\s*(?:서)?\s*(?:만지|애무)|손장난|핑거링|"
                    r"(?:가슴|유방|젖가슴|유두|젖꼭지)\s*(?:을|를)?\s*(?:주무르|만지|애무|비비|꼬집)|"
                    r"클리(?:토리스)?|음핵|질\s*(?:에|속|안)\s*(?:에)?\s*손가락|손가락\s*(?:을|를)?\s*(?:질|넣)")
_EN_RE = re.compile(r"\bmasturbat\w*|\b(?:touching|pleasuring|pleasing|fingering|stimulating|caressing) "
                    r"(?:herself|themselves|her (?:pussy|clit|clitoris|vagina))\b|\bplaying with herself\b|"
                    r"\b(?:fondling|groping|kneading|squeezing|massaging|rubbing|caressing) (?:her|their) "
                    r"(?:own )?(?:breasts?|boobs?|chest|nipples?)\b|\bclit(?:oris)?\b|"
                    r"\bfingers? (?:inside|in) (?:her|herself)\b", re.I)
_KIND_KO = {"breast": re.compile(r"가슴|유방|젖가슴|유두|젖꼭지"),
            "clit": re.compile(r"클리|음핵"),
            "finger": re.compile(r"손가락|핑거링|질\s*(?:에|속|안)")}
_KIND_EN = {"breast": re.compile(r"\b(?:breasts?|boobs?|nipples?|chest)\b", re.I),
            "clit": re.compile(r"\bclit(?:oris)?\b", re.I),
            "finger": re.compile(r"\b(?:fingering|fingers? (?:inside|in)|vagina|pussy|inside her)\b", re.I)}
# 쓴 영어 단어 (요소 키워드 검색 · 'The scene also features' 에 다시 나오지 않게)
CONSUMED = ("masturbating", "masturbate", "masturbates", "masturbation", "masturbate", "herself", "touching",
            "pleasuring", "pleasing", "fingering", "finger", "fingers", "clitoris", "clit", "breast", "breasts",
            "boobs", "nipple", "nipples", "vagina", "pussy", "inside", "fondling", "groping", "kneading",
            "squeezing", "massaging", "rubbing", "caressing", "stimulating", "playing", "chest", "own")

# 자위 장면에 쓰지 않는 포즈 (서 있기 · 이동 · 춤 …)
POSE_BAD_RE = re.compile(r"\b(?:standing|stands|walking|walks|running|jumping|dancing|twirling|spinning|strolling|"
                         r"wading|striding|crouching|squatting|leaning (?:against|on) a (?:wall|railing)|"
                         r"climbing|hugging (?:her|\{her\}) knees|crossed|cross-legged)\b", re.I)
POSE_GOOD_RE = re.compile(r"\b(?:lying|laying|reclining|reclined|lounging|sprawled|propped|kneeling|perched|"
                          r"sitting|seated|leaning back|on (?:her|\{her\}) back|on (?:her|\{her\}) side|"
                          r"legs (?:spread|apart|open)|knees (?:apart|raised|bent|up))\b", re.I)
FALLBACK_POSES = (
    "lying on {her} back with {her} knees bent and apart",
    "reclining against the pillows with {her} legs spread",
    "sitting back with {her} legs apart",
    "lying on {her} side with one knee raised",
)
# 장소를 말하지 않았을 때 우선할 곳 (실내 사적인 곳)
PRIVATE_PLACE_RE = re.compile(r"\b(?:bed|beds|bedroom|bedding|sheets|sofa|couch|hotel|room|suite|apartment|bathtub|bath|"
                              r"shower|futon|mattress|pillows?|loft|studio apartment|indoors?)\b", re.I)
# 쾌감 표정 우선, 밝은 웃음은 덜
EXPR_GOOD_RE = re.compile(r"\b(?:parted|pleasure|flushed|blush\w*|moan\w*|ecsta\w*|bliss\w*|aroused|breathless|"
                          r"half-closed|half-lidded|closed eyes|eyes closed|eyes shut|biting (?:her|\{her\}) (?:lower )?lip|"
                          r"mouth open|panting|dazed|lustful|heavy-lidded|sultry|sensual)\b", re.I)
EXPR_BAD_RE = re.compile(r"\b(?:smil\w*|laugh\w*|grin\w*|cheerful|radiant|playful|bright|wink\w*|confident|"
                         r"neutral|serious|surprised|shy)\b", re.I)
# 손이 바쁜 포즈 · 동작 (자위 동작과 함께 못 함) — undressing.BUSY_RE 와 함께 씀
HANDS_BUSY_RE = re.compile(r"\b(?:holding|holds|hand on (?:her|\{her\}) hip|hands? in (?:her|\{her\}) hair|"
                           r"arms? (?:crossed|above)|stretching)\b", re.I)


def detect(original: str = "", english: str = "") -> Tuple[bool, List[str]]:
    """요청이 자위 장면인지와, 말한 동작 종류 (breast / clit / finger)."""
    found = bool((original and _KO_RE.search(original)) or (english and _EN_RE.search(english)))
    if not found:
        return False, []
    kinds = [k for k in KINDS if (original and _KIND_KO[k].search(original))
             or (english and _KIND_EN[k].search(english))]
    return True, kinds


def line_kinds(text: str) -> Set[str]:
    """동작 문구 한 줄이 어떤 종류인지."""
    return {k for k in KINDS if _KIND_EN[k].search(text or "")}
