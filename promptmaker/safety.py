"""Safety rules: every generated character must be an adult (20+)."""
from __future__ import annotations

import re
from typing import Optional

MIN_AGE = 20

# Terms that imply a minor. Inputs or data items containing them are rejected.
_MINOR_TERMS = [
    r"child", r"children", r"kid", r"kids", r"minor", r"minors", r"underage",
    r"under-age", r"teen", r"teens", r"teenage", r"teenager", r"teenagers",
    r"preteen", r"pre-teen", r"loli", r"lolita", r"shota", r"toddler", r"infant",
    r"baby-faced girl", r"schoolgirl", r"schoolboy", r"school girl", r"school boy",
    r"junior high", r"middle school", r"elementary school", r"high school student",
    r"young girl", r"little girl", r"little boy", r"youngster", r"juvenile",
    r"pubescent", r"prepubescent", r"adolescent", r"jailbait",
    r"(?:1[0-9]|[1-9])[- ]?(?:year|yr)s?[- ]?old",
]
_MINOR_RE = re.compile(r"\b(?:" + "|".join(_MINOR_TERMS) + r")\b", re.IGNORECASE)

# Korean / Japanese minor-related words checked before translation.
_MINOR_CJK = [
    "어린이", "아동", "미성년", "초등학생", "중학생", "고등학생", "여고생", "남고생",
    "여중생", "남중생", "청소년", "소녀", "소년", "유아", "아기", "로리", "쇼타",
    "女子高生", "女子中学生", "小学生", "中学生", "高校生", "未成年", "児童", "ロリ",
]

_AGE_RE = re.compile(r"\b(\d{1,3})[- ]?(?:year|yr)s?[- ]?old\b", re.IGNORECASE)
_KO_AGE_RE = re.compile(r"(\d{1,3})\s*(?:살|세)")


class SafetyError(ValueError):
    """Raised when an input would describe a minor."""


def find_minor_reference(text: str) -> Optional[str]:
    if not text:
        return None
    m = _MINOR_RE.search(text)
    if m:
        return m.group(0)
    for word in _MINOR_CJK:
        if word in text:
            return word
    for m in list(_AGE_RE.finditer(text)) + list(_KO_AGE_RE.finditer(text)):
        if int(m.group(1)) < MIN_AGE:
            return m.group(0)
    return None


def check_text(text: str) -> None:
    hit = find_minor_reference(text)
    if hit:
        raise SafetyError(
            f"미성년자를 연상시키는 표현('{hit}')은 사용할 수 없습니다. "
            f"모든 인물은 {MIN_AGE}세 이상 성인이어야 합니다."
        )


def is_safe_item(text: str) -> bool:
    return find_minor_reference(text) is None


def clamp_age(value) -> int:
    try:
        age = int(str(value).strip().split("-")[0])
    except (TypeError, ValueError):
        return MIN_AGE
    return max(MIN_AGE, min(age, 80))
