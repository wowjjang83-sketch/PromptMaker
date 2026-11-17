"""사용자 고정 규칙: 허용 국적 · 여성 나이 범위.

기본값은 국적 Korean · Japanese, 여성 나이 20~22세입니다. 사용자 데이터 폴더의
``restrictions.json`` 에 저장되므로 라이브러리를 다시 설치(업데이트)해도 유지됩니다.

    {"nationalities": ["Korean", "Japanese"], "female_age_min": 20, "female_age_max": 22}

- 생성: 국적·나이 후보를 규칙 안의 값으로만 고르고, 요청이 규칙 밖 값을 말하면 규칙 안으로 바꾸고 경고를 남깁니다.
- 이미지 분석/학습 저장: ``nationality``/``age`` 요소 txt 에 규칙 밖 값은 저장하지 않습니다 (reason ``restricted``).
- 분석 풀프롬프트·학습할 프롬프트 문장 안의 ``25-year-old Chinese woman`` 같은 표현도 ``apply_to_text`` 로 바꿀 수 있습니다.
- 안전 규칙(``safety``, 20세 미만 금지)은 이 규칙과 별개로 항상 먼저 적용됩니다.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .safety import MIN_AGE

FILE_NAME = "restrictions.json"
DEFAULT_NATIONALITIES = ["Korean", "Japanese"]
DEFAULT_FEMALE_AGE_MIN = 20
DEFAULT_FEMALE_AGE_MAX = 22

# 문장 속 국적 형용사 (규칙 밖 값을 찾을 때 사용 · Asian 처럼 허용 국적과 함께 쓰는 말은 제외)
KNOWN_NATIONALITIES = (
    "Korean", "Japanese", "Chinese", "Taiwanese", "Thai", "Vietnamese", "Filipino", "Filipina", "Indonesian",
    "Malaysian", "Singaporean", "Mongolian", "Indian", "American", "Canadian", "Brazilian", "Mexican",
    "Argentinian", "Colombian", "French", "Italian", "Spanish", "Portuguese", "German", "Dutch", "Swedish",
    "Norwegian", "Danish", "Finnish", "Russian", "Ukrainian", "Polish", "Czech", "British", "English", "Irish",
    "Scottish", "Australian", "Greek", "Turkish", "Persian", "Iranian", "Arab", "Egyptian", "Israeli",
    "Nigerian", "Kenyan", "South African", "Ethiopian", "Hispanic", "Latina", "Latin", "Caucasian",
    "European", "Scandinavian", "Nordic", "Slavic", "African", "Hawaiian",
)
_FEMALE_NOUNS = r"(?:woman|women|girl|girls|lady|ladies|model|models|female|females|beauty|idol|actress)"


@dataclass
class Restrictions:
    nationalities: List[str] = field(default_factory=lambda: list(DEFAULT_NATIONALITIES))
    female_age_min: Optional[int] = DEFAULT_FEMALE_AGE_MIN
    female_age_max: Optional[int] = DEFAULT_FEMALE_AGE_MAX
    enabled: bool = True

    # ------------------------------------------------------------ io
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Restrictions":
        r = cls()
        if not isinstance(data, dict):
            return r
        if "enabled" in data:
            r.enabled = bool(data["enabled"])
        if "nationalities" in data:
            vals = data["nationalities"]
            if isinstance(vals, str):
                vals = re.split(r"[,\n]", vals)
            r.nationalities = [" ".join(str(v).split()) for v in vals or [] if str(v).strip()]
        for k in ("female_age_min", "female_age_max"):
            if k in data:
                v = data[k]
                setattr(r, k, int(v) if str(v).strip().lstrip("-").isdigit() else None)
        lo, hi = r.female_age_min, r.female_age_max
        if lo is not None:
            r.female_age_min = max(MIN_AGE, lo)
        if hi is not None:
            r.female_age_max = max(MIN_AGE, hi)
        if r.female_age_min is not None and r.female_age_max is not None and r.female_age_min > r.female_age_max:
            r.female_age_min, r.female_age_max = r.female_age_max, r.female_age_min
        return r

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, data_dir: Optional[str]) -> "Restrictions":
        """``<data_dir>/restrictions.json`` (없으면 기본값)."""
        if data_dir:
            path = os.path.join(data_dir, FILE_NAME)
            if os.path.isfile(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        return cls.from_dict(json.load(f))
                except (OSError, ValueError):
                    pass
        return cls()

    def save(self, data_dir: str) -> str:
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, FILE_NAME)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    # ------------------------------------------------------------ rules
    @property
    def nat_on(self) -> bool:
        return self.enabled and bool(self.nationalities)

    @property
    def age_on(self) -> bool:
        return self.enabled and (self.female_age_min is not None or self.female_age_max is not None)

    def _nat_match(self, value: str) -> Optional[str]:
        v = " ".join(str(value).split()).lower()
        for n in self.nationalities:
            if n.lower() == v:
                return n
        return None

    def allows(self, key: str, text: str) -> bool:
        """요소 txt 에 저장해도 되는 값인지 (nationality / age 만 검사)."""
        if key == "nationality" and self.nat_on:
            return self._nat_match(text) is not None
        if key == "age" and self.age_on:
            m = re.match(r"\s*(\d+)", str(text))
            return bool(m) and self.age_in_range(int(m.group(1)))
        return True

    def age_in_range(self, age: int) -> bool:
        lo = self.female_age_min if self.female_age_min is not None else MIN_AGE
        hi = self.female_age_max if self.female_age_max is not None else 10 ** 6
        return lo <= age <= hi

    def clamp_age(self, age: int) -> int:
        if not self.age_on:
            return age
        if self.female_age_min is not None and age < self.female_age_min:
            return self.female_age_min
        if self.female_age_max is not None and age > self.female_age_max:
            return self.female_age_max
        return age

    def nationality(self, value: str, rng: Optional[random.Random] = None) -> str:
        """규칙 안의 국적으로 (빈 값·규칙 밖 값이면 허용 목록에서 고름)."""
        if not self.nat_on:
            return value
        hit = self._nat_match(value) if value else None
        if hit:
            return hit
        return (rng or random).choice(self.nationalities)

    def filter_values(self, key: str, values: List[str]) -> List[str]:
        return [v for v in values if self.allows(key, v)]

    def apply_to_text(self, text: str, rng: Optional[random.Random] = None) -> str:
        """문장 속 여성 인물의 나이·국적을 규칙 안으로 바꿈.

        ``a 25-year-old Chinese woman`` → ``a 22-year-old Korean woman`` (남성 묘사는 그대로).
        """
        if not text or not self.enabled:
            return text
        rng = rng or random.Random(text)
        out = text
        if self.age_on:
            def _age(m: "re.Match") -> str:
                age = self.clamp_age(int(m.group(1)))
                return f"{age}{m.group(2)}"

            out = re.sub(r"\b(\d{2})(\s*-?\s*years?\s*-?\s*old\b(?=[^.]{0,40}?\b" + _FEMALE_NOUNS + r"\b))",
                         _age, out, flags=re.I)
            out = re.sub(r"(\b" + _FEMALE_NOUNS + r"\b[^.]{0,20}?\baged?\s+)(\d{2})\b",
                         lambda m: m.group(1) + str(self.clamp_age(int(m.group(2)))), out, flags=re.I)
        if self.nat_on:
            pat = "|".join(re.escape(n) for n in sorted(KNOWN_NATIONALITIES, key=len, reverse=True))
            rx = re.compile(r"\b(" + pat + r")\b(?=(?:[\s,-]+[\w-]+){0,3}?[\s,-]+" + _FEMALE_NOUNS + r"\b)", re.I)
            chosen: Dict[str, str] = {}

            def _nat(m: "re.Match") -> str:
                word = m.group(1)
                if self._nat_match(word):
                    return word
                key = word.lower()
                if key not in chosen:
                    chosen[key] = self.nationality("", rng)
                rep = chosen[key]
                return rep if word[:1].isupper() else rep.lower()

            out = rx.sub(_nat, out)
        return out


def load(data_dir: Optional[str]) -> Restrictions:
    return Restrictions.load(data_dir)


def clean_element_files(data_dir: str, rules: Optional[Restrictions] = None,
                        backup_dir: Optional[str] = None) -> Dict[str, Any]:
    """``<data_dir>/elements/nationality.txt`` · ``age.txt`` 에서 규칙 밖 줄을 지움 (먼저 백업).

    반환: {"removed": {key: [text]}, "backup": 백업 폴더 또는 ""}. 지울 줄이 없으면 파일을 건드리지 않습니다.
    """
    import datetime
    import shutil

    rules = rules or Restrictions.load(data_dir)
    out: Dict[str, Any] = {"removed": {}, "backup": ""}
    if not rules.enabled:
        return out
    plans: Dict[str, List[str]] = {}
    for key in ("nationality", "age"):
        path = os.path.join(data_dir, "elements", key + ".txt")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
        keep, removed = [], []
        for line in lines:
            body = line.strip()
            if not body or body.startswith("#"):
                keep.append(line)
                continue
            text = body.split("::", 1)[1] if re.match(r"^\d+(?:\.\d+)?::", body) else body
            if rules.allows(key, text.strip()):
                keep.append(line)
            else:
                removed.append(text.strip())
        if removed:
            plans[path] = keep
            out["removed"][key] = removed
    if not plans:
        return out
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = backup_dir or os.path.join(data_dir, "backup", f"restrictions-{stamp}")
    os.makedirs(backup, exist_ok=True)
    for path, keep in plans.items():
        shutil.copy2(path, os.path.join(backup, os.path.basename(path)))
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(keep) + "\n")
    out["backup"] = backup
    return out
