"""Loading of user-editable templates and element lists (JSON / TXT).

Element files
-------------
``elements/<key>.txt``            one item per line, ``#`` starts a comment,
                                   ``3::text`` gives the item weight 3.
``elements/<key>.<suffix>.txt``   (예전 수위 파일: ``.suggestive`` / ``.explicit``) 수위 구분은
                                   0.4.0 에서 없어졌습니다. 남아 있으면 ``<key>`` 요소에 그대로 합쳐
                                   읽고, ``merge_legacy_rating_files()`` 로 ``<key>.txt`` 에 합칠 수 있습니다.
``elements/<key>.json``           ``{"label": "...", "items": ["text", {"text": "...", "weight": 2}, ...]}``
                                   items may also carry extra fields (e.g. casting counts).

Template files
--------------
``templates/<id>.json``           see docs/GUIDE.md for the schema.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .safety import MIN_AGE, is_safe_item

# 0.4.0 부터 수위(content rating) 구분이 없습니다. 아래 접미사는 예전 파일 이름을 읽기 위해서만 씁니다.
LEGACY_RATING_SUFFIXES = ("sfw", "suggestive", "explicit")
PACKAGE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass
class Item:
    text: str
    weight: float = 1.0
    meta: Dict[str, Any] = field(default_factory=dict)
    tags: set = field(default_factory=set)       # "#tag": added to the scene context
    requires: set = field(default_factory=set)   # "+tag": needs one of these in the context
    excludes: set = field(default_factory=set)   # "!tag": not allowed with these
    explicit_tags: bool = False                  # True when the user wrote tags by hand


@dataclass
class Element:
    key: str
    label: str = ""
    description: str = ""
    items: List[Item] = field(default_factory=list)
    default: Optional[str] = None


@dataclass
class Template:
    id: str
    name: str
    type: str
    description: str = ""
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    casting_filter: Dict[str, int] = field(default_factory=dict)
    defaults: Dict[str, str] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    def element_keys(self) -> List[str]:
        import re

        keys: List[str] = []
        for s in self.sentences:
            for m in re.finditer(r"\{([a-zA-Z_][\w]*)(?::\d+)?\??\}", s.get("text", "")):
                k = m.group(1)
                if k not in keys:
                    keys.append(k)
        return keys


def _parse_txt_line(line: str) -> Optional[Item]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    weight = 1.0
    if "::" in line:
        head, tail = line.split("::", 1)
        try:
            weight = float(head.strip())
            line = tail.strip()
        except ValueError:
            pass
    item = Item(text=line, weight=weight)
    _parse_tag_tokens(item)
    return item


_TAG_TOKEN = re.compile(r"\s([#+!])([a-z][a-z0-9_]*)(?=\s|$)")


def _parse_tag_tokens(item: Item) -> None:
    """Split trailing ``#tag +tag !tag`` tokens off the item text."""
    text = " " + item.text
    found = False
    for sign, tag in _TAG_TOKEN.findall(text):
        found = True
        {"#": item.tags, "+": item.requires, "!": item.excludes}[sign].add(tag)
    if found:
        item.text = _TAG_TOKEN.sub(" ", text).strip()
        item.text = re.sub(r"\s{2,}", " ", item.text)
        item.explicit_tags = True


def _split_name(filename: str):
    stem, ext = os.path.splitext(filename)
    parts = stem.split(".")
    return parts[0], ext.lower()


def load_elements(dirs: Iterable[str]) -> Dict[str, Element]:
    elements: Dict[str, Element] = {}
    for base in dirs:
        edir = os.path.join(base, "elements")
        if not os.path.isdir(edir):
            continue
        for fname in sorted(os.listdir(edir)):
            key, ext = _split_name(fname)
            if ext not in (".txt", ".json"):
                continue
            path = os.path.join(edir, fname)
            el = elements.setdefault(key, Element(key=key, label=key))
            if ext == ".txt":
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        item = _parse_txt_line(line)
                        if item:
                            el.items.append(item)
            else:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("replace"):
                    el.items = []
                el.label = data.get("label", el.label)
                el.description = data.get("description", el.description)
                el.default = data.get("default", el.default)
                for raw in data.get("items", []):
                    if isinstance(raw, str):
                        el.items.append(Item(text=raw))
                    else:
                        raw = dict(raw)
                        text = raw.pop("text")
                        weight = float(raw.pop("weight", 1.0))
                        raw.pop("rating", None)  # 예전 데이터 호환 (무시)
                        tags = set(raw.pop("tags", []))
                        req = set(raw.pop("requires", []))
                        exc = set(raw.pop("excludes", []))
                        it = Item(text=text, weight=weight, meta=raw,
                                  tags=tags, requires=req, excludes=exc,
                                  explicit_tags=bool(tags or req or exc))
                        _parse_tag_tokens(it)
                        el.items.append(it)
    # Drop duplicates, anything that refers to minors, ages under 20 and tag dumps.
    for el in elements.values():
        seen = set()
        clean = []
        for item in el.items:
            item.text = normalize_phrase(el.key, item.text)
            k = item.text.lower()
            if k in seen or not is_safe_item(item.text) or not is_usable_item(el.key, item.text):
                continue
            seen.add(k)
            clean.append(item)
        el.items = clean
    return elements


def legacy_rating_files(data_dir: str) -> List[str]:
    """``<data_dir>/elements`` 안의 예전 수위 파일(``<key>.suggestive.txt`` 등) 경로 목록."""
    edir = os.path.join(data_dir, "elements")
    if not os.path.isdir(edir):
        return []
    out = []
    for fname in sorted(os.listdir(edir)):
        stem, ext = os.path.splitext(fname)
        parts = stem.split(".")
        if ext.lower() == ".txt" and len(parts) == 2 and parts[1] in LEGACY_RATING_SUFFIXES:
            out.append(os.path.join(edir, fname))
    return out


def merge_legacy_rating_files(data_dir: str, delete: bool = True) -> Dict[str, Any]:
    """예전 수위 파일을 ``<key>.txt`` 하나로 합칩니다 (수위 구분 통합, 0.4.0).

    ``<key>.suggestive.txt`` / ``<key>.explicit.txt`` / ``<key>.sfw.txt`` 의 항목 줄을
    ``<key>.txt`` 끝에 ``# 수위 통합: <파일> 에서 옮김`` 주석과 함께 덧붙이고 (이미 있는 줄은
    건너뜀), 원래 파일은 지웁니다 (``delete=False`` 면 남겨 둠). 여러 번 실행해도 안전합니다.

    반환: ``{"merged": {"attire_state.suggestive.txt": 3, ...}, "files": [...], "added": 5}``
    """
    report: Dict[str, Any] = {"merged": {}, "files": [], "added": 0}
    for path in legacy_rating_files(data_dir):
        fname = os.path.basename(path)
        key = fname.split(".")[0]
        target = os.path.join(os.path.dirname(path), f"{key}.txt")
        existing = set()
        if os.path.exists(target):
            with open(target, encoding="utf-8") as f:
                existing = {ln.strip().lower() for ln in f if ln.strip() and not ln.strip().startswith("#")}
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f]
        new = []
        for ln in lines:
            if not ln or ln.startswith("#") or ln.lower() in existing:
                continue
            existing.add(ln.lower())
            new.append(ln)
        if new:
            prefix = ""
            if os.path.exists(target):
                with open(target, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    if f.tell():
                        f.seek(-1, os.SEEK_END)
                        prefix = "" if f.read(1) == b"\n" else "\n"
            with open(target, "a", encoding="utf-8", newline="\n") as f:
                f.write(prefix + f"# 수위 통합: {fname} 에서 옮김\n" + "\n".join(new) + "\n")
        if delete:
            os.remove(path)
        report["merged"][fname] = len(new)
        report["files"].append(target)
        report["added"] += len(new)
    report["files"] = sorted(set(report["files"]))
    return report


def looks_like_tag_list(text: str) -> bool:
    """True for comma-separated keyword dumps ("Short, Firm, Square Face, Fair Skin, ...").

    Such lines are tags, not sentence fragments, and would break the grammar of a
    template sentence. Normal items have at most a couple of commas inside a phrase.
    """
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) < 4 or any(p.lower().startswith(("and ", "or ")) for p in parts):
        return False  # a normal English list ("red, gold, and black")
    short = sum(1 for p in parts if len(p.split()) <= 3)
    titled = sum(1 for p in parts if p[:1].isupper())
    return short >= len(parts) * 0.75 and (titled >= len(parts) * 0.5 or len(parts) >= 6)


PHRASE_KEYS = ("pose", "action", "motion", "interaction", "expression", "attire_state")


def normalize_phrase(key: str, text: str) -> str:
    """Turn a full sentence into the fragment a template expects.

    "Standing at the edge of a rooftop, she gazes at the city below." ->
    "standing at the edge of a rooftop"
    """
    if key not in PHRASE_KEYS:
        return text
    t = text.strip().rstrip(".!")
    m = re.match(r"^((?:[A-Za-z]+ly\s+)?[A-Za-z][\w'-]*ing\b[^,]*?),\s*(?:she|he|they|her|his|their)\b", t)
    if m:
        t = m.group(1)
    if len(t) > 1 and t[0].isupper() and t[1].islower() and not t.startswith(("I ", "Japanese", "Korean")):
        t = t[0].lower() + t[1:]
    return t


def is_usable_item(key: str, text: str) -> bool:
    """Element-level sanity checks used when loading (and when appending new lines)."""
    if looks_like_tag_list(text):
        return False
    if key == "age":
        m = re.match(r"\s*(\d+)", text)
        if m and int(m.group(1)) < MIN_AGE:
            return False
    return True


def load_templates(dirs: Iterable[str]) -> Dict[str, Template]:
    templates: Dict[str, Template] = {}
    for base in dirs:
        tdir = os.path.join(base, "templates")
        if not os.path.isdir(tdir):
            continue
        for fname in sorted(os.listdir(tdir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(tdir, fname), encoding="utf-8") as f:
                data = json.load(f)
            tpl = Template(
                id=data.get("id", os.path.splitext(fname)[0]),
                name=data.get("name", fname),
                type=data.get("type", "image"),
                description=data.get("description", ""),
                sentences=data.get("sentences", []),
                casting_filter=data.get("casting_filter", {}),
                defaults=data.get("defaults", {}),
                raw=data,
            )
            templates[tpl.id] = tpl
    return templates


def load_json(dirs: Iterable[str], relpath: str) -> Dict[str, Any]:
    """Merge a JSON object found at ``relpath`` in every data dir."""
    merged: Dict[str, Any] = {}
    for base in dirs:
        path = os.path.join(base, relpath)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                merged.update({k: v for k, v in json.load(f).items() if not k.startswith("_")})
    return merged


def load_text_files(dirs: Iterable[str], subdir: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for base in dirs:
        d = os.path.join(base, subdir)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith((".txt", ".md")):
                with open(os.path.join(d, fname), encoding="utf-8") as f:
                    out[os.path.splitext(fname)[0]] = f.read().strip()
    return out


def load_model_profiles(dirs: Iterable[str]):
    """``model_profiles/<id>.json`` from every data dir (later dirs override)."""
    from .model_profiles import ModelProfile

    out = {}
    for base in dirs:
        d = os.path.join(base, "model_profiles")
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            if fname.endswith(".json"):
                with open(os.path.join(d, fname), encoding="utf-8") as f:
                    data = json.load(f)
                pid = fname[:-5]
                out[data.get("id", pid)] = ModelProfile.from_dict(pid, data)
    return out
