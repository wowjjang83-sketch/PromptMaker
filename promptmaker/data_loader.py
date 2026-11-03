"""Loading of user-editable templates and element lists (JSON / TXT).

Element files
-------------
``elements/<key>.txt``            one item per line, ``#`` starts a comment,
                                   ``3::text`` gives the item weight 3.
``elements/<key>.<rating>.txt``   same, but only used when the content rating
                                   option is at least ``<rating>``
                                   (``sfw`` < ``suggestive`` < ``explicit``).
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

RATINGS = ["sfw", "suggestive", "explicit"]
PACKAGE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass
class Item:
    text: str
    weight: float = 1.0
    rating: str = "sfw"
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

    def items_for(self, rating: str) -> List[Item]:
        limit = RATINGS.index(rating) if rating in RATINGS else 0
        return [i for i in self.items if RATINGS.index(i.rating) <= limit]


@dataclass
class Template:
    id: str
    name: str
    type: str
    description: str = ""
    sentences: List[Dict[str, Any]] = field(default_factory=list)
    casting_filter: Dict[str, int] = field(default_factory=dict)
    defaults: Dict[str, str] = field(default_factory=dict)
    rating: str = "sfw"
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
    key = parts[0]
    rating = parts[1] if len(parts) > 1 and parts[1] in RATINGS else "sfw"
    return key, rating, ext.lower()


def load_elements(dirs: Iterable[str]) -> Dict[str, Element]:
    elements: Dict[str, Element] = {}
    for base in dirs:
        edir = os.path.join(base, "elements")
        if not os.path.isdir(edir):
            continue
        for fname in sorted(os.listdir(edir)):
            key, rating, ext = _split_name(fname)
            if ext not in (".txt", ".json"):
                continue
            path = os.path.join(edir, fname)
            el = elements.setdefault(key, Element(key=key, label=key))
            if ext == ".txt":
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        item = _parse_txt_line(line)
                        if item:
                            item.rating = rating
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
                        el.items.append(Item(text=raw, rating=rating))
                    else:
                        raw = dict(raw)
                        text = raw.pop("text")
                        weight = float(raw.pop("weight", 1.0))
                        r = raw.pop("rating", rating)
                        tags = set(raw.pop("tags", []))
                        req = set(raw.pop("requires", []))
                        exc = set(raw.pop("excludes", []))
                        it = Item(text=text, weight=weight, rating=r, meta=raw,
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
    m = re.match(r"^([A-Za-z][\w'-]*ing\b[^,]*?),\s*(?:she|he|they|her|his|their)\b", t)
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
                rating=data.get("rating", "sfw"),
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
