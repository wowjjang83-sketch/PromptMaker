"""Scene context, auto-tagging and relation rules (coherence engine).

Every picked item adds its tags to the *scene context*. The next element's
candidates are filtered and weighted against that context:

* hard rules  -> conflicts, ``!tag`` excludes, unmet ``+tag`` requirements => weight 0
* soft rules  -> affinity multipliers (``relations.json``) => conditional probability

Rules live in ``lexicon/relations.json`` and ``lexicon/tag_keywords.json`` so the
user can read and edit them; items may also carry ``#tag +tag !tag`` tokens.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .data_loader import Element, Item

# Elements whose items describe the scene itself: props they mention are *provided*.
DEFAULT_PROVIDERS = ("location", "time_weather")
MAX_BOOST = 20.0  # cap on the combined affinity multiplier
# Elements never auto-tagged (their words say little about the scene).
DEFAULT_SKIP = ("hair", "makeup", "skin", "beauty", "nationality", "age", "male_look",
                "expression", "casting", "mood")


def _compile(words: Iterable[str]) -> Optional[re.Pattern]:
    words = [w for w in words if w]
    if not words:
        return None
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
                      + r")", re.I)


@dataclass
class Relations:
    implies: Dict[str, List[str]] = field(default_factory=dict)
    conflicts: Set[Tuple[str, str]] = field(default_factory=set)
    affinity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tag_words: Dict[str, re.Pattern] = field(default_factory=dict)
    prop_words: Dict[str, re.Pattern] = field(default_factory=dict)
    providers: Tuple[str, ...] = DEFAULT_PROVIDERS
    skip: Tuple[str, ...] = DEFAULT_SKIP
    garments: Tuple[str, ...] = ()          # garment elements: the props they mention are worn
    enforced: Set[str] = field(default_factory=set)  # props required even when the scene is unknown

    @classmethod
    def load(cls, relations: dict, keywords: dict) -> "Relations":
        conflicts = set()
        for pair in relations.get("conflicts", []):
            if len(pair) == 2:
                a, b = pair
                conflicts.add((a, b))
                conflicts.add((b, a))
        return cls(
            implies={k: list(v) for k, v in relations.get("implies", {}).items()},
            conflicts=conflicts,
            affinity={k: {t: float(m) for t, m in v.items()}
                      for k, v in relations.get("affinity", {}).items()},
            tag_words={t: p for t, p in ((t, _compile(w)) for t, w in keywords.get("tags", {}).items()) if p},
            prop_words={t: p for t, p in ((t, _compile(w)) for t, w in keywords.get("props", {}).items()) if p},
            providers=tuple(keywords.get("provider_elements", DEFAULT_PROVIDERS)),
            skip=tuple(keywords.get("skip_elements", DEFAULT_SKIP)),
            garments=tuple(keywords.get("garment_elements", ())),
            enforced=set(keywords.get("enforced_props", ())),
        )

    # ------------------------------------------------------------ tagging
    def closure(self, tags: Iterable[str]) -> Set[str]:
        out = set(tags)
        stack = list(out)
        while stack:
            t = stack.pop()
            for implied in self.implies.get(t, []):
                if implied not in out:
                    out.add(implied)
                    stack.append(implied)
        return out

    def tags_in_text(self, text: str) -> Set[str]:
        return {t for t, p in self.tag_words.items() if p.search(text)}

    def props_in_text(self, text: str) -> Set[str]:
        return {t for t, p in self.prop_words.items() if p.search(text)}

    def auto_tag(self, elements: Dict[str, Element]) -> None:
        """Infer tags for items. Hand-written tags are kept and extended."""
        for key, el in elements.items():
            if key in self.skip:
                continue
            provider = key in self.providers or key in self.garments
            for item in el.items:
                if item.meta.get("auto_tagged"):
                    continue
                text = item.text
                item.tags |= self.tags_in_text(text)
                props = self.props_in_text(text)
                if provider:
                    item.tags |= props
                elif not item.explicit_tags:
                    # props a pose/action mentions must exist in the scene
                    item.requires |= props
                item.meta["auto_tagged"] = True

    def tags_for_request(self, text: str) -> Set[str]:
        return self.tags_in_text(text) | self.props_in_text(text)

    # ------------------------------------------------------------ scoring
    def explain(self, item: Item, context: Set[str], scene_known: bool) -> Tuple[float, str]:
        """Return (multiplier, reason). 0 means the item is not allowed."""
        tags = self.closure(item.tags)
        bad = item.excludes & context
        if bad:
            return 0.0, f"제외(!{sorted(bad)[0]})"
        for t in tags:
            for c in context:
                if (t, c) in self.conflicts:
                    return 0.0, f"충돌({c}✕{t})"
        missing = item.requires - context
        if missing and (scene_known or missing & self.enforced):
            return 0.0, f"필요(+{'/'.join(sorted(missing))}) 없음"
        # soft rules: each of the item's own tags is scaled once, by the strongest boost
        # and the strongest penalty found in the scene (no runaway compounding)
        mult, parts = 1.0, []
        for t in sorted(item.tags):
            ups = [(self.affinity[c][t], c) for c in context if t in self.affinity.get(c, {})]
            if not ups:
                continue
            hi, lo = max(ups), min(ups)
            for f, c in ([hi] if hi[0] > 1.0 else []) + ([lo] if lo[0] < 1.0 else []):
                mult *= f
                parts.append(f"{c}→{t}×{f:g}")
        mult = min(mult, MAX_BOOST)
        return mult, ", ".join(parts)


@dataclass
class SceneContext:
    rel: Relations
    tags: Set[str] = field(default_factory=set)
    scene_known: bool = False
    trace: List[str] = field(default_factory=list)

    def add(self, tags: Iterable[str], provider: bool = False) -> None:
        tags = set(tags)
        if provider and tags:
            self.scene_known = True
        self.tags |= self.rel.closure(tags)

    def weigh(self, items: List[Item], extra: Optional[Dict[str, float]] = None) -> List[Tuple[Item, float, str]]:
        out = []
        for it in items:
            mult, why = self.rel.explain(it, self.tags, self.scene_known)
            boost = (extra or {}).get(it.text.lower(), 1.0)
            out.append((it, it.weight * mult * boost, why))
        return out
