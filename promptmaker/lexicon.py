"""Keyword search and probability-based related word lookup.

Step 1  : extract keywords from the (English) user input.
Step 2  : search every element list for items containing the keywords
          -> those items get a higher pick weight.
Step 3  : look the keywords up in ``lexicon/related_words.json`` and draw
          related adjectives / verbs / nouns according to their probability.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .data_loader import Element
from .grammar import simple_lemma, tokenize

STOPWORDS = set(
    """a an the and or but of in on at to for with by from as is are was were be been being
    this that these those it its into over under very really so some any my your her his their our
    me you she he they we i please make create generate image picture photo video prompt want
    show draw give scene shot style about like""".split()
)


GENERIC_WORDS = {"woman", "women", "man", "men", "girl", "girls", "lady", "ladies", "person",
                 "people", "day", "time", "one", "two", "three"}


@dataclass
class LexiconResult:
    keywords: List[str] = field(default_factory=list)
    matches: Dict[str, List[str]] = field(default_factory=dict)  # element -> matched item texts
    related: Dict[str, List[str]] = field(default_factory=dict)  # pos -> words
    element_hints: Dict[str, List[str]] = field(default_factory=dict)  # element -> hinted items

    def boost_for(self, element_key: str) -> Dict[str, float]:
        """Extra weight multipliers for items of an element."""
        boosts: Dict[str, float] = {}
        for text in self.element_hints.get(element_key, []):
            boosts[text.lower()] = 40.0
        for text in self.matches.get(element_key, []):
            boosts[text.lower()] = max(boosts.get(text.lower(), 1.0), 60.0)
        return boosts

    def to_dict(self) -> dict:
        return {
            "keywords": self.keywords,
            "matches": self.matches,
            "related": self.related,
            "element_hints": self.element_hints,
        }


def extract_keywords(text: str) -> List[str]:
    words = [w.strip("'-") for w in tokenize(text)]
    out: List[str] = []
    for w in words:
        if len(w) < 3 or w in STOPWORDS:
            continue
        if w not in out:
            out.append(w)
    return out


def _word_in(word: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(word), text.lower()) is not None


def search_elements(keywords: List[str], elements: Dict[str, Element]) -> Dict[str, List[str]]:
    matches: Dict[str, List[str]] = {}
    if not keywords:
        return matches
    lemmas = {simple_lemma(k) for k in keywords} | set(keywords)
    lemmas = {l for l in lemmas if len(l) >= 3}
    for key, el in elements.items():
        hits = [i.text for i in el.items if any(_word_in(l, i.text) for l in lemmas)]
        if hits:
            matches[key] = hits
    return matches


def _norm_entries(entries) -> List[Tuple[str, float]]:
    out = []
    for e in entries or []:
        if isinstance(e, str):
            out.append((e, 0.5))
        elif isinstance(e, (list, tuple)) and e:
            out.append((str(e[0]), float(e[1]) if len(e) > 1 else 0.5))
        elif isinstance(e, dict):
            out.append((e["word"], float(e.get("p", 0.5))))
    return out


def related_words(keywords: List[str], related_db: Dict[str, dict], rng: random.Random,
                  max_per_pos: int = 3) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Draw related words. Each candidate is kept with its own probability."""
    pools: Dict[str, Dict[str, float]] = {}
    hints: Dict[str, List[str]] = {}
    for kw in keywords:
        entry = related_db.get(kw) or related_db.get(simple_lemma(kw))
        if entry is None:
            # alias lookup
            for k, v in related_db.items():
                if kw in v.get("aliases", []):
                    entry = v
                    break
        if not entry:
            continue
        for pos in ("adjectives", "verbs", "nouns"):
            for word, p in _norm_entries(entry.get(pos)):
                pool = pools.setdefault(pos, {})
                pool[word] = max(pool.get(word, 0.0), p)
        for el_key, texts in (entry.get("elements") or {}).items():
            bucket = hints.setdefault(el_key, [])
            for t in texts:
                if t not in bucket:
                    bucket.append(t)
    related: Dict[str, List[str]] = {}
    for pos, pool in pools.items():
        chosen = [w for w, p in sorted(pool.items(), key=lambda kv: -kv[1]) if rng.random() < p]
        rng.shuffle(chosen)
        if not chosen and pool:  # always keep at least the most probable one
            chosen = [max(pool.items(), key=lambda kv: kv[1])[0]]
        related[pos] = chosen[:max_per_pos]
    return related, hints


def analyze(text: str, elements: Dict[str, Element], related_db: Dict[str, dict],
            rng: random.Random) -> LexiconResult:
    keywords = extract_keywords(text)
    matches = search_elements(keywords, elements)
    related, hints = related_words(keywords, related_db, rng)
    return LexiconResult(keywords=keywords, matches=matches, related=related, element_hints=hints)
