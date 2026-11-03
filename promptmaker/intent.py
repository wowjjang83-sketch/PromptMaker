"""Extract hard constraints ("intent") from the user's request.

The request must drive the result: what the user explicitly asked for
(age, nationality, number of people, location, style ...) is locked, and
randomness only fills the slots the request did not mention.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data_loader import Element
from .grammar import simple_lemma, tokenize
from .lexicon import STOPWORDS
from .safety import clamp_age

COUNTRY_TO_ADJ = {
    "korea": "Korean", "japan": "Japanese", "china": "Chinese", "taiwan": "Taiwanese",
    "thailand": "Thai", "vietnam": "Vietnamese", "philippines": "Filipino", "indonesia": "Indonesian",
    "malaysia": "Malaysian", "singapore": "Singaporean", "mongolia": "Mongolian", "india": "Indian",
    "america": "American", "usa": "American", "canada": "Canadian", "brazil": "Brazilian",
    "mexico": "Mexican", "argentina": "Argentinian", "colombia": "Colombian", "france": "French",
    "italy": "Italian", "spain": "Spanish", "portugal": "Portuguese", "germany": "German",
    "netherlands": "Dutch", "sweden": "Swedish", "norway": "Norwegian", "russia": "Russian",
    "ukraine": "Ukrainian", "poland": "Polish", "britain": "British", "england": "British",
    "ireland": "Irish", "greece": "Greek", "turkey": "Turkish", "egypt": "Egyptian",
    "australia": "Australian", "iran": "Persian", "hawaii": "Hawaiian",
}
NUM_WORDS = {"one": 1, "single": 1, "two": 2, "three": 3, "four": 4, "a": 1, "an": 1}
FEMALE_SINGULAR = {"woman", "girl", "lady", "model", "idol", "actress", "female", "she", "her",
                   "wife", "girlfriend", "bride", "dancer", "ballerina", "gravure"}
FEMALE_PLURAL = {"women", "girls", "ladies", "models", "idols", "actresses", "females"}
MALE_SINGULAR = {"man", "guy", "boyfriend", "husband", "male", "groom", "he", "him"}
MALE_PLURAL = {"men", "guys", "males"}
COUPLE_WORDS = {"couple", "lovers", "date", "dating"}
SOLO_WORDS = {"solo", "alone", "single"}
NOUN_OVERRIDES = {"model": ("model", "models"), "models": ("model", "models"),
                  "idol": ("idol", "idols"), "idols": ("idol", "idols"),
                  "actress": ("actress", "actresses"), "dancer": ("dancer", "dancers")}
IMAGE_WORDS = {"photo", "photograph", "picture", "image", "pictorial", "photoshoot", "gravure",
               "portrait", "selfie", "snapshot", "editorial"}
VIDEO_WORDS = {"video", "clip", "footage", "movie", "film", "animation", "motion"}
_KO_IMAGE = ("사진", "화보", "그라비아", "이미지", "셀카", "포트레이트")
_KO_VIDEO = ("영상", "동영상", "비디오", "클립", "뮤비")

# Elements that are fully locked to the items the request mentions.
# Order matters: a keyword locks only the first element (in this order) it matches,
# so "beach" locks the location and not "honey-blonde beach waves" in hair.
# Groups share a keyword; a keyword consumed by an earlier group is not used by later groups.
LOCK_GROUPS = (("time_weather",), ("location",), ("photo_style", "video_style"),
               ("swimwear", "dress", "top", "bottom", "sleepwear", "lingerie", "traditional", "sportswear",
                "outerwear", "footwear", "accessory", "male_attire"), ("pose", "action", "motion"), ("expression",),
               ("lighting",), ("hair",), ("makeup",), ("skin",), ("camera",), ("color_grade",),
               ("mood",))
LOCKABLE = tuple(k for g in LOCK_GROUPS for k in g)
GENERIC = {"woman", "women", "man", "men", "girl", "girls", "lady", "model", "models", "person",
           "people", "day", "time", "beautiful", "pretty", "photo", "photograph", "picture", "image", "video", "taken", "shot", "wearing"}


@dataclass
class Intent:
    locks: Dict[str, str] = field(default_factory=dict)          # element -> fixed value
    pools: Dict[str, List[str]] = field(default_factory=dict)    # element -> allowed items
    female: Optional[int] = None
    male: Optional[int] = None
    female_noun: Optional[tuple] = None
    prompt_type: Optional[str] = None
    template_keywords: List[str] = field(default_factory=list)
    consumed: List[str] = field(default_factory=list)            # words used by the intent

    def to_dict(self) -> Dict[str, Any]:
        return {"locks": self.locks, "pools": self.pools, "female": self.female, "male": self.male,
                "female_noun": self.female_noun, "prompt_type": self.prompt_type,
                "consumed": self.consumed}

    def casting_ok(self, female: int, male: int) -> bool:
        if self.female is not None and female != self.female:
            return False
        if self.male is not None and male != self.male:
            return False
        return True


def _word_in(word: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(word) + r"\b", text, re.I) is not None


def _age(original: str, english: str) -> Optional[int]:
    m = (re.search(r"\b(\d{2})[- ]?(?:year|yr)s?[- ]?old\b", english, re.I)
         or re.search(r"\b(?:age|aged)\s*(\d{2})\b", english, re.I)
         or re.search(r"(\d{2})\s*(?:살|세)", original))
    if m:
        return clamp_age(m.group(1))
    m = re.search(r"\b(?:in (?:her|their) )?(early|mid|late)?[- ]?(twenties|thirties|20s|30s)\b", english, re.I)
    if m:
        base = 20 if m.group(2).lower() in ("twenties", "20s") else 30
        return base + {"early": 2, "mid": 5, "late": 8}.get((m.group(1) or "mid").lower(), 5)
    m = re.search(r"(20|30)\s*대\s*(초반|중반|후반)?", original)
    if m:
        return int(m.group(1)) + {"초반": 2, "중반": 5, "후반": 8}.get(m.group(2) or "중반", 5)
    return None


def _casting(words: List[str], text: str, intent: Intent) -> None:
    female = male = None
    for i, w in enumerate(words):
        prev = words[i - 1] if i else ""
        if w in FEMALE_PLURAL:
            female = max(female or 0, NUM_WORDS.get(prev, 2) if prev not in ("a", "an") else 2)
        elif w in FEMALE_SINGULAR:
            female = max(female or 0, 1)
        elif w in MALE_PLURAL:
            male = max(male or 0, NUM_WORDS.get(prev, 2) if prev not in ("a", "an") else 2)
        elif w in MALE_SINGULAR:
            male = max(male or 0, 1)
        elif w in COUPLE_WORDS:
            female, male = max(female or 0, 1), max(male or 0, 1)
    if any(w in SOLO_WORDS for w in words):
        male = 0
    if female is not None and male is None:
        male = 0
    if male and female is None:
        female = 1
    intent.female, intent.male = female, male
    for w in words:
        if w in NOUN_OVERRIDES and intent.female_noun is None:
            intent.female_noun = NOUN_OVERRIDES[w]


# ---------------------------------------------------------------- attribute phrases
# "white miniskirt", "fair skin": the modifier belongs to that noun only, so it must
# not leak into other elements ("white" studio, "bright" afternoon).
GARMENT_NOUNS = {
    "bottom": ["miniskirt", "mini skirt", "mini-skirt", "skirt", "shorts", "hot pants", "jeans", "pants",
               "trousers", "leggings"],
    "top": ["crop top", "tank top", "top", "blouse", "shirt", "t-shirt", "tee", "sweater", "camisole",
            "hoodie", "turtleneck", "tube top", "bralette"],
    "dress": ["mini dress", "sundress", "slip dress", "dress", "gown"],
    "swimwear": ["bikini", "swimsuit", "monokini", "one-piece"],
    "outerwear": ["coat", "jacket", "blazer", "cardigan", "trench"],
    "footwear": ["high heels", "heels", "boots", "sneakers", "sandals", "loafers", "flats"],
    "sleepwear": ["pajamas", "pyjamas", "nightgown", "bathrobe"],
    "traditional": ["hanbok", "kimono", "yukata", "qipao", "ao dai"],
}
MODIFIERS = {"white", "black", "red", "pink", "blue", "navy", "green", "yellow", "purple", "beige", "brown",
             "grey", "gray", "silver", "gold", "golden", "ivory", "cream", "pastel", "denim", "leather",
             "lace", "satin", "silk", "sheer", "pleated", "tight", "short", "long", "mini", "floral",
             "striped", "knit", "cropped", "oversized", "sequined", "velvet", "tweed", "wet", "see-through",
             "high-waisted", "low-rise", "frilly", "ruffled", "transparent"}
SKIN_TONES = {"bright": ["fair", "porcelain", "pale", "luminous"], "fair": ["fair", "porcelain"],
              "light": ["fair", "porcelain"], "pale": ["pale", "porcelain", "fair"], "white": ["fair", "porcelain", "pale"],
              "porcelain": ["porcelain"], "glowing": ["glow", "luminous", "dewy"], "dewy": ["dewy"],
              "tan": ["tan", "bronzed", "sun-kissed"], "tanned": ["tan", "bronzed", "sun-kissed"],
              "dark": ["dark", "deep", "ebony"], "brown": ["brown", "caramel", "deep"],
              "olive": ["olive"], "smooth": ["smooth"], "glossy": ["glow", "glistening"]}


def _phrases(text: str, elements: Dict[str, Element], intent: Intent) -> set:
    """Lock garments / skin tone written as '<modifier> <noun>'. Returns the words used."""
    used = set()
    low = text.lower()
    for key, nouns in GARMENT_NOUNS.items():
        if key not in elements:
            continue
        for noun in nouns:
            m = re.search(r"((?:[a-z-]+\s+){0,3})\b" + re.escape(noun) + r"s?\b", low)
            if not m:
                continue
            # keep only the modifiers directly in front of the noun
            pre = m.group(1).split()
            while pre and pre[0] not in MODIFIERS:
                pre.pop(0)
            mods = [w for w in pre if w in MODIFIERS]
            used |= set(mods) | set(noun.replace("-", " ").split())
            canon = "mini skirt" if noun in ("miniskirt", "mini-skirt") else noun
            if not mods:
                break  # a bare noun: the normal keyword search picks matching items
            items = [i.text for i in elements[key].items
                     if all(_word_in(w, i.text) for w in mods) and _word_in(canon.split()[-1], i.text)]
            if items:
                intent.pools[key] = items
            else:  # nothing like it in the list: use the user's own words
                phrase = " ".join(mods + [canon])
                intent.locks[key] = ("an " if phrase[0] in "aeiou" else "a ") + phrase \
                    if not canon.endswith(("s", "jeans", "pants", "shorts", "heels", "boots")) else phrase
            break
    m = re.search(r"\b([a-z-]+)[- ]skin(?:ned)?\b|\bskin\s+(?:is|that is)\s+([a-z-]+)", low)
    if m and "skin" in elements:
        adj = m.group(1) or m.group(2)
        tones = SKIN_TONES.get(adj)
        if tones:
            items = [i.text for i in elements["skin"].items if any(_word_in(t, i.text) for t in tones)]
            if items:
                intent.pools["skin"] = items
            else:
                intent.locks["skin"] = adj
            used |= {adj, "skin"}
    return used


def extract(original: str, english: str, elements: Dict[str, Element]) -> Intent:
    intent = Intent()
    text = english or ""
    words = tokenize(text)

    # prompt type
    if any(w in VIDEO_WORDS for w in words) or any(k in original for k in _KO_VIDEO):
        intent.prompt_type = "video"
    elif any(w in IMAGE_WORDS for w in words) or any(k in original for k in _KO_IMAGE):
        intent.prompt_type = "image"

    age = _age(original, text)
    if age:
        intent.locks["age"] = str(age)
        intent.consumed.append("age")

    # nationality: adjective in the list, or country name
    nat_items = [i.text for i in elements.get("nationality", Element("nationality")).items]
    for nat in nat_items:
        if _word_in(nat, text):
            intent.locks["nationality"] = nat
            intent.consumed.append(nat.lower())
            break
    else:
        for w in words:
            if w in COUNTRY_TO_ADJ:
                intent.locks["nationality"] = COUNTRY_TO_ADJ[w]
                intent.consumed.append(w)
                break

    _casting(words, text, intent)

    # direct item matches -> locked pools
    skip = GENERIC | STOPWORDS | set(NUM_WORDS) | FEMALE_SINGULAR | FEMALE_PLURAL | MALE_SINGULAR \
        | MALE_PLURAL | set(COUNTRY_TO_ADJ) | {n.lower() for n in nat_items}
    skip.discard("gravure")  # also a style word
    keywords = [w for w in dict.fromkeys(words) if len(w) >= 3 and w not in skip]
    lemmas = {k: {k, simple_lemma(k)} for k in keywords}
    used = _phrases(text, elements, intent)
    intent.consumed.extend(sorted(used))
    if used and intent.female is None and intent.male is None:
        # the request describes one person's look ("fair skin, white miniskirt"): one woman
        intent.female, intent.male = 1, 0
    for group in LOCK_GROUPS:
        group_kws = set()
        for key in group:
            el = elements.get(key)
            if not el:
                continue
            hits = []
            for item in el.items:
                for kw, forms in lemmas.items():
                    if kw in used:
                        continue
                    if any(len(f) >= 3 and _word_in(f, item.text) for f in forms):
                        hits.append(item.text)
                        group_kws.add(kw)
                        break
            if hits and key not in intent.pools and key not in intent.locks:
                intent.pools[key] = hits
        used |= group_kws
        for kw in group_kws:
            if kw not in intent.consumed:
                intent.consumed.append(kw)
    intent.template_keywords = keywords
    return intent
