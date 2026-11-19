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
NUDE_WORDS = {"nude", "naked", "undressed", "unclothed", "nudity", "disrobed"}
_KO_NUDE = ("누드", "알몸", "나체", "전라", "벗은", "벗고", "맨몸", "올누드")
_NOT_NUDE_RE = re.compile(r"\b(?:nude|naked)[- ](?:colou?r(?:ed)?|tone[sd]?|beige|lip(?:stick|s)?|makeup|heels|pumps|shade)\b"
                          r"|\b(?:semi|partially|half)[- ]?(?:nude|naked)\b", re.I)
_KO_IMAGE = ("사진", "화보", "그라비아", "이미지", "셀카", "포트레이트")
_KO_VIDEO = ("영상", "동영상", "비디오", "클립", "뮤비")

# Elements that are fully locked to the items the request mentions.
# Order matters: a keyword locks only the first element (in this order) it matches,
# so "beach" locks the location and not "honey-blonde beach waves" in hair.
# Groups share a keyword; a keyword consumed by an earlier group is not used by later groups.
LOCK_GROUPS = (("time_weather",), ("location",), ("photo_style", "video_style"), ("beauty",),
               ("swimwear", "dress", "top", "bottom", "sleepwear", "lingerie", "traditional", "sportswear",
                "outerwear", "footwear", "accessory", "male_attire"), ("pose", "action", "motion"), ("expression",),
               ("lighting",), ("hair",), ("makeup",), ("skin",), ("camera",), ("color_grade",),
               ("mood",))
LOCKABLE = tuple(k for g in LOCK_GROUPS for k in g)
GENERIC = {"woman", "women", "man", "men", "girl", "girls", "lady", "model", "models", "person",
           "people", "day", "time", "beautiful", "pretty", "photo", "photograph", "picture", "image", "video", "taken", "shot", "wearing",
           "costume", "costumes", "outfit", "outfits", "attire", "clothes", "clothing", "scene"}
# [0.13.0] 수식어(색·길이·재질)만으로는 의상·포즈 요소를 잠그지 않음
# ('long hair' 의 long 이 'long skirt' · 'long-sleeved top' 을 잠그던 문제)
MODIFIER_ONLY_GROUPS = {"swimwear", "dress", "top", "bottom", "sleepwear", "lingerie", "traditional", "sportswear",
                        "outerwear", "footwear", "accessory", "male_attire", "pose", "action", "motion"}
# 다른 요소 이름인 말은 그 요소에서만 찾음 ('hair' 가 'white shorts and her hair' 같은 의상 줄을 잠그지 않게)
DOMAIN_WORDS = {"hair": "hair", "hairstyle": "hair", "skin": "skin", "makeup": "makeup", "lipstick": "makeup",
                "expression": "expression", "lighting": "lighting"}
# 장소 중심 명사 앞에 붙으면 다른 뜻이 되는 말 ('gravel garden bed' 화단, 'stream bed' 하천 바닥)
PLACE_COMPOUND = {"garden", "gravel", "stream", "river", "plant", "flower", "sea", "creek", "lake", "ocean", "seed",
                  "oyster", "truck", "kelp", "moss", "rose", "vegetable", "reed", "coral", "pebble", "rock", "water",
                  "tulip", "herb", "lily", "test", "nail", "railway", "road", "canal", "pond"}
_PLACE_CUT_RE = re.compile(r"\s(?:with|of|in|on|at|beside|behind|above|below|under|near|by|to|across|against|"
                           r"where|that|and|next|between|over|from|for|surrounded|overlooking|facing)\b.*$")
_PLACE_LEAD_RE = re.compile(r"^(?:(?:on|in|at|inside|by|near|beside|along|atop|onto|into|across|under|amid|"
                            r"against|outside|within|on top of)\s+)?(?:(?:a|an|the)\s+)?")
_NOT_PLACE_LOCK = {"set", "setting", "backdrop", "wall", "window", "interior", "indoors", "outdoors", "concert",
                   "festival", "car", "vehicle"}
ON_PLACES = {"bed", "sofa", "couch", "beach", "floor", "rooftop", "balcony", "bench", "futon", "mattress", "rug",
             "carpet", "lawn", "grass", "sand", "deck", "terrace", "street", "bridge", "pier", "yacht", "boat", "stage",
             "desk", "table", "chair", "stairs", "staircase", "train", "bus", "swing"}
HAIR_MODS = {"long", "short", "straight", "wavy", "curly", "black", "brown", "blonde", "blond", "silky", "sleek",
             "shoulder-length", "medium-length", "layered", "loose", "messy", "dark", "light", "glossy", "shiny",
             "jet-black", "chestnut", "platinum", "red", "pink", "silver", "ash", "auburn", "bob", "braided", "wet",
             "flowing", "voluminous", "thick", "fine", "bobbed", "permed", "dark-brown", "light-brown"}


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
    nude: bool = False                                           # the request asks for no clothes at all
    undressing: bool = False                                     # '옷을 벗는 중/장면': 옷을 입고 벗는 동작 (누드와 다름)
    framing: Optional[str] = None                                # 샷 구도 (framing.FRAMINGS 중 하나)
    framings_found: List[str] = field(default_factory=list)      # 요청에서 찾은 구도 전부 (충돌 확인용)
    body_part: Optional[str] = None                              # 신체 부위 클로즈업 ('다리 클로즈업' → legs)
    masturbation: bool = False                                   # [0.13.0] 자위 장면
    masturbation_kinds: List[str] = field(default_factory=list)  # 말한 자위 동작 (breast / clit / finger)
    distance: Optional[str] = None                               # [0.14.0] 피사체 거리 (distance.DISTANCES)
    distances_found: List[str] = field(default_factory=list)     # 요청에서 찾은 거리 전부

    def to_dict(self) -> Dict[str, Any]:
        return {"locks": self.locks, "pools": self.pools, "female": self.female, "male": self.male,
                "female_noun": self.female_noun, "prompt_type": self.prompt_type,
                "consumed": self.consumed, "nude": self.nude, "undressing": self.undressing,
                "framing": self.framing, "framings_found": self.framings_found, "body_part": self.body_part,
                "masturbation": self.masturbation, "masturbation_kinds": self.masturbation_kinds,
                "distance": self.distance, "distances_found": self.distances_found}

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
    "dress": ["mini slip dress", "mini dress", "minidress", "mini-dress", "sundress", "slip dress", "dress", "gown"],
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


def _place_terms() -> set:
    from .element_check import PLACE_TERMS
    return set(PLACE_TERMS)


def place_head(text: str) -> str:
    """장소 문구의 중심 명사 ('on a gravel garden bed' → 'garden bed', 'on a bed with white sheets' → 'bed').

    중심 명사 바로 앞 말이 :data:`PLACE_COMPOUND` 이면 둘을 붙여 돌려줌 (다른 장소로 봄).
    """
    low = re.sub(r"\{[^}]*\}", "", (text or "").lower()).strip(" ,.")
    if re.match(r"(?:beside|near|by|next to|outside|behind|around|across from|opposite)\b", low):
        return ""  # 그 장소 '옆·근처' 는 그 장소가 아님
    low = _PLACE_LEAD_RE.sub("", low, count=1)
    low = _PLACE_CUT_RE.sub("", low).strip(" ,")
    words = re.findall(r"[a-z][a-z'-]*", low)
    if not words:
        return ""
    if len(words) >= 2 and words[-2] in PLACE_COMPOUND:
        return f"{words[-2]} {words[-1]}"
    return words[-1]


def _hair_phrase(low: str, elements: Dict[str, Element], intent: Intent) -> set:
    """'long straight hair' → 머리 요소를 그 수식어가 모두 들어간 줄로 잠금 (없으면 요청 문구 그대로)."""
    if "hair" not in elements:
        return set()
    for m in re.finditer(r"((?:[a-z-]+\s+){1,4})hair(?:ed)?\b|\b([a-z]+)-haired\b", low):
        pre = m.group(1).split() if m.group(1) else [m.group(2)]
        mods: List[str] = []
        for w in reversed(pre):
            if w in HAIR_MODS:
                mods.insert(0, w)
            else:
                break
        if not mods:
            continue
        items = [i.text for i in elements["hair"].items
                 if all(_word_in(w, i.text) for w in mods) and _word_in("hair", i.text)]
        if items:
            intent.pools["hair"] = items
        else:
            intent.locks["hair"] = " ".join(mods) + " hair"
        return set(mods) | {"hair", "haired", f"{mods[-1]}-haired"}
    return set()


def _phrases(text: str, elements: Dict[str, Element], intent: Intent) -> set:
    """Lock garments / skin tone written as '<modifier> <noun>'. Returns the words used."""
    used = set()
    low = text.lower()
    used |= _hair_phrase(low, elements, intent)
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
            canon = "mini skirt" if noun in ("miniskirt", "mini-skirt") else noun
            canon = "mini dress" if noun in ("minidress", "mini-dress") else canon
            if not mods and " " in canon and canon != "mini skirt":
                # 0.15.2: 'mini dress' · 'slip dress' 처럼 두 단어 의상은 수식어가 없어도 두 단어가 다 있는 줄로
                # (전엔 'dress' 만 보고 아무 원피스나 골라 '미니원피스' 요청이 무시되고 'features mini' 로 남음)
                parts = canon.split()
                items = [i.text for i in elements[key].items if all(_word_in(w, i.text) for w in parts)]
                if items:
                    intent.pools[key] = items
                else:
                    intent.locks[key] = ("an " if canon[0] in "aeiou" else "a ") + canon
                used |= set(parts) | set(noun.replace("-", " ").split())
                break
            if not mods:
                # a bare noun: the normal keyword search picks matching items
                # (그래서 used 에 넣지 않음 — 넣으면 키워드 검색이 건너뛰어 '셔츠' 요청이 무시됨)
                break
            used |= set(mods) | set(noun.replace("-", " ").split())
            head = canon if " " in noun and canon != "mini skirt" else canon.split()[-1]  # 'slip dress' 는 통째로
            items = [i.text for i in elements[key].items
                     if all(_word_in(w, i.text) for w in mods) and _word_in(head, i.text)]
            if items:
                intent.pools[key] = items
            else:  # nothing like it in the list: use the user's own words
                phrase = " ".join(mods + [canon])
                intent.locks[key] = ("an " if phrase[0] in "aeiou" else "a ") + phrase \
                    if canon.endswith("ss") or not canon.endswith("s") else phrase
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
    from .restrictions import KNOWN_NATIONALITIES  # 요소 목록에 없는 국적도 알아봄 (규칙으로 바꾸기 위해)
    nat_items = [i.text for i in elements.get("nationality", Element("nationality")).items]
    nat_items += [n for n in KNOWN_NATIONALITIES if n not in nat_items]
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

    # 누드 요청: 의상 요소를 채우지 않음 (누드톤 립스틱 · 반누드 같은 표현은 제외)
    plain = _NOT_NUDE_RE.sub(" ", text)
    if any(w in NUDE_WORDS for w in tokenize(plain)) or any(k in original for k in _KO_NUDE):
        intent.nude = True
        intent.consumed.extend(w for w in tokenize(plain) if w in NUDE_WORDS)

    # 옷을 벗는 중 / 벗는 장면: 옷은 입은 채 벗는 동작 ('벗고 있는' 이 누드로 잡히지 않게)
    from . import framing as _framing, undressing as _undressing
    pre_used = set()
    if _undressing.detect(original, text):
        intent.undressing = True
        intent.nude = False
        pre_used |= {w for w in words if w in _undressing.CONSUMED}

    # [0.13.0] 자위 장면: 동작은 masturbation.txt 에서 (가슴·클리토리스·손가락 동작 단어는 키워드로 쓰지 않음)
    from . import masturbation as _mast
    found, kinds = _mast.detect(original, text)
    if found:
        intent.masturbation, intent.masturbation_kinds = True, kinds
        pre_used |= {w for w in words if w in _mast.CONSUMED}
        if intent.female is None:
            intent.female, intent.male = 1, 0

    # 샷 구도 (얼빡샷 · 클로즈업 · 미디엄 · 카우보이 · 전신) — 동작·장소 키워드로 쓰이지 않게 따로 뺌
    intent.framing, intent.framings_found, fwords = _framing.detect(original, text)
    for phrase in fwords:  # 'close-up' 처럼 붙은 말과 'close', 'up' 둘 다
        toks = set(tokenize(phrase.lower()))
        pre_used |= toks | {p for tok in toks for p in tok.split("-")}
    if intent.framing:
        pre_used |= {w for w in ("shot", "shots", "face", "only", "frame", "framed") if w in words}

    # [0.14.0] 피사체 거리 (가까이서 · 중거리 · 멀리서 · 원경) — 'distance', 'far' 가 장소·포즈 키워드로 걸리지 않게
    from . import distance as _distance
    intent.distance, intent.distances_found, dwords = _distance.detect(original, text)
    for phrase in dwords:
        toks = set(tokenize(phrase.lower()))
        pre_used |= toks | {p for tok in toks for p in tok.split("-")}

    # 신체 부위 클로즈업 ('다리 클로즈업', 'close-up of her legs'): 일반 구도보다 우선
    from . import body_parts as _body_parts
    part, _ = _body_parts.detect(original, "")
    en_part, en_phrase = _body_parts.detect("", text)
    part = part or en_part
    if part:
        intent.body_part = part
        intent.framing = _framing.BODY_PART
        intent.framings_found = [_framing.BODY_PART]
        if en_phrase:
            toks = set(tokenize(en_phrase.lower()))
            pre_used |= toks | {p for tok in toks for p in tok.split("-")}
        # 번역문에 남은 부위 이름 ('clavicle close-up shot') 도 키워드로 쓰지 않음
        part_re = re.compile(r"\b(?:" + str(_body_parts.PARTS[part]["en"]) + r")\b", re.I)
        pre_used |= {w for w in words if part_re.fullmatch(w)}
        if intent.framing and en_phrase == "":
            pre_used |= {w for w in ("close-up", "closeup", "close", "up", "macro", "detail") if w in words}

    # direct item matches -> locked pools
    skip = GENERIC | STOPWORDS | set(NUM_WORDS) | FEMALE_SINGULAR | FEMALE_PLURAL | MALE_SINGULAR \
        | MALE_PLURAL | set(COUNTRY_TO_ADJ) | {n.lower() for n in nat_items}
    skip.discard("gravure")  # also a style word
    keywords = [w for w in dict.fromkeys(words) if len(w) >= 3 and w not in skip]
    lemmas = {k: {k, simple_lemma(k)} for k in keywords}
    used = _phrases(text, elements, intent)
    intent.consumed.extend(sorted(used | pre_used))
    used |= pre_used
    if (used - pre_used) and intent.female is None and intent.male is None:
        # the request describes one person's look ("fair skin, white miniskirt"): one woman
        intent.female, intent.male = 1, 0
    for group in LOCK_GROUPS:
        group_kws = set()
        for key in group:
            el = elements.get(key)
            if not el:
                continue
            hits = []
            head_hits = []
            for item in el.items:
                for kw, forms in lemmas.items():
                    if kw in used:
                        continue
                    dom = next((DOMAIN_WORDS[f] for f in forms if f in DOMAIN_WORDS), None)
                    if dom and dom != key:
                        continue
                    if key in MODIFIER_ONLY_GROUPS and kw in MODIFIERS:
                        continue
                    if any(len(f) >= 3 and _word_in(f, item.text) for f in forms):
                        hits.append(item.text)
                        group_kws.add(kw)
                        if key == "location" and kw in _place_terms() and place_head(item.text) in forms:
                            head_hits.append(item.text)
                        break
            if key == "location" and head_hits:
                # [0.13.0] 'bed' → 침대인 장소만 ('gravel garden bed' · 'telephone beside bed' 제외)
                hits = head_hits
            elif key == "location" and key not in intent.pools and key not in intent.locks:
                # 요소 목록에 그 장소가 없으면 ('침대', '소파') 요청한 말 그대로 장소로
                from .element_check import PLACE_TERMS
                place = next((kw for kw in lemmas if kw not in used and kw in PLACE_TERMS
                              and kw not in GENERIC and kw not in _NOT_PLACE_LOCK), None)
                if place:
                    prep = "on" if place in ON_PLACES else "in"
                    intent.locks["location"] = f"{prep} {'an' if place[0] in 'aeiou' else 'a'} {place}"
                    hits = []
                    group_kws = {place}
            if hits and key not in intent.pools and key not in intent.locks:
                intent.pools[key] = hits
        used |= group_kws
        for kw in group_kws:
            if kw not in intent.consumed:
                intent.consumed.append(kw)
    intent.template_keywords = keywords
    return intent
