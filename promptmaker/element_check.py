"""요소 한 줄 검사 — 문장 조각 걸러내기 · 다듬기 · 맞는 요소 찾기 (규칙 + 사전, AI 미사용).

요소 txt 한 줄은 '단어 또는 짧은 구' 여야 합니다. PromptMaker 가 요소들을 템플릿에 넣어
문법에 맞는 문장을 만들기 때문입니다. 그런데 이미지 분석·생성 프롬프트 자동 추가가 긴 문장을
쉼표로만 잘라 넣으면 이런 줄이 생깁니다.

    camera.txt    "19age is visible in a close-up portrait shot"        (주어 + 동사 문장)
    camera.txt    "her dark black hair styled with straight bangs ..."  (헤어인데 'framing' 때문에 카메라)
    lighting.txt  "a light blue flower pinned near her neck"            ('light' 가 색 이름인데 조명)
    color_grade   "a muted pink lipstick"                               (메이크업인데 'muted' 때문에 색감)
    action.txt    "suggesting an indoor setting with ambient lighting"  (설명용 분사구)
    pose.txt      "looking directly at the camera with a gentle"        (끝이 잘린 조각)

이 모듈은 그런 줄을 판단합니다.

    from promptmaker.element_check import check_item
    check_item("camera", "her dark brown hair styled in a high bun")
    # {"status": "moved", "key": "hair", "text": "...", "reason": "..."}

status
- ``ok``        그대로 써도 됨
- ``fixed``     잘린 문장을 다듬어서 씀 (``text`` 가 다듬은 문구)
- ``moved``     다른 요소에 맞음 (``key`` 가 맞는 요소)
- ``fragment``  문장 조각이라 쓸 수 없음
- ``wrong_key`` 이 요소 형식에 맞지 않고 맞는 요소도 못 찾음

판단 대상 요소는 ``CHECK_KEYS`` 뿐이며 (영상·상호작용·의상 상태 등은 판단하지 않음), 수위(NSFW)
제한은 없습니다. 기본 요소(패키지 데이터)는 모두 ``ok`` 여야 합니다 (tests/test_element_check.py).
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Tuple

# ------------------------------------------------------------------ 대상 요소
GARMENT_KEYS = ("swimwear", "lingerie", "sleepwear", "sportswear", "traditional", "dress", "top", "bottom",
                "outerwear", "footwear", "legwear", "accessory")
POSE_GROUP = ("pose", "action")
CHECK_KEYS = frozenset(("hair", "makeup", "skin", "expression", "pose", "action", "location", "time_weather",
                        "lighting", "camera", "color_grade", "photo_style", "mood", "beauty") + GARMENT_KEYS)
# 요소 한 줄 최대 길이 (단어). 더 길면 문장으로 봄
MAX_WORDS = 14
MAX_CHARS = 110

# ------------------------------------------------------------------ 사전
GARMENT_TERMS: Dict[str, Tuple[str, ...]] = {
    "swimwear": ("bikini", "bikini top", "bikini bottom", "bikini bottoms", "swimsuit", "monokini",
                 "bathing suit", "swimwear", "rash guard", "tankini"),
    "lingerie": ("lingerie", "bralette", "bra", "panties", "thong", "g-string", "underwear", "briefs", "garter",
                 "garter belt", "garter belts", "babydoll", "negligee", "teddy", "chemise", "corset", "bustier", "bodysuit",
                 "lace set"),
    "sleepwear": ("pajamas", "pajama", "pyjamas", "nightgown", "nightie", "nightdress", "bathrobe", "robe",
                  "sleep shirt", "sleepwear", "loungewear", "lounge set", "onesie", "nightshirt"),
    "sportswear": ("yoga set", "yoga pants", "sports bra", "tracksuit", "workout set", "gym outfit",
                   "tennis outfit", "athletic set", "activewear", "sportswear", "jersey", "leotard", "bike shorts"),
    "traditional": ("hanbok", "kimono", "yukata", "ao dai", "qipao", "cheongsam", "sari", "saree", "hanfu",
                    "dirndl", "kebaya"),
    "dress": ("dress", "sundress", "minidress", "gown", "jumpsuit", "romper", "playsuit"),
    "top": ("shirt", "t-shirt", "tee", "blouse", "tank top", "camisole", "cami", "crop top", "sweater", "hoodie",
            "sweatshirt", "turtleneck", "top", "tube top", "halter top", "polo", "knit", "vest top",
            "undershirt", "cardigan sweater"),
    "bottom": ("skirt", "miniskirt", "mini-skirt", "micro skirt", "shorts", "jeans", "pants", "trousers", "leggings", "culottes", "slacks",
               "joggers", "hot pants"),
    "outerwear": ("coat", "jacket", "blazer", "cardigan", "parka", "windbreaker", "poncho", "cape", "raincoat",
                  "overcoat", "cover-up", "shawl", "bolero", "vest"),
    "footwear": ("shoes", "heels", "stilettos", "pumps", "boots", "sneakers", "sandals", "loafers", "flats",
                 "slippers", "mules", "wedges", "espadrilles", "flip-flops", "barefoot", "mary janes", "clogs"),
    "legwear": ("stockings", "tights", "pantyhose", "socks", "thigh-highs", "fishnets", "leg warmers",
                "bare legs", "legs"),
    "accessory": ("necklace", "earrings", "earring", "bracelet", "ring", "choker", "pendant", "hat", "cap",
                  "beret", "beanie", "sunglasses", "glasses", "eyeglasses", "headband", "hairpin", "hair clip",
                  "hair ribbon", "ribbon", "scarf", "bag", "handbag", "purse", "tote", "backpack", "watch",
                  "wristwatch", "gloves", "tiara", "veil", "umbrella", "strap", "collar", "anklet", "brooch", "headphones", "hair bow", "bow",
                  "belt", "clutch", "crown", "parasol", "fan", "mask", "scrunchie", "chain", "jewelry",
                  "jewellery", "bangle", "hairband", "piercing", "harness", "cuff", "wristband", "headpiece",
                  "accessories"),
}
GENERAL_GARMENTS = ("top", "bottom", "dress", "outerwear")
# 의상 단어 말고도 그 요소임을 알려 주는 말 ('a tennis skirt and polo shirt' → 운동복)
GARMENT_CONTEXT: Dict[str, Tuple[str, ...]] = {
    "sportswear": ("tennis", "running", "workout", "yoga", "gym", "athletic", "cycling", "golf", "ski",
                   "leggings", "crop top", "bike shorts"),
    "sleepwear": ("nightshirt", "sleep", "lounge"),
    "traditional": ("hanbok", "traditional"),
}
# 상의·하의·원피스보다 먼저 보는 특정 의상 (순서 중요: 'sports bra' 는 운동복, 'bikini top' 은 수영복)
SPECIFIC_GARMENTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("sportswear", ("sports bra", "yoga pants", "yoga set", "tracksuit", "gym outfit", "tennis outfit",
                    "athletic set", "activewear", "sportswear", "leotard", "bike shorts")),
    ("swimwear", ("bikini", "swimsuit", "swimwear", "monokini", "bathing suit", "tankini")),
    ("lingerie", ("lingerie", "bralette", "bra", "panties", "thong", "g-string", "underwear", "briefs", "garter",
                  "negligee", "bustier", "teddy", "chemise")),
    ("sleepwear", ("pajama", "pajamas", "pyjamas", "nightgown", "nightie", "nightdress", "bathrobe",
                   "sleep shirt", "sleepwear", "onesie")),
    ("traditional", ("hanbok", "kimono", "yukata", "ao dai", "qipao", "cheongsam", "sari", "saree", "hanfu")),
)
HAIR_TERMS = ("hair", "hairstyle", "ponytail", "pigtails", "twintails", "twin tails", "bun", "braid", "braids",
              "braided", "bob", "bangs", "fringe", "curls", "ringlets", "updo", "chignon", "pixie cut", "hime cut",
              "lob", "perm", "waves", "dreadlocks", "cornrows", "buzz cut", "space buns", "top knot", "side part",
              "undercut", "mullet", "wolf cut", "locks", "tresses", "blonde", "brunette", "redhead", "strands")
MAKEUP_TERMS = ("makeup", "make-up", "lipstick", "lips", "lip", "lip gloss", "gloss", "eyeliner", "eyeshadow",
                "eye shadow", "blush", "mascara", "lashes", "eyelashes", "brows", "eyebrows", "liner",
                "highlighter", "contour", "contoured", "freckles", "rouge", "lip tint", "nail polish", "manicure",
                "cheeks", "cheekbones", "nails", "glitter", "beauty mark", "eyes", "irises", "skin glow",
                "foundation", "concealer")
MAKEUP_PRODUCTS = ("makeup", "make-up", "lipstick", "lip gloss", "eyeliner", "eyeshadow", "eye shadow", "blush",
                   "mascara", "highlighter", "lip tint", "nail polish", "foundation", "concealer")
EXPRESSION_TERMS = ("smile", "grin", "smirk", "pout", "expression", "gaze", "look", "glance", "stare", "wink",
                    "frown", "laugh", "half-smile", "eyes closed", "teary eyes", "lips parted", "tongue out",
                    "blushing", "surprised", "mouth")
CAMERA_TERMS = ("lens", "close-up", "closeup", "close up", "framing", "angle", "perspective", "composition",
                "depth of field", "bokeh", "wide-angle", "telephoto", "macro", "full-body", "full body",
                "upper body", "cowboy shot", "medium shot", "wide shot", "long shot", "shot", "headshot",
                "from above", "from below", "aerial view", "overhead view", "bird's-eye view", "pov",
                "point of view", "dutch angle", "rule of thirds", "rule-of-thirds", "selfie", "mirror selfie",
                "shot on", "shot with", "captured with", "focus", "motion blur", "tilt-shift", "fisheye", "frame",
                "framed", "view", "crop", "cropped", "zoom", "eye-level", "waist up", "shoulders up", "profile")
# 'light' 가 색 이름(light blue, light-colored ...)이나 '가벼운' 뜻일 때는 조명이 아님
_COLOR_NAMES = ("blue", "gray", "grey", "pink", "brown", "green", "purple", "yellow", "beige", "red", "orange",
                "violet", "teal", "tan", "olive", "peach", "skin", "colored", "coloured", "toned", "wash",
                "background", "fabric", "makeup", "blush", "freckles", "stubble", "touch", "smile", "breeze")
LIGHTING_TERMS = ("lighting", "sunlight", "moonlight", "candlelight", "lamplight", "daylight", "starlight",
                  "firelight", "glow", "backlight", "backlighting", "backlit", "rim light", "flash",
                  "illumination", "spotlight", "light rays", "god rays", "sunbeams", "sun rays", "shadows",
                  "chiaroscuro", "light leaks", "lit", "neon", "lamp", "lamps", "candles", "sun", "sunshine",
                  "golden hour", "window light", "softbox", "ring light")
COLOR_TERMS = ("tones", "toned", "colors", "colours", "color palette", "colour palette", "color grading",
               "colour grading", "grade", "graded", "film grain", "grain", "black and white", "black-and-white",
               "monochrome", "saturated", "desaturated", "muted", "contrast", "faded", "film look", "hues",
               "palette", "sepia", "vintage", "pastel", "cross-processed", "color", "colour", "tint", "tinted")
PHOTO_TERMS = ("photograph", "photo", "photography", "snapshot", "still", "portrait", "headshot", "selfie",
               "picture", "image", "editorial", "pictorial", "polaroid", "film", "photoshoot", "photo shoot",
               "style", "aesthetic", "shot")
TIME_TERMS = ("night", "evening", "morning", "afternoon", "dusk", "dawn", "sunset", "sunrise", "twilight",
              "midnight", "noon", "midday", "golden hour", "blue hour", "rainy", "rain", "snowy", "snow",
              "snowfall", "foggy", "fog", "misty", "mist", "cloudy", "overcast", "stormy", "storm", "sunny",
              "winter", "summer", "spring", "autumn", "fall", "daytime", "nighttime", "day", "season", "weather",
              "sky", "breeze", "wind", "windy", "clouds")
PLACE_TERMS = (
    "cafe", "café", "coffee shop", "beach", "shore", "seaside", "coast", "street", "alley", "avenue", "road",
    "room", "bedroom", "living room", "bathroom", "kitchen", "studio", "park", "garden", "forest", "woods",
    "jungle", "field", "meadow", "city", "skyline", "rooftop", "pool", "library", "office", "hotel", "suite",
    "apartment", "balcony", "terrace", "patio", "porch", "sea", "ocean", "lake", "river", "pond", "waterfall",
    "mountain", "mountains", "hill", "valley", "cliff", "desert", "market", "shop", "store", "boutique",
    "restaurant", "bar", "pub", "club", "lounge", "subway", "station", "train", "bus", "car", "taxi", "bridge",
    "temple", "shrine", "palace", "castle", "courtyard", "hanok", "greenhouse", "gym", "museum", "gallery",
    "church", "village", "town", "square", "plaza", "pier", "harbor", "dock", "boat", "yacht", "cabin", "tent",
    "vineyard", "farm", "barn", "warehouse", "factory", "stage", "runway", "lobby", "hallway", "corridor",
    "staircase", "stairs", "elevator", "airport", "sauna", "spa", "onsen", "hot spring", "bathtub", "bath",
    "diner", "arcade", "mall", "dressing room", "locker room", "set", "backstage", "concert", "festival",
    "aquarium", "zoo", "island", "lagoon", "bay", "countryside", "neighborhood", "downtown", "bookstore",
    "bakery", "attic", "basement", "penthouse", "villa", "mansion", "resort", "poolside", "cottage", "chapel",
    "ruins", "sidewalk", "highway", "parking lot", "veranda", "bed", "sofa", "couch", "vehicle", "interior",
    "indoors", "outdoors", "classroom", "window", "floor", "wall", "doorway", "setting", "backdrop")
PREPS = ("in", "on", "at", "inside", "by", "near", "beside", "along", "amid", "amidst", "among", "atop",
         "beneath", "under", "underneath", "within", "across", "through", "outside", "in front of", "next to",
         "against", "outdoors", "indoors", "during", "late", "early")
MOOD_NOUNS = ("mood", "atmosphere", "vibe", "vibes", "ambiance", "ambience", "feel", "feeling", "aesthetic")
# 0.10.0 — 피부·분위기 요소로 쓰기엔 뜻이 없는 말 (와일드카드 분석에서 나온 것)
SKIN_REJECT = {"medium", "subtle", "moisture", "exposed", "detailed textures", "realistic detailed", "realistic",
               "minimal exposure", "soft natural"}
MOOD_REJECT = {"form", "lit", "clean", "personal", "focused"}
# -ing 로 끝나지만 동작이 아니라 형용사/명사로 쓰이는 단어
ADJ_ING = {"stunning", "charming", "alluring", "captivating", "breathtaking", "striking", "dazzling", "glistening",
           "shimmering", "flickering", "glowing", "sparkling", "gleaming", "twinkling", "flattering", "revealing",
           "plunging", "matching", "cascading", "billowing", "flowing", "backlighting", "lighting", "clothing",
           "evening", "morning", "string", "ring", "earring", "wedding", "amazing", "interesting", "loving",
           "inviting", "enchanting", "mesmerizing", "intriguing", "teasing", "pleasing", "piercing", "smoldering",
           "smouldering", "sweltering", "blazing", "freezing", "dripping", "soaking", "darling", "ceiling",
           "building", "painting", "bedding", "awning", "railing", "landing", "clearing", "refreshing",
           "relaxing", "calming", "soothing", "warming", "stocking", "legging", "flooring", "lingering",
           "seductive-looking", "sweeping", "fitting", "form-fitting", "figure-hugging", "willing",
           "outgoing", "engaging", "disarming", "welcoming", "comforting", "commanding", "unassuming"}
# 분사로 시작하지만 사람의 포즈·동작이 아니라 사진·장면을 설명하는 말 (문장을 자른 조각)
DESCRIBE_ING = {"suggesting", "conveying", "contributing", "consisting", "displaying", "cropping", "focusing",
                "obscuring", "emphasizing", "emphasising", "highlighting", "creating", "enhancing", "reflecting",
                "adding", "giving", "featuring", "showcasing", "showing", "depicting", "capturing", "evoking",
                "exuding", "radiating", "complementing", "contrasting", "blending", "surrounding", "including",
                "accentuating", "illuminating", "casting", "softening", "making", "filling", "indicating",
                "implying", "hinting", "revealing", "rendering", "occupying", "dominating", "positioning",
                "centering", "framing", "isolating", "separating", "drawing", "maintaining", "allowing",
                "ensuring", "providing", "offering", "presenting", "producing", "resulting", "appearing",
                "seeming", "being", "having", "wearing", "partially",
                # 0.10.0 — 추상 표현 ('embodying a mix of innocence') · 옷 입기 동사 ('rocking a mesh bodysuit')
                "embodying", "channeling", "channelling", "combining", "feeling", "keeping", "originating",
                "coming", "embracing", "grounding", "blurring", "styling", "moving", "donning", "rocking",
                "sporting", "flaunting", "layering", "pairing", "adorning", "lending", "projecting", "obscured",
                "dressed", "clad"}
PARTICIPLE_EXTRA = {"perched", "seated", "curled", "sprawled", "propped", "crouched", "draped", "tucked", "bent",
                    "half-lying", "half-sitting", "laid", "posed", "caught", "mid-laugh", "mid-stride",
                    "mid-twirl", "kneeling", "squatting", "lying", "sitting", "standing"}
POSE_WORDS = {"standing", "sitting", "seated", "lying", "laying", "kneeling", "leaning", "crouching", "squatting",
              "reclining", "lounging", "perched", "posing", "walking", "sprawled", "bending", "resting",
              "stretching", "arching", "straddling", "curled", "looking", "facing", "turning", "propped",
              "crouched", "half-lying", "half-sitting", "balancing", "floating", "on"}


def _terms_re(terms: Iterable[str]) -> "re.Pattern[str]":
    alts = sorted({t.lower() for t in terms}, key=len, reverse=True)
    return re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(t) for t in alts) + r")(?:s|es)?(?![\w-])", re.I)


_HAIR_RE = _terms_re(HAIR_TERMS)
_MAKEUP_RE = _terms_re(MAKEUP_TERMS)
_MAKEUP_PRODUCT_RE = _terms_re(MAKEUP_PRODUCTS)
_EXPR_RE = _terms_re(EXPRESSION_TERMS)
_CAMERA_RE = re.compile(_terms_re(CAMERA_TERMS).pattern + r"|\b\d{2,3}\s?mm\b|\bf/\d", re.I)
_LIGHT_WORD_RE = re.compile(r"(?<![\w-])light(?:s)?(?![\w-])(?!\s*(?:-|\s)(?:" + "|".join(_COLOR_NAMES) + r")\b)"
                            r"|(?<![\w-])light-(?!colou?red|toned|blue|pink|gray|grey|brown|skinned)", re.I)
_LIGHT_RE = re.compile(_terms_re(LIGHTING_TERMS).pattern + "|" + _LIGHT_WORD_RE.pattern, re.I)
_COLOR_RE = _terms_re(COLOR_TERMS)
_PHOTO_RE = _terms_re(PHOTO_TERMS)
_TIME_RE = _terms_re(TIME_TERMS)
_PLACE_RE = _terms_re(PLACE_TERMS)
_MOOD_NOUN_RE = _terms_re(MOOD_NOUNS)
_PREP_RE = re.compile(r"^(?:" + "|".join(re.escape(p) for p in sorted(PREPS, key=len, reverse=True)) + r")\b",
                      re.I)
_GARMENT_INDEX = sorted([(t, k) for k, terms in GARMENT_TERMS.items() for t in terms],
                        key=lambda x: len(x[0]), reverse=True)
_GARMENT_RE = _terms_re(t for t, _ in _GARMENT_INDEX)
_GARMENT_KEY = {t.lower(): k for t, k in _GARMENT_INDEX}
_SPECIFIC_RES = [(k, _terms_re(terms)) for k, terms in SPECIFIC_GARMENTS]
_KEY_GARMENT_RES = {k: _terms_re(terms) for k, terms in GARMENT_TERMS.items()}
_BODY_LEAD_RE = re.compile(r"^(?:(?:\{her\}|her|his|\{his\}|the)\s+)?(?:(?:left|right|both)\s+)?"
                           r"(?:head|shoulders?|face|body|arms?|hands?|fingers?|legs?|back|chin|palms?|knees?|"
                           r"feet|hips?|waist|torso|neck)(?![\w-])", re.I)
_MALE_RE = re.compile(r"\b(?:his|him|\{him\}|\{his\}|he|man|man’s|man's|male|partner|boyfriend)\b", re.I)


def _words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][\w'/.-]*", text)


def _first_is(low: str, a: "re.Pattern[str]", b: "re.Pattern[str]") -> bool:
    """``a`` 의 단어가 ``b`` 의 단어보다 먼저 나오는지 ('dark hair framing her face' → 헤어가 먼저)."""
    ma, mb = a.search(low), b.search(low)
    return bool(ma and (not mb or ma.start() < mb.start()))


def _plain(text: str) -> str:
    """비교용: {her} → her, 가중치·태그 제거, 소문자."""
    t = re.sub(r"^\s*\d+(?:\.\d+)?::", "", text)
    t = re.sub(r"(?:\s+[#+!][A-Za-z0-9_]+)+\s*$", "", t)
    t = re.sub(r"\{(\w+)\}", r"\1", t)
    return " ".join(t.split()).strip()


def _lead_word(low: str) -> str:
    lead = re.sub(r"^(?:[a-z]+ly\s+)+", "", low)
    w = _words(lead)
    return w[0].lower() if w else ""


def is_participle(text: str) -> bool:
    """'-ing / 과거분사' 로 시작하는 사람 동작 구인지 ('smiling softly', 'seated on a bench')."""
    w = _lead_word(_plain(text).lower())
    if not w or w in ADJ_ING or w in DESCRIBE_ING or w.endswith(("-looking", "lighting")):
        return False
    return (w.endswith("ing") and len(w) > 4) or w in PARTICIPLE_EXTRA


def specific_garment(text: str) -> Optional[str]:
    """의상 '중심' 부분(with/that/over 앞)에 특정 의상 단어가 있으면 그 요소."""
    zone = _head_zone(text)
    for k, rx in _SPECIFIC_RES:
        if rx.search(zone):
            return k
    return None


def garment_key(text: str) -> Optional[str]:
    """의상 중심 명사로 정한 요소 ('a red dress with a lace collar' → dress)."""
    sg = specific_garment(text)
    if sg:
        return sg
    zone = _head_zone(text)
    last = None
    for m in _GARMENT_RE.finditer(zone):
        last = m
    if not last:
        return None
    word = last.group(0).lower()
    key = _GARMENT_KEY.get(word) or _GARMENT_KEY.get(re.sub(r"(?:es|s)$", "", word))
    return key


def _head_zone(text: str) -> str:
    """중심 명사가 있는 앞부분 ('her hair styled with bangs framing her face' → 'her hair styled')."""
    return re.split(r"\s+(?:with|featuring|that|which|over|under|underneath|beneath|paired|tucked|worn|framing|"
                    r"against|near|around|between|across|behind|beside|revealing|while|as|on|in|from|by)\s+|;",
                    _plain(text), maxsplit=1, flags=re.I)[0]


# ------------------------------------------------------------------ 문장 조각
_FRAG_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"\b\d{1,2}\s?age\b|\baged \d+|^[–-]\d+|^\d{1,2}\s*,|\b(?:approximately|about) \d+ years old|"
                r"\b(?:late|early|mid)[- ]\d0s\b|\b\d+'\d+|\b\d+\s+years?\s+old\b", re.I), "나이·키 표기가 섞인 문장 조각"),
    (re.compile(r"\b(?:lips|face|body|eyes|hair|it|she|skin|chest|features)\s+(?:is\s+)?(?:fully\s+|partially\s+|clearly\s+)?"
                r"visible\b|,\s*(?:captured|shot|seen|framed|illuminated|lit|bathed)\b", re.I), "잘못 잘린 문장 조각"),
    (re.compile(r";|,\s*the\b|^[^,]+,\s+with\b"), "여러 문장이 붙은 조각"),
    (re.compile(r"^(?:a|an)\s+(?:she|he|wears|to|only|no|several|sits|stands|likely|visible|unbuttoned|"
                r"partially|expressive eyes|soft$)\b", re.I), "잘못 잘린 문장 조각"),
    (re.compile(r"^(?:she|he|it|they|is|are|was|were|stands|creates|highlights|takes|protagonist|person|"
                r"the (?:image|shot|composition|scene|background|lighting|camera|framing|surrounding|top|subtle|"
                r"fabric|contrast|neutral|smoothness|overall|other|texture|photo|photograph|frame)|"
                r"the\s+[\w-]+\s+of)\b", re.I), "잘못 잘린 문장 조각"),
    (re.compile(r"\S\s+(?:that|with|the|a|an|and|or|of|to|by|as|in|on|while|where|which|from|for|at|into|whose|"
                r"is|are|very|than|beyond|without|toward|towards)$|\b(?:face|body|head|hair|figure)\s+"
                r"(?:fully|partially)$", re.I), "끝이 잘린 문장 조각"),
    (re.compile(r"\b(?:a|an|the|with|and|is)\s+(?:soft|gentle|subtle|warm|natural|clean|candid|serene|bright|even|"
                r"tight|smooth|delicate|dry|light|low)$", re.I), "끝이 잘린 문장 조각"),
    (re.compile(r"^no\b|\bno visible\b|\bnot visible\b|\bwithout (?:any|visible)\b", re.I),
     "'없음'을 뜻하는 문장 (요소가 아님)"),
    (re.compile(r"\b(?:likely|possibly|probably|appears to|seems to|suggesting)\b", re.I), "추측 표현 (요소가 아님)"),
    (re.compile(r"^(?:(?:the|a|an|her|his|\{her\}|\{his\})\s+)?(?:makeup|make-up|hair|face|skin|lips|eyes|expression|"
                r"outfit|clothing|clothes|lighting|camera|background|photo|image|pose|body)$", re.I),
     "너무 일반적인 말 (요소가 아님)"),
    (re.compile(r"^[\w-]+(?:\s+[\w-]+)?,\s*(?:eyes|lips|mouth|face|skin|hair|nose|cheeks|body)$", re.I),
     "잘못 잘린 문장 조각"),
    # 0.10.0 — 와일드카드 분석에서 나온 잘못된 조각
    (re.compile(r"^(?:(?:the|a|an|her|\{her\})\s+)?(?:cheeks|mouth|facial expression|exposed|setting)$|"
                r"^(?:her|\{her\}|s)\s+(?:bare\s+)?legs$|^s\s|^an?\s+(?:so|eyes)\b", re.I), "너무 일반적인 말 · 잘린 조각 (요소가 아님)"),
    (re.compile(r"\b(?:a|an|the|with|and|is|for|of|casting|creating)\s+(?:calm|wide|slight|neutral|relaxed|daring|"
                r"plush|playful|sultry|muted|realistic|long)$|"
                r"\b(?:a|an|the|with|of)\s+(?:pink|red|white|black|blue|green|purple|gold|silver)$|"
                r"\b(?:it|another)$|\ba full-body$", re.I), "끝이 잘린 문장 조각"),
    (re.compile(r"\b(?:bust size|cup size|[a-h]-cup)\b", re.I), "체형 수치 (요소가 아님)"),
    (re.compile(r"[:—]|https?://|www\.|\b(?:this|these)\s+(?:ensures?|creates?|gives?)\b", re.I),
     "라벨·기호가 섞인 조각 ('technical details : …')"),
    (re.compile(r"(?<![\w-])(?:micro-?pores|subject'?s?|woman'?s?|girl|person)(?![\w-])|^(?:hair|skin) texture$", re.I),
     "피사체 · 피부결 묘사가 섞인 조각"),
    (re.compile(r"\S\s+(?:small|large|tiny|big|dark|light|bright|pale|deep)\s+(?:pink|red|white|black|blue|green|purple|"
                r"gold|silver|brown|gray|grey)$", re.I), "끝이 잘린 문장 조각"),
    (re.compile(r"\S\s+wearing\b", re.I), "여러 요소가 붙은 조각 ('… wearing …')"),
    (re.compile(r"\b(?:is|are|was|were)\b", re.I), "문장(동사 포함) · 요소는 짧은 구여야 함"),
    (re.compile(r"\b(?:stands|sits|lies|gazes|holds|crops|occupies|reveals|keeps|emphasizes|accentuates|suggests|"
                r"creates|enhances|centers|conveys|shows|blurs|falls|rests|touches|looks|kisses|streams|takes|"
                r"drifts|filters|illuminates|fills|casts|adds|gives|leans|frames|captures|depicts|features|"
                r"wears|has|appears|seems|includes|include)\b", re.I), "문장(동사 포함) · 요소는 짧은 구여야 함"),
]
_TRIM_LEAD = [
    re.compile(r"^\d+\s+years?\s+old\s*,?\s*(?:with\s+)?", re.I),
    re.compile(r"^(?:\d{1,2}\s?age|aged\s+\d+(?:\s*[–-]\s*\d+)?|\d{1,2})(?:\s+(?:person|woman|girl))?\s*,?\s*"
               r"(?:with\s+)?", re.I),
    re.compile(r"^(?:the\s+|a\s+)?(?:protagonist|person|woman|girl|model)\s*,?\s+with\s+", re.I),
    re.compile(r"^(?:face|upper back|upper neck|neck|body|torso)\s*,\s*", re.I),
    re.compile(r"^(?:a|an)\s+(?:she\s+wears|wears|only|to the \w+,?)\s*", re.I),
    re.compile(r"^(?:she|he)\s+(?:wears|is wearing|is in|is|has)\s+", re.I),
    re.compile(r"^(?:is|are)\s+(?:captured|visible|framed|shown)\s*(?:in|from|with)?\s+", re.I),
    re.compile(r"^(?:bathed|captured|illuminated|lit|shot|rendered|set)\s+(?:in|by|with|under|against)\s+", re.I),
    re.compile(r"^(?:complemented|accompanied|paired|accented|adorned)\s+(?:by|with)\s+", re.I),
]
_TRIM_TAIL = [
    re.compile(r"\s+(?:that|which|where|while|as|whose)\b.*$", re.I),
    re.compile(r"\s+(?:rests|falls|stands|sits|lies|gazes|holds|touches|keeps|blurs|centers|conveys|shows|is|are|frames|"
               r"occupies|crops|reveals|emphasizes|enhances|highlights|creates|looks|streams|drifts)\b.*$", re.I),
    re.compile(r",?\s+with\s+(?:an?\s+)?(?:gentle|soft|sweet|serene|subtle|warm|natural|pink|clean|bright)$", re.I),
    re.compile(r"[,\s]+(?:with|and|or|the|a|an|of|to|by|in|on|from|for|at|into|is|under|soft|gentle|subtle)$", re.I),
    re.compile(r"\s+(?:fully|partially)$", re.I),
    re.compile(r"\s+with\s+(?:the|her|\{her\})\s+face$", re.I),
]
_LIGHTING_IS_RE = re.compile(r"^(?:the\s+)?lighting\s+(?:is\s+)?([\w-]+)$", re.I)


def fragment_reason(text: str) -> str:
    """문장 조각이면 이유(한글), 아니면 빈 문자열."""
    t = _plain(text)
    if not t:
        return "빈 줄"
    for rx, why in _FRAG_PATTERNS:
        if rx.search(t):
            return why
    w = _lead_word(t.lower())
    if w in DESCRIBE_ING:
        return "장면 설명 분사구 (사람 동작이 아님)"
    if len(_words(t)) > MAX_WORDS or len(t) > MAX_CHARS:
        return f"요소로 쓰기엔 긴 문장 ({len(_words(t))}단어)"
    return ""


def trim_fragment(text: str) -> str:
    """잘린 문장 조각을 요소 문구로 다듬기 ('bathed in warm sunlight that' → 'warm sunlight').

    ``{her}`` 같은 자리표시자는 그대로 둡니다. 다듬을 수 없으면 원래 문구를 돌려줍니다.
    """
    t = " ".join(str(text).split()).strip(" ,.;")
    m = _LIGHTING_IS_RE.match(t)
    if m:
        return f"{m.group(1)} lighting"
    t = t.split(";")[0].strip(" ,")
    for _ in range(4):
        before = t
        for rx in _TRIM_LEAD:
            t = rx.sub("", t).strip(" ,")
        for rx in _TRIM_TAIL:
            t = rx.sub("", t).strip(" ,")
        if t == before:
            break
    return t


# ------------------------------------------------------------------ 요소 형식
def fits(key: str, text: str) -> bool:
    """``text`` 가 ``key`` 요소의 형식·사전에 맞는지 (``CHECK_KEYS`` 밖의 요소는 항상 True)."""
    if key not in CHECK_KEYS:
        return True
    t = _plain(text)
    low = t.lower()
    words = _words(low)
    if not words:
        return False
    if key in POSE_GROUP:
        return is_participle(t) or (key == "pose" and bool(re.match(r"^(?:on (?:all fours|{?her}? \w+)|"
                                                                     r"head\b|shoulders\b)", low)))
    if key in GARMENT_KEYS:
        if re.search(r"\bwine glass", low):
            return False
        if key == "legwear" and re.search(r"\blegs?\b", low) and not _BODY_LEAD_RE.match(low):
            return True
        sg = specific_garment(t)
        if sg and sg != key and key in GENERAL_GARMENTS and sg in ("swimwear", "lingerie", "sleepwear"):
            return False  # 'a teal strapless bikini top' 은 상의가 아니라 수영복
        if _KEY_GARMENT_RES[key].search(low) or any(w in low for w in GARMENT_CONTEXT.get(key, ())):
            return True
        return garment_key(t) == key
    if key == "hair":
        if re.search(r"\bvellus\b|peach fuzz", low):
            return False
        return bool(_HAIR_RE.search(low)) and not garment_key(t) and not _BODY_LEAD_RE.match(low) \
            and not _first_is(low, _CAMERA_RE, _HAIR_RE) and not _first_is(low, _MAKEUP_RE, _HAIR_RE)
    if key == "makeup":
        return bool(_MAKEUP_RE.search(low)) and not _BODY_LEAD_RE.match(low) and not is_participle(t)
    if key == "expression":
        return bool(_EXPR_RE.search(low) or re.search(r"\beyes\b", low)) and not garment_key(t) \
            and (not is_participle(t) or bool(_EXPR_RE.fullmatch(_lead_word(low)))) and len(words) <= 8
    if key == "location":
        if garment_key(t) or re.match(r"(?i:^(?:on|in|at)\s+an?\s+)(?:[\w-]+\s+)?[a-z]*[^aus\WA-Z]s(?:\s|$)", t):
            return False  # 'on a transparent mesh thong set' · 'on a palm trees lining sidewalk' · 'on a kneels on a bed'
        if re.match(r"^(?:outdoors|indoors|outside|inside)\b", low):
            return True
        if not _PREP_RE.match(low):
            return bool(_PLACE_RE.search(low)) and len(words) <= 6 and not _LIGHT_RE.search(low)
        body = _PREP_RE.sub("", low, count=1).strip()
        if _LIGHT_RE.search(body) and not _PLACE_RE.search(body):
            return False
        if _PLACE_RE.search(body):
            return True
        return not (_TIME_RE.search(body) or _CAMERA_RE.search(low))
    if key == "time_weather":
        return bool(_TIME_RE.search(low))
    if key == "lighting":
        if _BODY_LEAD_RE.match(low) or garment_key(t) or _MAKEUP_PRODUCT_RE.search(low):
            return False
        head = _head_zone(re.sub(r"^(?:bathed|lit|illuminated|washed|backlit)\s+(?:in|by|with)\s+", "", t)).lower()
        if _LIGHT_RE.search(head):
            return not _MOOD_NOUN_RE.search(head)
        if re.search(r"\bunder\b", low) or _MOOD_NOUN_RE.search(low):
            return False
        return bool(_LIGHT_RE.search(low)) and not (_HAIR_RE.search(head) or _MAKEUP_RE.search(head)
                                                    or _EXPR_RE.search(head) or _PHOTO_RE.search(head)
                                                    or _PLACE_RE.search(head) or _CAMERA_RE.search(head))
    if key == "camera":
        if re.match(r"^(?:her\s+|\{her\}\s+)?upper body\s+(?!shot|portrait|framing|view|crop|close-up)", low):
            return False
        return bool(_CAMERA_RE.search(low)) and not _BODY_LEAD_RE.match(low) and not _first_is(low, _HAIR_RE, _CAMERA_RE) \
            and not _first_is(low, _EXPR_RE, _CAMERA_RE)
    if key == "color_grade":
        if re.search(r"^(?:evenly\s+)?toned$|\btoned\s+(?:build|physique|thighs?|body|figure|appearance|abs|legs|arms|"
                     r"stomach|midriff)\b|\bwood grain\b|\blip\b", low):
            return False
        return bool(_COLOR_RE.search(low) or re.search(r"tint|toned\b", low) or re.search(r"\b(?:warm|cool|cold|teal|orange)\b", low)
                    and not _LIGHT_RE.search(low)) and not _MAKEUP_PRODUCT_RE.search(low) and not garment_key(t)
    if key == "photo_style":
        return bool(_PHOTO_RE.search(low))
    if key == "skin":
        if low in SKIN_REJECT or re.search(r"\b(?:penis|forearms?|possess\w*|droplets?)\b|^(?:her|the)\b", low):
            return False
        return len(words) <= 6 and not (_BODY_LEAD_RE.match(low) or garment_key(t) or _HAIR_RE.search(low)
                                        or _PREP_RE.match(low) or _MAKEUP_PRODUCT_RE.search(low))
    if key in ("beauty", "mood"):
        if low in MOOD_REJECT:
            return False
        return len(words) <= 4 and not (_BODY_LEAD_RE.match(low) or garment_key(t) or _HAIR_RE.search(low)
                                        or _PREP_RE.match(low) or _CAMERA_RE.search(low) or is_participle(t))
    return True


def suggest_key(text: str, keys: Optional[Iterable[str]] = None) -> Optional[str]:
    """문구에 가장 맞는 요소 (못 찾으면 None). ``keys`` 를 주면 그 안에서만 고릅니다."""
    allowed = set(keys) if keys is not None else None

    def ok(k: str) -> bool:
        return (allowed is None or k in allowed) and fits(k, text)

    t = _plain(text)
    low = t.lower()
    head = _head_zone(t).lower()
    if is_participle(t):
        k = "pose" if _lead_word(low) in POSE_WORDS else "action"
        for kk in (k, "action" if k == "pose" else "pose"):
            if ok(kk):
                return kk
        return None
    if _PREP_RE.match(low):
        for k in ("location", "time_weather", "lighting", "camera"):
            if ok(k):
                return k
        return None
    gk = garment_key(t)
    order: List[str] = []
    if gk:
        order.append(gk)
    if _HAIR_RE.search(head):
        order.append("hair")
    if _MAKEUP_PRODUCT_RE.search(low):
        order.append("makeup")
    if _EXPR_RE.search(head):
        order.append("expression")
    if _CAMERA_RE.search(head):
        order.append("camera")
    if _LIGHT_RE.search(head):
        order.append("lighting")
    if re.search(r"\bskin(?:\s+tone)?$", low):
        order.append("skin")
    order += ["makeup", "expression", "camera", "lighting", "color_grade", "photo_style", "location",
              "time_weather"]
    m = re.match(r"^(?:an?\s+)?([\w-]+)\s+(?:" + "|".join(MOOD_NOUNS) + r")$", low)
    if m:
        order.insert(0, "mood")
    seen = set()
    for k in order:
        if k in seen:
            continue
        seen.add(k)
        if ok(k):
            return k
    return None


def check_item(key: str, text: str, keys: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """요소 ``key`` 에 들어갈 ``text`` 를 검사합니다 (설명은 모듈 맨 위).

    ``keys`` 는 옮겨 갈 수 있는 요소 목록 (없으면 ``CHECK_KEYS`` 전체).
    """
    raw = " ".join(str(text or "").split()).strip()
    res = {"status": "ok", "key": key, "text": raw, "reason": ""}
    if key not in CHECK_KEYS or not raw:
        return res
    targets = [k for k in (keys if keys is not None else CHECK_KEYS) if k in CHECK_KEYS]
    why = fragment_reason(raw)
    if why:
        if _MALE_RE.search(_plain(raw)):
            return {**res, "status": "fragment", "reason": why + " · 남녀 묘사"}
        fixed = trim_fragment(raw)
        if not fixed or fixed == raw or fragment_reason(fixed) or len(_words(fixed)) < 2 \
                and key not in ("skin", "mood", "beauty") \
                or _plain(fixed).lower() in ("her hair", "hair", "her", "the camera", "the framing") \
                or re.search(r",\s*[\w-]+$", fixed) and len(_words(fixed)) <= 4:
            return {**res, "status": "fragment", "reason": why}
        if fits(key, fixed):
            return {**res, "status": "fixed", "text": fixed, "reason": why + " → 다듬음"}
        k2 = suggest_key(fixed, targets)
        if k2 and not (k2 in POSE_GROUP and key in POSE_GROUP):
            return {**res, "status": "moved", "key": k2, "text": fixed, "reason": why + f" → 다듬어서 '{k2}'"}
        return {**res, "status": "fragment", "reason": why}
    if fits(key, raw):
        return res
    k2 = suggest_key(raw, targets)
    if k2 and "," in raw and any(fits(key, p) for p in raw.split(",")[1:]):
        return {**res, "status": "wrong_key", "reason": f"여러 요소가 섞인 줄 ('{key}' + '{k2}')"}
    if k2 and k2 != key and not (k2 in POSE_GROUP and key in POSE_GROUP):
        return {**res, "status": "moved", "key": k2, "reason": f"'{key}' 형식에 맞지 않음 → '{k2}'"}
    return {**res, "status": "wrong_key", "reason": f"'{key}' 형식에 맞지 않고 맞는 요소를 찾지 못함"}


def is_clean_phrase(key: str, text: str) -> bool:
    """불러오기·생성 때 쓰는 가벼운 검사: 문장 조각이 아니면 True (형식 불일치는 따지지 않음)."""
    return key not in CHECK_KEYS or not fragment_reason(text)


# ------------------------------------------------------------------ 요소 파일 정리
_LINE_RE = re.compile(r"^(\s*\d+(?:\.\d+)?::)?(.*?)((?:\s+[#+!][A-Za-z0-9_]+)*)\s*$")
MOVED_HEADER = "# 요소 검사로 옮겨온 항목 ({date})"


def _norm(text: str) -> str:
    t = re.sub(r"[\s,.;]+", " ", _plain(text).lower()).strip()
    return re.sub(r"^(?:a|an|the) ", "", t)


def scan_element_files(data_dir: str) -> List[Dict[str, str]]:
    """``<data_dir>/elements/*.txt`` 를 검사해 바꿀 줄 목록을 돌려줍니다 (파일은 바꾸지 않음).

    각 행: ``{"file", "line_no", "line", "key", "status", "to", "text", "reason"}``
    (status: fixed / moved / fragment / wrong_key). ``wrong_key`` 는 정리할 때도 그대로 둡니다.
    """
    import os

    edir = os.path.join(data_dir, "elements")
    rows: List[Dict[str, str]] = []
    if not os.path.isdir(edir):
        return rows
    keys = [f[:-4] for f in os.listdir(edir) if f.endswith(".txt") and "." not in f[:-4]]
    for fname in sorted(os.listdir(edir)):
        key = fname.split(".")[0]
        if not fname.endswith(".txt") or key not in CHECK_KEYS:
            continue
        with open(os.path.join(edir, fname), encoding="utf-8") as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            m = _LINE_RE.match(line)
            text = (m.group(2) if m else line).strip()
            r = check_item(key, text, keys)
            if r["status"] == "ok":
                continue
            rows.append({"file": fname, "line_no": i + 1, "line": line, "key": key, "status": r["status"],
                         "to": r["key"], "text": r["text"], "reason": r["reason"]})
    return rows


def clean_element_files(data_dir: str, apply: bool = False, backup_dir: Optional[str] = None) -> Dict[str, object]:
    """사용자 요소 파일 정리: 다듬기(fixed) · 맞는 요소로 옮기기(moved) · 문장 조각 지우기(fragment).

    ``apply=False`` 면 미리보기만 합니다. 적용하면 먼저 ``<data_dir>/backups/elements_<시각>/`` 에
    elements 폴더 전체를 백업하고 ``_manifest.json`` 에 바꾼 내용을 남깁니다 (AI Gallery Manager 의
    '요소 정리 · 되돌리기' 와 같은 백업 형식). ``wrong_key`` 줄은 지우지 않고 그대로 둡니다.
    """
    import datetime
    import json
    import os
    import shutil

    rows = scan_element_files(data_dir)
    summary: Dict[str, int] = {}
    for r in rows:
        summary[r["status"]] = summary.get(r["status"], 0) + 1
    out: Dict[str, object] = {"items": rows, "summary": summary, "backup": "", "applied": False}
    todo = [r for r in rows if r["status"] in ("fixed", "moved", "fragment")]
    if not apply or not todo:
        return out
    edir = os.path.join(data_dir, "elements")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = backup_dir or os.path.join(data_dir, "backups", f"elements_{stamp}")
    shutil.copytree(edir, backup, dirs_exist_ok=True)
    with open(os.path.join(backup, "_backup_info.json"), "w", encoding="utf-8") as f:
        json.dump({"created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                   "note": "요소 검사 정리 (PromptMaker)"}, f, ensure_ascii=False, indent=1)
    by_file: Dict[str, List[Dict[str, str]]] = {}
    for r in todo:
        by_file.setdefault(r["file"], []).append(r)
    add: Dict[str, List[str]] = {}
    for fname, lst in by_file.items():
        path = os.path.join(edir, fname)
        with open(path, encoding="utf-8", newline="") as f:
            content = f.read()
        eol = "\r\n" if "\r\n" in content else "\n"
        lines = content.splitlines()
        change = {r["line_no"] - 1: r for r in lst}
        new_lines = []
        for i, line in enumerate(lines):
            r = change.get(i)
            if r is None or line != r["line"]:
                new_lines.append(line)
                continue
            m = _LINE_RE.match(line)
            weight, tags = (m.group(1) or "", m.group(3) or "") if m else ("", "")
            if r["status"] == "fixed":
                new_lines.append(f"{weight}{r['text']}{tags}")
            elif r["status"] == "moved":
                add.setdefault(r["to"], []).append(f"{weight}{r['text']}{tags}")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(eol.join(new_lines) + (eol if new_lines else ""))
    header = MOVED_HEADER.format(date=datetime.date.today().isoformat())
    added: Dict[str, List[str]] = {}
    for key, vals in add.items():
        path = os.path.join(edir, f"{key}.txt")
        content = ""
        if os.path.isfile(path):
            with open(path, encoding="utf-8", newline="") as f:
                content = f.read()
        eol = "\r\n" if "\r\n" in content else "\n"
        have = {_norm(_LINE_RE.match(l).group(2)) for l in content.splitlines()
                if l.strip() and not l.lstrip().startswith("#")}
        new = []
        for v in vals:
            n = _norm(_LINE_RE.match(v).group(2))
            if n and n not in have:
                have.add(n)
                new.append(v)
        if not new:
            continue
        chunk = "" if (not content or content.endswith("\n")) else eol
        if header not in content:
            chunk += header + eol
        with open(path, "a", encoding="utf-8", newline="") as f:
            f.write(chunk + eol.join(new) + eol)
        added[key] = new
    with open(os.path.join(backup, "_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"applied_at": datetime.datetime.now().isoformat(timespec="seconds"),
                   "note": "요소 검사 정리 (PromptMaker)", "done": todo, "added": added},
                  f, ensure_ascii=False, indent=1)
    out.update({"backup": backup, "applied": True, "added": added})
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m promptmaker.element_check <사용자 데이터 폴더> [--apply]``"""
    import argparse

    ap = argparse.ArgumentParser(description="PromptMaker 요소 파일 검사 · 정리")
    ap.add_argument("data_dir", help="사용자 데이터 폴더 (elements/ 가 들어 있는 폴더)")
    ap.add_argument("--apply", action="store_true", help="백업 후 다듬기 · 옮기기 · 조각 지우기 적용")
    args = ap.parse_args(argv)
    res = clean_element_files(args.data_dir, apply=args.apply)
    label = {"fixed": "다듬기", "moved": "옮기기", "fragment": "지우기", "wrong_key": "확인 필요(그대로 둠)"}
    for r in res["items"]:  # type: ignore[union-attr]
        dest = f" → [{r['to']}] {r['text']}" if r["status"] in ("fixed", "moved") else ""
        print(f"[{label[r['status']]}] {r['file']}:{r['line_no']} {r['line'].strip()}{dest}  ({r['reason']})")
    print(f"\n요약: {res['summary']}")
    if res["applied"]:
        print(f"적용 완료 · 백업: {res['backup']}")
    elif args.apply:
        print("바꿀 줄이 없습니다.")
    else:
        print("미리보기만 했습니다. 적용하려면 --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
