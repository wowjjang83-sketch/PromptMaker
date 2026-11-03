"""English grammar helpers used to turn filled templates into natural sentences."""
from __future__ import annotations

import re
from typing import Iterable, List

NUMBER_WORDS = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}

# Words that start with a vowel letter but a consonant sound, and vice versa.
_A_EXCEPTIONS = ("uni", "use", "usu", "uro", "eu", "one", "once", "ufo", "ukr", "uti")
_AN_EXCEPTIONS = ("hour", "honest", "honor", "honour", "heir", "herb")


def number_word(n: int) -> str:
    return NUMBER_WORDS.get(n, str(n))


def indefinite_article(phrase: str) -> str:
    """Return 'a' or 'an' for the given phrase."""
    word = phrase.strip().lower()
    if not word:
        return "a"
    if word[0].isdigit():
        # 8, 11, 18, 80 ... are read with a vowel sound
        return "an" if re.match(r"^(8|11|18)(\D|$)|^8\d", word) else "a"
    if word.startswith(_AN_EXCEPTIONS):
        return "an"
    if word.startswith(_A_EXCEPTIONS):
        return "a"
    return "an" if word[0] in "aeiou" else "a"


def with_article(phrase: str) -> str:
    phrase = phrase.strip()
    if not phrase:
        return phrase
    return f"{indefinite_article(phrase)} {phrase}"


def join_list(items: Iterable[str], conj: str = "and") -> str:
    items = [i.strip() for i in items if i and i.strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conj} {items[1]}"
    return ", ".join(items[:-1]) + f", {conj} {items[-1]}"


def capitalize_first(text: str) -> str:
    for i, ch in enumerate(text):
        if ch.isalpha():
            return text[:i] + ch.upper() + text[i + 1:]
    return text


_ARTICLE_RE = re.compile(r"\b(a|an|A|An)\s+([\"']?)([A-Za-z0-9][\w-]*)")


def fix_articles(text: str) -> str:
    """Correct a/an in front of every word in the text."""

    def repl(m: re.Match) -> str:
        art, quote, word = m.group(1), m.group(2), m.group(3)
        correct = indefinite_article(word)
        if art[0].isupper():
            correct = correct.capitalize()
        return f"{correct} {quote}{word}"

    return _ARTICLE_RE.sub(repl, text)


def tidy(text: str) -> str:
    """Clean whitespace/punctuation and capitalise sentence starts."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"([.!?])\s*[,.]+", r"\1", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"([,.;:!?])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = fix_articles(text.strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(capitalize_first(s) for s in sentences if s).strip()


def ensure_sentence(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text


def simple_lemma(word: str) -> str:
    """Very small stemmer: plural/verb suffixes -> base form."""
    w = word.lower()
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        base = w[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 3 and w.endswith("es") and w[-3] in "sxz":
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower())
