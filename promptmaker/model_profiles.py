"""Target-model profiles: reshape the rule-based prompt into the style a model expects.

Profiles live in ``data/model_profiles/<id>.json`` (user-editable). Formats:

* ``prose``  - natural-language paragraph (Krea 2, Qwen-Image 2.1, Flux ...)
* ``tags``   - comma-separated tags in a fixed order (MOSS-Image tag mode, SDXL ...)
* ``h3``     - MiniMax H3 sections: integrated_multimodal_description / overall_soundscape /
               non_diegetic_music
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_TAG_ORDER = ["photo_style", "video_style", "subject", "hair", "makeup", "skin", "attire",
                     "attire_state", "pose", "action", "motion", "expression", "interaction", "location",
                     "time_weather", "lighting", "lighting_effect", "camera", "camera_movement",
                     "color_grade", "mood"]

# scene tag -> a sentence of sound (for the H3 overall_soundscape section)
DEFAULT_SOUNDS = [
    ("rain", "Steady rain patters on the ground and drips from nearby edges."),
    ("beach", "Waves wash gently onto the shore while a light sea breeze blows."),
    ("pool", "Water laps softly against the pool edge."),
    ("cafe", "Low café chatter mixes with the soft clink of cups."),
    ("bar", "Muffled conversation and glasses clinking fill the room."),
    ("street", "Distant traffic hums and footsteps pass on the pavement."),
    ("urban", "The city hums faintly in the distance."),
    ("nature", "Birds chirp and leaves rustle in the wind."),
    ("studio", "The studio is quiet except for the soft click of a camera shutter."),
    ("bedroom", "The room is quiet, with only the soft rustle of fabric."),
    ("indoor", "A quiet room tone with faint fabric rustles."),
]


@dataclass
class ModelProfile:
    id: str
    label: str = ""
    type: str = "image"               # image | video | any
    default_for: str = ""             # "image" / "video": used when no model is chosen
    format: str = "prose"             # prose | tags | h3
    description: str = ""
    max_chars: int = 0                # 0 = no limit
    prefix: str = ""
    suffix: str = ""
    negative: str = ""                # separate negative prompt (models that use one)
    inline_exclusions: str = ""       # models without a negative field: "no text, no watermark"
    drop_phrases: List[str] = field(default_factory=list)
    replace: Dict[str, str] = field(default_factory=dict)
    tag_order: List[str] = field(default_factory=lambda: list(DEFAULT_TAG_ORDER))
    quality_tags: str = ""
    music: str = "N/A"
    sounds: List[Tuple[str, str]] = field(default_factory=lambda: list(DEFAULT_SOUNDS))
    notes: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, pid: str, data: Dict[str, Any]) -> "ModelProfile":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k not in ("id", "raw")}
        if "sounds" in known:
            known["sounds"] = [tuple(x) for x in known["sounds"]]
        return cls(id=data.get("id", pid), raw=data, **known)

    def to_info(self) -> Dict[str, Any]:
        return {"id": self.id, "label": self.label or self.id, "type": self.type, "format": self.format,
                "default_for": self.default_for,
                "description": self.description}


@dataclass
class Formatted:
    prompt: str
    negative: str = ""


def _clean_slot_text(text: str) -> str:
    text = re.sub(r"\{(her|her_obj|his)\}", "her", text)
    text = re.sub(r"\{(she|She)\}", "she", text)
    text = re.sub(r"\{(herself)\}", "herself", text)
    text = re.sub(r"\{[^}]*\}", "", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,")


def _limit(text: str, max_chars: int) -> str:
    if not max_chars or len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    end = max(cut.rfind(". "), cut.rfind(", "))
    return (cut[:end + 1] if end > max_chars * 0.6 else cut).rstrip(" ,")


def _apply_common(text: str, p: ModelProfile) -> str:
    for phrase in p.drop_phrases:
        text = re.sub(re.escape(phrase), "", text, flags=re.I)
    for a, b in p.replace.items():
        text = re.sub(r"\b" + re.escape(a) + r"\b", b, text, flags=re.I)
    return re.sub(r"\s{2,}", " ", text).strip()


def _subject(picks: Dict[str, str], prompt_type: str) -> str:
    bits = [picks.get("beauty", ""), f"{picks['age']}-year-old" if picks.get("age") else "",
            picks.get("nationality", ""), "woman"]
    cast = picks.get("casting", "")
    subj = " ".join(b for b in bits if b)
    if cast and cast != "one woman":
        subj = f"{cast}, {subj}"
    return subj


def format_prompt(prompt: str, picks: Dict[str, str], scene_tags: Iterable[str], p: Optional[ModelProfile],
                  prompt_type: str = "image") -> Formatted:
    if p is None:
        return Formatted(prompt)
    negative = p.negative
    if p.format == "tags":
        out: List[str] = [p.quality_tags] if p.quality_tags else []
        for key in p.tag_order:
            val = _subject(picks, prompt_type) if key == "subject" else picks.get(key, "")
            if not val:
                continue
            val = _clean_slot_text(val)
            if key == "skin" and "skin" not in val:
                val += " skin"
            if key == "mood":
                val += " mood"
            out.append(val)
        text = _apply_common(", ".join(x for x in out if x), p)
    elif p.format == "h3":
        body = _apply_common(prompt, p)
        tags = set(scene_tags)
        sounds = [s for t, s in p.sounds if t in tags][:2] or ["A quiet ambient room tone."]
        text = ("integrated_multimodal_description: " + body + "\n"
                "overall_soundscape: " + " ".join(sounds) + "\n"
                "non_diegetic_music: " + (p.music or "N/A"))
    else:
        text = _apply_common(prompt, p)
        if p.inline_exclusions:
            text = text.rstrip(". ") + ". " + p.inline_exclusions.rstrip(".") + "."
    text = " ".join(x for x in (p.prefix.strip(), text, p.suffix.strip()) if x) if p.format != "h3" else text
    return Formatted(_limit(text, p.max_chars), negative)
