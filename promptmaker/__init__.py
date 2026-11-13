"""PromptMaker - LLM 없이 규칙/템플릿 기반으로 영문 자연어 프롬프트를 생성하는 라이브러리."""
from .core import MODES, PROMPT_TYPES, GenerateOptions, LLMRefiner, PromptMaker, PromptResult
from .safety import MIN_AGE, SafetyError
from .translator import DictionaryTranslator, GoogleTranslator, NoTranslator, is_english

__version__ = "0.4.0"

__all__ = [
    "PromptMaker", "GenerateOptions", "PromptResult", "LLMRefiner", "MODES", "PROMPT_TYPES",
    "SafetyError", "MIN_AGE", "GoogleTranslator", "DictionaryTranslator", "NoTranslator",
    "is_english", "generate",
]

_default_maker = None


def generate(user_input: str = "", **options) -> PromptResult:
    """Shortcut: ``promptmaker.generate("비 오는 카페", prompt_type="image")``."""
    global _default_maker
    if _default_maker is None:
        _default_maker = PromptMaker()
    return _default_maker.generate(user_input, **options)
