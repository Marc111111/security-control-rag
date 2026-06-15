from __future__ import annotations

import re

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore (?:all )?(?:previous|prior|above) instructions", re.I),
    re.compile(r"you are now .+", re.I),
    re.compile(r"system prompt", re.I),
    re.compile(r"developer message", re.I),
    re.compile(r"reveal (?:the )?(?:prompt|instructions|secrets)", re.I),
]


class HumanTextSanitizer:
    """Conservative sanitizer for human-generated questionnaire text.

    This does not decide truth. It removes control characters and neutralizes common prompt
    injection phrases before the text is placed into a downstream LLM prompt.
    """

    def sanitize(self, value: str) -> str:
        text = _strip_control_characters(value)
        text = re.sub(r"\s+", " ", text).strip()
        for pattern in PROMPT_INJECTION_PATTERNS:
            text = pattern.sub("[removed instruction-like text]", text)
        return text[:4_000]


def _strip_control_characters(value: str) -> str:
    return "".join(char for char in value if char in "\n\r\t" or ord(char) >= 32)

