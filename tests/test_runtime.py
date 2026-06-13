import json

import pytest

from secure_rag.runtime import build_criteria


def test_build_criteria_accepts_natural_language_only() -> None:
    assert build_criteria(message="ransomware controls") == "ransomware controls"


def test_build_criteria_appends_structured_context() -> None:
    criteria = build_criteria(
        message="recommend controls",
        context={"tier": 2, "risk": "ransomware"},
    )

    assert "recommend controls" in criteria
    assert "Structured criteria JSON:" in criteria
    assert json.dumps({"risk": "ransomware", "tier": 2}, indent=2) in criteria


def test_build_criteria_rejects_empty_message() -> None:
    with pytest.raises(ValueError, match="message"):
        build_criteria(message="  ")

