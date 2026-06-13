from secure_rag.prompts import build_grounded_prompt
from secure_rag.schema import Chunk, RetrievalHit


def test_grounded_prompt_includes_sources_and_criteria() -> None:
    hit = RetrievalHit(
        chunk=Chunk(
            id="abc",
            text="Maintain offline backups.",
            metadata={"source_path": "nist.md", "page": 4},
        ),
        score=0.91,
    )

    messages = build_grounded_prompt("ransomware tier 2", [hit])

    assert messages[0]["role"] == "system"
    assert "Use only the retrieved source excerpts" in messages[0]["content"]
    assert "ransomware tier 2" in messages[1]["content"]
    assert "[S1]" in messages[1]["content"]
    assert "nist.md#page=4" in messages[1]["content"]

