from app.generation.prompts import build_structured_answer_prompt
from app.planning.planner import RiskQuestionPlanner
from app.schemas import DocumentChunk, RetrievedEvidence
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


def test_structured_answer_prompt_uses_filtered_graph_hints_not_raw_graph() -> None:
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(
                id="chunk-1",
                text="CIS Safeguard 10.1 requires anti-malware software.",
                metadata={"source_path": "standards/cis.pdf"},
            ),
            score=1.0,
            source="standards/cis.pdf",
            retrieval_method="keyword",
        )
    ]
    plan = RiskQuestionPlanner().plan("No anti-malware is deployed.")
    messages = build_structured_answer_prompt(
        "No anti-malware is deployed.",
        plan,
        evidence,
        [
            {
                "evidence": "S1",
                "relationship": "CONTROL_MITIGATES_RISK",
                "source_type": "Control",
                "source": "CIS Safeguard 10.1",
                "target_type": "Risk",
                "target": "Malware infection",
            }
        ],
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert "Retrieved text evidence is authoritative" in prompt
    assert (
        "S1: Control 'CIS Safeguard 10.1' CONTROL_MITIGATES_RISK Risk "
        "'Malware infection'"
    ) in prompt
    assert "source_chunk_id" not in prompt


def test_structured_answer_prompt_keeps_retrieved_evidence_compact() -> None:
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(
                id=f"chunk-{index}",
                text="Important control text. " + ("x" * 4_000),
                metadata={"source_path": f"standards/source-{index}.pdf"},
            ),
            score=1.0,
            source=f"standards/source-{index}.pdf",
            retrieval_method="keyword",
        )
        for index in range(8)
    ]
    plan = RiskQuestionPlanner().plan("No disaster recovery test report was provided.")

    messages = build_structured_answer_prompt(
        "No disaster recovery test report was provided.",
        plan,
        evidence,
        [],
    )
    prompt = "\n".join(message["content"] for message in messages)

    assert "Search focus:" in prompt
    assert '"sub_questions"' not in prompt
    assert len(prompt) < 12_000
