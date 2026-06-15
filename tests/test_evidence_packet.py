from app.retrieval.evidence_packet import select_prompt_evidence
from app.schemas import DocumentChunk, RetrievedEvidence


def test_prompt_evidence_selector_keeps_relevant_controls_and_drops_endpoint_noise() -> None:
    question = "No endpoint anti-malware solution is currently deployed."
    evidence = [
        _hit(
            "c1",
            1.0,
            "CIS Safeguard 10.1: Deploy and Maintain Anti-Malware Software.",
            control_id="10.1",
        ),
        _hit(
            "c2",
            0.98,
            "SCF Control: Malicious Code Protection (Anti-Malware) SCF #: END-04.",
            control_id="END-04",
        ),
        _hit(
            "c3",
            0.97,
            "Safeguard 9.7: Deploy and Maintain Email Server Anti-Malware Protections.",
            control_id="9.7",
        ),
        _hit(
            "c4",
            0.96,
            "Business stakeholders and process owners appropriately scope END domain capabilities.",
            control_id="3",
        ),
    ]

    selected = select_prompt_evidence(question, evidence, max_items=3)

    selected_text = " ".join(hit.chunk.text for hit in selected).lower()
    assert "safeguard 10.1" in selected_text
    assert "end-04" in selected_text
    assert "email server anti-malware" not in selected_text
    assert "business stakeholders" not in selected_text
    assert len(selected) <= 3
    assert all(len(hit.chunk.text) <= 526 for hit in selected)


def _hit(
    chunk_id: str,
    score: float,
    text: str,
    *,
    control_id: str,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        chunk=DocumentChunk(
            id=chunk_id,
            text=text,
            metadata={
                "source_path": "standards/example.pdf",
                "framework": "CIS Controls",
                "control_id": control_id,
            },
        ),
        score=score,
        source="standards/example.pdf",
        retrieval_method="keyword",
    )
