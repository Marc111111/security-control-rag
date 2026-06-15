from app.retrieval.graph_context import build_graph_context_rows, graph_context_prompt_text
from app.schemas import DocumentChunk, RetrievedEvidence


def test_graph_context_keeps_only_source_linked_readable_rows() -> None:
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(id="chunk-good", text="Anti-malware evidence."),
            score=1.0,
            source="standards/cis.pdf",
            retrieval_method="keyword",
        )
    ]
    rows = [
        {
            "source_chunk_id": "chunk-good",
            "relationship": "CONTROL_MITIGATES_RISK",
            "source": {"name": "CIS Safeguard 10.1", "type": "Control"},
            "target": {"name": "Malware infection", "type": "Risk"},
        },
        {
            "source_chunk_id": "chunk-good",
            "relationship": "CONTROL_MITIGATES_RISK",
            "source": {"name": "CIS Controls 4.0", "type": "Control"},
            "target": {"name": "2a RISK-2b RISK-2c RISK-2g", "type": "Risk"},
        },
        {
            "source_chunk_id": "chunk-other",
            "relationship": "CONTROL_MITIGATES_RISK",
            "source": {"name": "Endpoint protection", "type": "Control"},
            "target": {"name": "Malware infection", "type": "Risk"},
        },
    ]

    cleaned = build_graph_context_rows(rows, evidence)

    assert cleaned == [
        {
            "evidence": "S1",
            "relationship": "CONTROL_MITIGATES_RISK",
            "source_type": "Control",
            "source": "CIS Safeguard 10.1",
            "target_type": "Risk",
            "target": "Malware infection",
        }
    ]
    assert "chunk-other" not in graph_context_prompt_text(cleaned)


def test_graph_context_prompt_explains_empty_context() -> None:
    assert "retrieved text evidence only" in graph_context_prompt_text([])
