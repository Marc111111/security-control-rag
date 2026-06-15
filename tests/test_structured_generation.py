from app.generation.structured import parse_structured_answer
from app.schemas import DocumentChunk, RetrievedEvidence


def test_parse_structured_answer_normalizes_common_local_model_json() -> None:
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(
                id="chunk-1",
                text="CIS Controls 10.1: Deploy and maintain anti-malware protection.",
                metadata={"source_path": "standards/cis.pdf"},
            ),
            score=0.9,
            source="standards/cis.pdf",
            retrieval_method="keyword",
        )
    ]
    raw = """
    {
      "executive_summary": "Anti-malware is missing.",
      "threats": [{"name": "Malware"}],
      "vulnerabilities": [{"name": "Unprotected endpoints", "id": "v1"}],
      "risks": [{"name": "Business disruption"}],
      "recommended_controls": [{"name": "CIS Controls 10.1"}],
      "risk_control_matrix": [
        {
          "gap": "No anti-malware",
          "threat": {"name": "Malware"},
          "vulnerability": {"name": "Unprotected endpoints"},
          "risk": {"name": "Business disruption"},
          "controls": [{"name": "CIS Controls 10.1"}],
          "evidence": ["S1"]
        }
      ],
      "source_citations": ["S1"]
    }
    """

    answer = parse_structured_answer(raw, evidence)

    assert answer.threats == ["Malware"]
    assert answer.vulnerabilities == ["Unprotected endpoints"]
    assert answer.recommended_controls == ["CIS Controls 10.1"]
    assert answer.risk_control_matrix[0].threat == "Malware"
    assert answer.source_citations[0]["chunk_id"] == "chunk-1"


def test_parse_structured_answer_extracts_control_labels_from_evidence() -> None:
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(
                id="chunk-1",
                text=(
                    "Safeguard 10.1: Deploy and Maintain Anti-Malware Software "
                    "Asset Type: Devices. SCF Control: Malicious Code Protection "
                    "(Anti-Malware) SCF #: END-04 Secure Controls Framework."
                ),
                metadata={"source_path": "standards/cis.pdf"},
            ),
            score=0.9,
            source="standards/cis.pdf",
            retrieval_method="keyword",
        )
    ]
    raw = """
    {
      "executive_summary": "Endpoint protection is missing.",
      "recommended_controls": ["CIS Controls 1", "CIS Controls 4.0"],
      "risk_control_matrix": [{"gap": "No anti-malware"}],
      "source_citations": ["S1"]
    }
    """

    answer = parse_structured_answer(raw, evidence)

    assert "CIS Safeguard 10.1 - Deploy and Maintain Anti-Malware Software" in (
        answer.recommended_controls
    )
    assert "SCF END-04 - Malicious Code Protection (Anti-Malware)" in (
        answer.recommended_controls
    )
    assert "CIS Controls 1" not in answer.recommended_controls


def test_parse_structured_answer_caps_enriched_controls() -> None:
    text = " ".join(
        [
            f"Safeguard 10.{index}: Control Title {index} Asset Type: Devices."
            for index in range(1, 8)
        ]
    )
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(
                id="chunk-1",
                text=text,
                metadata={"source_path": "standards/cis.pdf"},
            ),
            score=0.9,
            source="standards/cis.pdf",
            retrieval_method="keyword",
        )
    ]
    raw = """
    {
      "executive_summary": "Endpoint protection is missing.",
      "threats": ["Malware"],
      "vulnerabilities": ["Unprotected endpoints"],
      "risks": ["Business disruption"],
      "recommended_controls": [],
      "risk_control_matrix": [
        {
          "gap": "No anti-malware",
          "threat": "Malware",
          "vulnerability": "Unprotected endpoints",
          "risk": "Business disruption",
          "controls": [],
          "evidence": ["S1"]
        }
      ],
      "source_citations": ["S1"]
    }
    """

    answer = parse_structured_answer(raw, evidence)

    assert len(answer.recommended_controls) == 5
    assert len(answer.risk_control_matrix[0].controls) == 5
