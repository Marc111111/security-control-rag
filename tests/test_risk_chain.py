from app.assessment.findings import classify_findings, sanitize_packet
from app.assessment.mock_data import sample_foundation_packet
from app.workflows.risk_chain import build_risk_assessment_chains


def test_build_risk_assessment_chains_records_added_value_delta() -> None:
    packet = sanitize_packet(sample_foundation_packet())
    findings = classify_findings(packet)
    risk_evidence = [
        {
            "answer": {
                "threats": ["Malware"],
                "vulnerabilities": ["No anti-malware deployment"],
                "risks": ["Malware execution"],
                "recommended_controls": [
                    "CIS Safeguard 10.1 - Deploy and Maintain Anti-Malware Software",
                    "SCF END-04.7 - Always On Protection",
                ],
                "risk_control_matrix": [
                    {
                        "gap": "No anti-malware deployment",
                        "threat": "Malware",
                        "vulnerability": "No anti-malware deployment",
                        "risk": "Malware execution",
                        "likelihood": "high",
                        "impact": "high",
                        "controls": ["CIS Safeguard 10.1"],
                        "evidence": ["S1"],
                    }
                ],
                "missing_information": ["Confirm monitoring ownership."],
            },
            "sources": [
                {
                    "id": "S1",
                    "source": "standards/cis.pdf",
                    "metadata": {"control_id": "10.1", "framework": "CIS Controls"},
                }
            ],
        }
    ]

    result = build_risk_assessment_chains(
        packet=packet,
        weaknesses=findings["weaknesses"][:1],
        risk_evidence=risk_evidence,
    )

    chain = result["risk_assessment_chains"][0]
    assert chain["known_from_assessment"]
    assert chain["standards_requirements_added"]
    assert chain["inherent_risk"]["risk_statement"] == "Malware execution"
    assert chain["recommended_controls_by_function"]
    assert chain["residual_concern"]["remaining_issue"]
    assert result["toolchain_delta"]["added_by_rag"]
    assert "Confirm monitoring ownership." in result["toolchain_delta"]["remaining_uncertainty"]
