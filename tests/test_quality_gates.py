from app.quality_gates import validate_risk_answer, validate_workflow_step_payload
from app.schemas import DocumentChunk, MatrixRow, RetrievedEvidence, StructuredRiskAnswer


def test_risk_answer_warnings_do_not_block_otherwise_complete_answer() -> None:
    answer = StructuredRiskAnswer(
        executive_summary="Disaster recovery testing gap increases resilience risk.",
        threats=["System outage"],
        vulnerabilities=["Untested recovery procedures"],
        risks=["Extended downtime"],
        recommended_controls=["Scenario testing (S1)"],
        risk_control_matrix=[
            MatrixRow(
                gap="Disaster recovery plan not tested",
                threat="System outage",
                vulnerability="Untested recovery procedures",
                risk="Extended downtime",
                likelihood="medium",
                impact="high",
                controls=["Scenario testing"],
                evidence=["S1"],
            )
        ],
    )
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(
                id="chunk-1",
                text="Recovery capabilities should be tested using available backups.",
            ),
            score=0.9,
            source="standards/scf.xlsx",
            retrieval_method="keyword",
        )
    ]

    gate = validate_risk_answer(
        answer=answer,
        evidence=evidence,
        raw_model_output=answer.model_dump_json(),
        question="Untested disaster recovery",
    )

    assert gate.passed is True
    assert gate.issues
    assert {issue.severity for issue in gate.issues} == {"warning"}


def test_risk_answer_gate_blocks_verbose_structured_fields() -> None:
    verbose_threat = " ".join(["malware"] * 35)
    answer = StructuredRiskAnswer(
        executive_summary="Summary",
        threats=[verbose_threat],
        vulnerabilities=["Untested recovery"],
        risks=["Extended downtime"],
        recommended_controls=["CIS Safeguard 10.1"],
        risk_control_matrix=[
            MatrixRow(
                gap="Missing control",
                threat="Malware",
                vulnerability="Untested recovery",
                risk="Extended downtime",
                controls=["CIS Safeguard 10.1"],
                evidence=["S1"],
            )
        ],
    )
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(id="chunk-1", text="CIS Safeguard 10.1 malware"),
            score=0.9,
            source="standards/cis.pdf",
            retrieval_method="keyword",
        )
    ]

    gate = validate_risk_answer(
        answer=answer,
        evidence=evidence,
        raw_model_output=answer.model_dump_json(),
        question="Malware gap",
    )

    assert gate.passed is False
    assert any("too verbose" in issue.message for issue in gate.issues)


def test_workflow_payload_gate_blocks_oversized_step_output() -> None:
    gate = validate_workflow_step_payload(
        step_name="Bad step",
        input_payload={"ok": True},
        output_payload={"huge": "x" * 70_000},
    )

    assert gate.passed is False
    assert gate.gate == "workflow_step_payload"
    assert gate.issues[0].field == "output"


def test_workflow_payload_gate_blocks_debug_leakage() -> None:
    gate = validate_workflow_step_payload(
        step_name="Bad step",
        input_payload={"ok": True},
        output_payload={"answer": "ok", "debug": {"prompt_messages": ["secret"]}},
    )

    assert gate.passed is False
    assert "debug" in gate.issues[0].message


def test_workflow_payload_gate_allows_compact_prompt_preview() -> None:
    gate = validate_workflow_step_payload(
        step_name="Prompt step",
        input_payload={"sources": [{"id": "S1"}]},
        output_payload={
            "model_prompt": {
                "total_characters_sent_to_model": 8_000,
                "messages": [{"role": "user", "preview": "short"}],
            }
        },
    )

    assert gate.passed is True
