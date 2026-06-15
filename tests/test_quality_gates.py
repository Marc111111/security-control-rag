from app.quality_gates import (
    prune_unsupported_risk_answer,
    validate_final_paragraphs,
    validate_gap_storyline,
    validate_prompt_quality,
    validate_risk_answer,
    validate_risk_assessment_chains,
    validate_workflow_step_contract,
    validate_workflow_step_payload,
)
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
                text=(
                    "Recovery capabilities should be tested using available backups to reduce "
                    "system outage and extended downtime."
                ),
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


def test_risk_answer_gate_blocks_unsupported_threat_labels() -> None:
    answer = StructuredRiskAnswer(
        executive_summary="Summary",
        threats=["Living-off-the-land attacks"],
        vulnerabilities=["Unprotected endpoints"],
        risks=["Malware execution"],
        recommended_controls=["CIS Safeguard 10.1"],
        risk_control_matrix=[
            MatrixRow(
                gap="No anti-malware",
                threat="Living-off-the-land attacks",
                vulnerability="Unprotected endpoints",
                risk="Malware execution",
                controls=["CIS Safeguard 10.1"],
                evidence=["S1"],
            )
        ],
    )
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(
                id="chunk-1",
                text="CIS Safeguard 10.1 deploy anti-malware to prevent malicious code.",
            ),
            score=0.9,
            source="standards/cis.pdf",
            retrieval_method="keyword",
        )
    ]

    gate = validate_risk_answer(
        answer=answer,
        evidence=evidence,
        raw_model_output=answer.model_dump_json(),
        question="No endpoint anti-malware is deployed.",
    )

    assert gate.passed is False
    assert any("does not reuse meaningful terminology" in issue.message for issue in gate.issues)


def test_prune_unsupported_risk_answer_removes_extra_unsupported_labels() -> None:
    answer = StructuredRiskAnswer(
        executive_summary="Summary",
        threats=["Malware", "Unrelated phishing"],
        vulnerabilities=["No anti-malware deployment", "Non-compliant configuration"],
        risks=["Malware execution", "Data breach"],
        recommended_controls=["CIS Safeguard 10.1"],
        risk_control_matrix=[
            MatrixRow(
                gap="No anti-malware",
                threat="Malware",
                vulnerability="No anti-malware deployment",
                risk="Malware execution",
                controls=["CIS Safeguard 10.1"],
                evidence=["S1"],
            ),
            MatrixRow(
                gap="Other",
                threat="Unrelated phishing",
                vulnerability="Non-compliant configuration",
                risk="Data breach",
                controls=["CIS Safeguard 10.1"],
                evidence=["S1"],
            ),
        ],
    )
    evidence = [
        RetrievedEvidence(
            chunk=DocumentChunk(
                id="chunk-1",
                text="CIS Safeguard 10.1 deploy anti-malware to prevent malware execution.",
            ),
            score=0.9,
            source="standards/cis.pdf",
            retrieval_method="keyword",
        )
    ]

    pruned = prune_unsupported_risk_answer(
        answer=answer,
        evidence=evidence,
        question="No anti-malware is deployed.",
    )

    assert pruned.threats == ["Malware"]
    assert pruned.vulnerabilities == ["No anti-malware deployment"]
    assert pruned.risks == ["Malware execution"]
    assert len(pruned.risk_control_matrix) == 1


def test_workflow_payload_gate_blocks_oversized_step_output() -> None:
    gate = validate_workflow_step_payload(
        step_name="Bad step",
        input_payload={"ok": True},
        output_payload={"huge": "x" * 70_000},
    )

    assert gate.passed is False
    assert gate.gate == "workflow_step_payload"
    assert gate.issues[0].field == "output"


def test_final_paragraph_gate_blocks_unsupported_acceptable_risk_claim() -> None:
    gate = validate_final_paragraphs(
        paragraphs={
            "management_summary": "Acme SaaS is a Tier 2 vendor with anti-malware gaps.",
            "introduction": "Acme SaaS is reviewed for vendor risk.",
            "objective": "The objective is to review endpoint and recovery gaps.",
            "risk_exposure": "The gaps exceed acceptable risk thresholds.",
            "conclusion": "Acme SaaS should remediate anti-malware and recovery gaps.",
        },
        raw_model_output="{}",
        vendor_name="Acme SaaS",
        tier_level=2,
        weakness_summaries=["anti-malware gap"],
        validated_facts={"validated_risk_facts": [{"risks": ["anti-malware gap"]}]},
    )

    assert gate.passed is False
    assert any("acceptance threshold" in issue.message for issue in gate.issues)


def test_final_paragraph_gate_requires_toolchain_added_value() -> None:
    gate = validate_final_paragraphs(
        paragraphs={
            "management_summary": "Acme SaaS is a Tier 2 vendor with endpoint gaps.",
            "introduction": "Acme SaaS is reviewed for vendor risk.",
            "objective": "The objective is to review questionnaire gaps.",
            "risk_exposure": "The vendor has no anti-malware.",
            "conclusion": "The vendor should fix the gap.",
        },
        raw_model_output="{}",
        vendor_name="Acme SaaS",
        tier_level=2,
        weakness_summaries=["anti-malware gap"],
        validated_facts={
            "validated_risk_facts": [{"risks": ["anti-malware gap"]}],
            "toolchain_delta": {
                "added_by_rag": ["CIS Safeguard 10.1 - anti-malware"],
                "added_by_resilience_analysis": [],
            },
        },
    )

    assert gate.passed is False
    assert any("standards/RAG value" in issue.message for issue in gate.issues)


def test_final_paragraph_gate_blocks_over_word_cap() -> None:
    long_paragraph = (
        "Acme SaaS is a Tier 2 vendor where CIS endpoint controls and recovery evidence "
        + " ".join(f"remain relevant item {index}" for index in range(30))
    )

    gate = validate_final_paragraphs(
        paragraphs={
            "management_summary": long_paragraph,
            "introduction": "Acme SaaS is reviewed as a Tier 2 vendor.",
            "objective": "The objective is to review CIS controls and recovery risk.",
            "risk_exposure": "CIS endpoint controls and recovery risk drive exposure.",
            "conclusion": "CIS controls and recovery evidence remain required.",
        },
        raw_model_output="{}",
        vendor_name="Acme SaaS",
        tier_level=2,
        weakness_summaries=["anti-malware gap"],
        validated_facts={
            "validated_risk_facts": [{"risks": ["anti-malware gap"]}],
            "toolchain_delta": {
                "added_by_rag": ["CIS Safeguard 10.1 - anti-malware"],
                "added_by_resilience_analysis": ["recovery evidence"],
            },
        },
    )

    assert gate.passed is False
    assert any("maximum 120 words" in issue.message for issue in gate.issues)


def test_final_paragraph_gate_requires_named_added_control_in_risk_or_conclusion() -> None:
    gate = validate_final_paragraphs(
        paragraphs={
            "management_summary": "Acme SaaS is a Tier 2 vendor with endpoint gaps.",
            "introduction": "Acme SaaS is reviewed for vendor risk.",
            "objective": "The objective is to review endpoint and recovery risk.",
            "risk_exposure": "The toolchain mapped the issue to 7 standards-backed controls.",
            "conclusion": "Controls should be implemented and evidence should be reviewed.",
        },
        raw_model_output="{}",
        vendor_name="Acme SaaS",
        tier_level=2,
        weakness_summaries=["anti-malware gap"],
        validated_facts={
            "validated_risk_facts": [{"risks": ["malware infection"]}],
            "toolchain_delta": {
                "added_by_rag": ["CIS Safeguard 10.1 - anti-malware"],
                "added_by_resilience_analysis": ["recovery evidence"],
            },
        },
    )

    assert gate.passed is False
    assert any("standards/control reference" in issue.message for issue in gate.issues)


def test_final_paragraph_gate_blocks_unsupported_urgency_language() -> None:
    gate = validate_final_paragraphs(
        paragraphs={
            "management_summary": "Acme SaaS is a Tier 2 vendor with endpoint gaps.",
            "introduction": "Acme SaaS is reviewed for vendor risk.",
            "objective": "The objective is to review CIS controls and recovery risk.",
            "risk_exposure": "CIS endpoint gaps create malware exposure.",
            "conclusion": "Acme SaaS requires immediate action on CIS controls.",
        },
        raw_model_output="{}",
        vendor_name="Acme SaaS",
        tier_level=2,
        weakness_summaries=["anti-malware gap"],
        validated_facts={
            "validated_risk_facts": [{"risks": ["malware infection"]}],
            "toolchain_delta": {
                "added_by_rag": ["CIS Safeguard 10.1 - anti-malware"],
                "added_by_resilience_analysis": ["recovery evidence"],
            },
        },
    )

    assert gate.passed is False
    assert any("urgency or severity" in issue.message for issue in gate.issues)


def test_risk_assessment_chain_gate_requires_delta_and_business_fields() -> None:
    gate = validate_risk_assessment_chains(
        {
            "risk_assessment_chains": [
                {
                    "known_from_assessment": ["No anti-malware"],
                    "standards_requirements_added": [],
                }
            ],
            "toolchain_delta": {"added_by_rag": []},
        }
    )

    assert gate.passed is False
    assert any("toolchain_delta" in issue.field for issue in gate.issues)


def test_gap_storyline_gate_requires_business_chain_coverage() -> None:
    chain = {
        "question_id": "Q2",
        "confirmed_gaps": ["No anti-malware deployment"],
        "threat_scenarios": ["Malware"],
        "vulnerabilities": ["Unprotected endpoints"],
        "inherent_risk": {"risk_statement": "Malware execution"},
        "standards_requirements_added": ["CIS Safeguard 10.1"],
        "resilience_effects": ["Detection supports response only when ownership is evidenced."],
        "residual_concern": {
            "remaining_issue": "Operating effectiveness evidence is still missing."
        },
    }
    storyline = {
        "question_id": "Q2",
        "gap_story": "No anti-malware deployment is the validated gap.",
        "business_meaning": "The gap leaves unprotected endpoints exposed to malware.",
        "risk_logic": "The inherent risk is malware execution for the vendor.",
        "control_logic": "The chain adds CIS Safeguard 10.1 as the named control.",
        "resilience_logic": "Detection supports response only when ownership is evidenced.",
        "residual_conclusion": "Operating effectiveness evidence is still missing.",
    }

    gate = validate_gap_storyline(
        storyline=storyline,
        risk_chain=chain,
        raw_model_output=str(storyline),
    )

    assert gate.passed is True


def test_gap_storyline_gate_blocks_generic_storyline() -> None:
    chain = {
        "question_id": "Q2",
        "confirmed_gaps": ["No anti-malware deployment"],
        "threat_scenarios": ["Malware"],
        "vulnerabilities": ["Unprotected endpoints"],
        "inherent_risk": {"risk_statement": "Malware execution"},
        "standards_requirements_added": ["CIS Safeguard 10.1"],
        "resilience_effects": ["Detection supports response only when ownership is evidenced."],
        "residual_concern": {
            "remaining_issue": "Operating effectiveness evidence is still missing."
        },
    }
    storyline = {
        "question_id": "Q2",
        "gap_story": "The vendor has some security gaps.",
        "business_meaning": "This may create business issues.",
        "risk_logic": "The organization should improve security.",
        "control_logic": "Controls should be implemented.",
        "resilience_logic": "Resilience should be improved.",
        "residual_conclusion": "More work is needed.",
    }

    gate = validate_gap_storyline(
        storyline=storyline,
        risk_chain=chain,
        raw_model_output=str(storyline),
    )

    assert gate.passed is False
    assert any("validated control" in issue.message for issue in gate.issues)


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


def test_prompt_quality_gate_blocks_large_single_risk_prompt() -> None:
    gate = validate_prompt_quality(
        gate="risk_answer_prompt_quality",
        messages=[
            {"role": "system", "content": "Role: analyst"},
            {"role": "user", "content": "\n".join([f"[S{i}] " + "x" * 900 for i in range(1, 8)])},
        ],
        task_name="risk answer",
        max_total_chars=8_500,
        max_user_chars=6_900,
        max_source_markers=5,
    )

    assert gate.passed is False
    assert any("too many cited source excerpts" in issue.message for issue in gate.issues)


def test_workflow_step_contract_blocks_raw_planner_json_handoff() -> None:
    gate = validate_workflow_step_contract(
        step_name="Plan searches for Q2 / PR.PS-01",
        input_payload={"selected_risk_question": {"question_id": "Q2"}},
        output_payload={"search_plan": {"sub_questions": []}, "top_k": 8},
    )

    assert gate.passed is False
    assert any("internal planner JSON" in issue.message for issue in gate.issues)


def test_workflow_step_contract_requires_prompt_evidence() -> None:
    gate = validate_workflow_step_contract(
        step_name="Search standards evidence for Q2 / PR.PS-01",
        input_payload={"search_focus": []},
        output_payload={
            "retrieved_chunks": [],
            "prompt_evidence": {"retrieved_source_count": 0, "prompt_source_count": 0},
        },
    )

    assert gate.passed is False
    assert any("select any compact evidence" in issue.message for issue in gate.issues)
