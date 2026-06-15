from app.workflows.step_audit import audit_workflow_run


def test_workflow_step_audit_flags_uncontrolled_handoff() -> None:
    run = {
        "steps": [
            {
                "name": "Read assessment input",
                "input": {"payload": "db row"},
                "process": "read",
                "output": {"normalized_packet": {"assessment_id": "A1"}},
            },
            {
                "name": "Clean and sort questionnaire answers",
                "input": {"different": "object"},
                "process": "clean",
                "output": {"findings": {"weaknesses": []}},
            },
        ]
    }

    audit = audit_workflow_run(run)

    assert audit["passed"] is False
    assert any("not the previous step output" in issue["message"] for issue in audit["issues"])


def test_workflow_step_audit_accepts_final_report_controlled_derivation() -> None:
    previous_output = {
        "raw_model_output_preview": "{}",
        "parsed_paragraphs": {"management_summary": "Acme SaaS Tier 2 has a gap."},
        "quality_gate": {"passed": True},
    }
    run = {
        "steps": [
            {
                "name": "Ask model to draft report paragraphs",
                "input": {
                    "validated_fact_packet_from_previous_step": {"validated_risk_facts": []},
                    "model_prompt": {"total_characters_sent_to_model": 1000},
                },
                "process": "draft",
                "output": previous_output,
            },
            {
                "name": "Prepare final result for the application",
                "input": {
                    "paragraphs_from_previous_step": previous_output["parsed_paragraphs"],
                },
                "process": "package",
                "output": {"assessment_id": "A1"},
            },
        ]
    }

    audit = audit_workflow_run(run)

    assert audit["passed"] is True
