from __future__ import annotations

import json
import re
from typing import Any

from app.assessment.findings import classify_findings, sanitize_packet
from app.assessment.prompts import build_foundation_summary_prompt
from app.assessment.schemas import (
    AssessmentFinding,
    ComplianceStatus,
    FoundationAssessmentPacket,
    FoundationSummaryDraft,
    FoundationSummaryResponse,
    SanitizedAssessmentPacket,
)
from app.generation.clients import ChatModel


class FoundationAssessmentWorkflow:
    def __init__(self, chat_model: ChatModel) -> None:
        self.chat_model = chat_model

    def summarize(
        self,
        packet: FoundationAssessmentPacket,
        *,
        debug: bool = False,
    ) -> FoundationSummaryResponse:
        sanitized = sanitize_packet(packet)
        findings = classify_findings(sanitized)
        messages = build_foundation_summary_prompt(sanitized, _dump_findings(findings))
        raw = self.chat_model.chat(messages)
        draft = parse_summary_draft(raw, sanitized, findings)
        payload = postgres_payload(packet, draft)
        response_debug = (
            {
                "sanitized_packet": sanitized.model_dump(),
                "prompt_messages": messages,
                "raw_model_response": raw,
            }
            if debug
            else {}
        )
        return FoundationSummaryResponse(
            assessment_id=packet.assessment_id,
            vendor_id=packet.vendor.vendor_id,
            draft=draft,
            findings=findings,
            postgres_payload=payload,
            debug=response_debug,
        )


def parse_summary_draft(
    raw: str,
    packet: SanitizedAssessmentPacket,
    findings: dict[str, list[AssessmentFinding]],
) -> FoundationSummaryDraft:
    try:
        return FoundationSummaryDraft.model_validate(json.loads(_extract_json(raw)))
    except Exception:
        return deterministic_summary(packet, findings)


def deterministic_summary(
    packet: SanitizedAssessmentPacket,
    findings: dict[str, list[AssessmentFinding]],
) -> FoundationSummaryDraft:
    strengths = [finding.summary for finding in findings["strengths"]]
    weaknesses = [finding.summary for finding in findings["weaknesses"]]
    unknowns = [finding.summary for finding in findings["unknowns"]]
    missing_evidence = [
        result.question_id
        for result in packet.questionnaire_results
        if result.compliance in {ComplianceStatus.PARTIAL, ComplianceStatus.NO}
        and not result.evidence
    ]
    source_ids = [result.question_id for result in packet.questionnaire_results]
    return FoundationSummaryDraft(
        management_summary=(
            f"{packet.vendor.name} is assessed as a Tier {packet.tier.level} vendor. "
            f"The assessment identified {len(strengths)} strengths and "
            f"{len(weaknesses)} weaknesses across the questionnaire results."
        ),
        introduction=(
            f"This assessment summarizes the security posture of {packet.vendor.name}, "
            f"a {packet.vendor.vendor_type} supporting {packet.vendor.business_relationship}."
        ),
        objective=(
            "The objective is to help the business owner understand the vendor's current "
            "risk posture based on tier attributes, questionnaire responses, comments, "
            "and evidence descriptions."
        ),
        key_findings=[*strengths[:3], *weaknesses[:5], *unknowns[:3]],
        strengths=strengths,
        weaknesses=weaknesses,
        risk_exposure=(
            f"Tier {packet.tier.level}: {packet.tier.definition} "
            f"Open weaknesses should be reviewed before relying on the vendor's controls."
        ),
        conclusion=(
            "The assessment draft is ready for analyst review. Partial, non-compliant, "
            "and unknown responses should be confirmed before creating an immutable snapshot."
        ),
        missing_information=[
            f"Missing evidence for question {question_id}" for question_id in missing_evidence
        ],
        source_question_ids=source_ids,
        from_assessment_data=(
            "The draft was generated from tier attributes, questionnaire answers, comments, "
            "compliance status, maturity, and evidence descriptions."
        ),
        general_model_reasoning=(
            "No external standards or invented citations were used in the deterministic fallback."
        ),
    )


def postgres_payload(
    packet: FoundationAssessmentPacket,
    draft: FoundationSummaryDraft,
) -> dict[str, Any]:
    return {
        "assessment_id": packet.assessment_id,
        "vendor_id": packet.vendor.vendor_id,
        "draft_sections": draft.model_dump(),
        "snapshot_ready": False,
        "source_question_ids": draft.source_question_ids,
    }


def _dump_findings(
    findings: dict[str, list[AssessmentFinding]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [finding.model_dump() for finding in value]
        for key, value in findings.items()
    }


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text

