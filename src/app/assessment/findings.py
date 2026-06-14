from __future__ import annotations

from app.assessment.sanitizer import HumanTextSanitizer
from app.assessment.schemas import (
    AssessmentFinding,
    ComplianceStatus,
    FoundationAssessmentPacket,
    MaturityLevel,
    SanitizedAssessmentPacket,
    SanitizedQuestionnaireResult,
)


def sanitize_packet(
    packet: FoundationAssessmentPacket,
    sanitizer: HumanTextSanitizer | None = None,
) -> SanitizedAssessmentPacket:
    sanitizer = sanitizer or HumanTextSanitizer()
    results: list[SanitizedQuestionnaireResult] = []
    for result in packet.questionnaire_results:
        sanitized_vendor_comment = sanitizer.sanitize(result.vendor_comment)
        sanitized_analyst_comment = sanitizer.sanitize(result.analyst_comment)
        data = result.model_dump()
        data["vendor_comment"] = sanitized_vendor_comment
        data["analyst_comment"] = sanitized_analyst_comment
        results.append(
            SanitizedQuestionnaireResult(
                **data,
                sanitized_vendor_comment=sanitized_vendor_comment,
                sanitized_analyst_comment=sanitized_analyst_comment,
            )
        )
    return SanitizedAssessmentPacket(
        assessment_id=packet.assessment_id,
        vendor=packet.vendor,
        tier=packet.tier,
        questionnaire_results=results,
        metadata=packet.metadata,
    )


def classify_findings(
    packet: SanitizedAssessmentPacket,
) -> dict[str, list[AssessmentFinding]]:
    strengths: list[AssessmentFinding] = []
    weaknesses: list[AssessmentFinding] = []
    unknowns: list[AssessmentFinding] = []

    for result in packet.questionnaire_results:
        finding = _finding_for_result(result)
        if result.compliance == ComplianceStatus.FULL:
            strengths.append(finding)
        elif result.compliance in {ComplianceStatus.PARTIAL, ComplianceStatus.NO}:
            weaknesses.append(finding)
        elif result.compliance != ComplianceStatus.NOT_APPLICABLE:
            unknowns.append(finding)

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "unknowns": unknowns,
    }


def _finding_for_result(result: SanitizedQuestionnaireResult) -> AssessmentFinding:
    evidence_ids = [item.evidence_id for item in result.evidence]
    maturity = result.maturity if result.maturity != MaturityLevel.UNKNOWN else "unknown maturity"
    summary = (
        f"{result.control.framework} {result.control.control_id} "
        f"({result.control.title or 'untitled control'}) is assessed as "
        f"{result.compliance.value} compliance with {maturity} maturity."
    )
    if result.sanitized_analyst_comment:
        summary = f"{summary} Analyst note: {result.sanitized_analyst_comment}"
    elif result.sanitized_vendor_comment:
        summary = f"{summary} Vendor note: {result.sanitized_vendor_comment}"
    return AssessmentFinding(
        question_id=result.question_id,
        control=result.control,
        compliance=result.compliance,
        maturity=result.maturity,
        summary=summary,
        evidence_ids=evidence_ids,
    )
