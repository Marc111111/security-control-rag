from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ComplianceStatus(StrEnum):
    FULL = "full"
    PARTIAL = "partial"
    NO = "no"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class MaturityLevel(StrEnum):
    BASIC = "basic"
    DEVELOPING = "developing"
    IMPLEMENTED = "implemented"
    MANAGED = "managed"
    OPTIMIZED = "optimized"
    UNKNOWN = "unknown"


class VendorProfile(BaseModel):
    vendor_id: str
    name: str
    vendor_type: str
    business_relationship: str = ""
    country: str | None = None
    services: list[str] = Field(default_factory=list)


class TierAttribute(BaseModel):
    name: str
    value: str | int | float | bool
    definition: str


class TierProfile(BaseModel):
    level: int = Field(..., ge=1, le=4)
    definition: str
    attributes: list[TierAttribute] = Field(default_factory=list)


class ControlReference(BaseModel):
    framework: str
    control_id: str
    title: str = ""
    control_type: Literal[
        "preventative",
        "detective",
        "corrective",
        "recovery",
        "response",
        "unknown",
    ] = "unknown"


class EvidenceItem(BaseModel):
    evidence_id: str
    description: str
    file_type: Literal["pdf", "jpg", "png"]


class QuestionnaireResult(BaseModel):
    question_id: str
    question_text: str
    control: ControlReference
    response: str
    vendor_comment: str = ""
    analyst_comment: str = ""
    compliance: ComplianceStatus = ComplianceStatus.UNKNOWN
    maturity: MaturityLevel = MaturityLevel.UNKNOWN
    evidence: list[EvidenceItem] = Field(default_factory=list)


class FoundationAssessmentPacket(BaseModel):
    assessment_id: str
    vendor: VendorProfile
    tier: TierProfile
    questionnaire_results: list[QuestionnaireResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssessmentFinding(BaseModel):
    question_id: str
    control: ControlReference
    compliance: ComplianceStatus
    maturity: MaturityLevel
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)


class SanitizedQuestionnaireResult(QuestionnaireResult):
    sanitized_vendor_comment: str = ""
    sanitized_analyst_comment: str = ""


class SanitizedAssessmentPacket(BaseModel):
    assessment_id: str
    vendor: VendorProfile
    tier: TierProfile
    questionnaire_results: list[SanitizedQuestionnaireResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FoundationSummaryDraft(BaseModel):
    management_summary: str = ""
    introduction: str = ""
    objective: str = ""
    key_findings: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    risk_exposure: str = ""
    conclusion: str = ""
    missing_information: list[str] = Field(default_factory=list)
    source_question_ids: list[str] = Field(default_factory=list)
    from_assessment_data: str = ""
    general_model_reasoning: str = ""


class FoundationSummaryResponse(BaseModel):
    assessment_id: str
    vendor_id: str
    draft: FoundationSummaryDraft
    findings: dict[str, list[AssessmentFinding]]
    postgres_payload: dict[str, Any]
    debug: dict[str, Any] = Field(default_factory=dict)

