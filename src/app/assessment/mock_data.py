from __future__ import annotations

import json

from app.assessment.schemas import (
    ComplianceStatus,
    ControlReference,
    EvidenceItem,
    FoundationAssessmentPacket,
    MaturityLevel,
    QuestionnaireResult,
    TierAttribute,
    TierProfile,
    VendorProfile,
)


class MockFoundationChatModel:
    def chat(self, messages: list[dict[str, str]]) -> str:
        return json.dumps(
            {
                "management_summary": (
                    "Acme SaaS is a Tier 2 vendor with access to sensitive business data. "
                    "The assessment shows one implemented access-control strength and two "
                    "areas requiring analyst follow-up: endpoint malware protection and "
                    "untested recovery capability."
                ),
                "introduction": (
                    "This draft summarizes Acme SaaS based on simulated vendor profile, "
                    "tier attributes, questionnaire answers, comments, and evidence descriptions."
                ),
                "objective": (
                    "The objective is to provide a business-readable view of the vendor's "
                    "security posture before analyst approval and snapshot creation."
                ),
                "key_findings": [
                    "Privileged access review evidence was provided and assessed as implemented.",
                    "No anti-malware solution is currently deployed on endpoints.",
                    "The disaster recovery plan exists, but the vendor has not tested it.",
                ],
                "strengths": [
                    "Access rights review is implemented and supported by PDF evidence."
                ],
                "weaknesses": [
                    "Endpoint protection is not implemented, increasing malware exposure.",
                    "Recovery readiness is only developing because the DR plan is untested.",
                ],
                "risk_exposure": (
                    "Risk exposure remains elevated for malware disruption and recovery "
                    "uncertainty until the endpoint protection and DR testing gaps are remediated."
                ),
                "conclusion": (
                    "The draft should be reviewed by the analyst and business owner before "
                    "being saved as an immutable assessment snapshot."
                ),
                "missing_information": [
                    "Missing evidence for Q2 anti-malware deployment.",
                    "Missing evidence for Q3 disaster recovery test results.",
                ],
                "source_question_ids": ["Q1", "Q2", "Q3"],
                "from_assessment_data": (
                    "Generated from the simulated assessment packet and deterministic finding "
                    "classification."
                ),
                "general_model_reasoning": "",
            }
        )


def sample_foundation_packet() -> FoundationAssessmentPacket:
    return FoundationAssessmentPacket(
        assessment_id="A-100",
        vendor=VendorProfile(
            vendor_id="V-1",
            name="Acme SaaS",
            vendor_type="SaaS provider",
            business_relationship="customer data processing",
            country="LU",
            services=["case management platform", "customer support workflow"],
        ),
        tier=TierProfile(
            level=2,
            definition="Important vendor with access to sensitive business data.",
            attributes=[
                TierAttribute(
                    name="company_size",
                    value="medium",
                    definition="Medium vendor with a meaningful operational footprint.",
                ),
                TierAttribute(
                    name="sensitive_data_access",
                    value=True,
                    definition="Vendor processes sensitive business and customer data.",
                ),
                TierAttribute(
                    name="privileged_access",
                    value=False,
                    definition="Vendor does not receive privileged access to internal systems.",
                ),
                TierAttribute(
                    name="geolocation",
                    value="EU",
                    definition="Vendor operates primarily in the European Union.",
                ),
            ],
        ),
        questionnaire_results=[
            QuestionnaireResult(
                question_id="Q1",
                question_text="Do you periodically review privileged and administrative access?",
                control=ControlReference(
                    framework="ISO 27002",
                    control_id="5.18",
                    title="Access rights",
                    control_type="preventative",
                ),
                response="Yes",
                vendor_comment="Access reviews are performed quarterly.",
                analyst_comment="Access review evidence was provided and looks complete.",
                compliance=ComplianceStatus.FULL,
                maturity=MaturityLevel.IMPLEMENTED,
                evidence=[
                    EvidenceItem(
                        evidence_id="E1",
                        description="Quarterly access review PDF signed by IT owner.",
                        file_type="pdf",
                    )
                ],
            ),
            QuestionnaireResult(
                question_id="Q2",
                question_text="Do you deploy and monitor anti-malware protection on endpoints?",
                control=ControlReference(
                    framework="NIST CSF",
                    control_id="PR.PS-01",
                    title="Endpoint protection",
                    control_type="preventative",
                ),
                response="No",
                vendor_comment="Endpoint protection is planned for next quarter.",
                analyst_comment="No anti-malware solution is currently deployed.",
                compliance=ComplianceStatus.NO,
                maturity=MaturityLevel.BASIC,
            ),
            QuestionnaireResult(
                question_id="Q3",
                question_text="Have you tested disaster recovery for the service in scope?",
                control=ControlReference(
                    framework="ISO 27002",
                    control_id="5.30",
                    title="ICT readiness for business continuity",
                    control_type="recovery",
                ),
                response="Partially",
                vendor_comment="A disaster recovery plan exists, but the last test is pending.",
                analyst_comment="No test report was provided.",
                compliance=ComplianceStatus.PARTIAL,
                maturity=MaturityLevel.DEVELOPING,
            ),
        ],
        metadata={"source": "simulated_postgresql"},
    )

