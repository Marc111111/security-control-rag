from __future__ import annotations

import json

from app.assessment.schemas import SanitizedAssessmentPacket

SYSTEM_PROMPT = """You are a third-party risk management assessment assistant.
Write business-friendly assessment sections from the provided structured assessment packet.
Do not invent evidence, certifications, controls, questionnaire answers, or citations.
Return only valid JSON with the requested fields."""


def build_foundation_summary_prompt(
    packet: SanitizedAssessmentPacket,
    findings: dict[str, object],
) -> list[dict[str, str]]:
    payload = {
        "assessment_packet": packet.model_dump(),
        "classified_findings": findings,
    }
    user_prompt = f"""Create a foundation vendor assessment summary.

Input data:
{json.dumps(payload, ensure_ascii=True, indent=2, default=str)}

Return exactly this JSON shape:
{{
  "management_summary": "",
  "introduction": "",
  "objective": "",
  "key_findings": [],
  "strengths": [],
  "weaknesses": [],
  "risk_exposure": "",
  "conclusion": "",
  "missing_information": [],
  "source_question_ids": [],
  "from_assessment_data": "",
  "general_model_reasoning": ""
}}

Rules:
- Write for a business owner, not a security engineer.
- Use only the assessment packet and classified findings.
- Separate strengths from weaknesses.
- Treat partial and no compliance as weaknesses.
- Mention the vendor tier and why it matters.
- If evidence descriptions are missing, list that under missing_information.
- Keep each paragraph concise enough for a report field in PostgreSQL."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]

