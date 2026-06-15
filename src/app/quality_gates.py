from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from app.schemas import RetrievedEvidence, StructuredRiskAnswer

GateSeverity = Literal["blocking", "warning"]


@dataclass(frozen=True)
class GateIssue:
    field: str
    message: str
    operator_fix: str
    system_fix: str
    severity: GateSeverity = "blocking"

    def as_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "message": self.message,
            "operator_fix": self.operator_fix,
            "system_fix": self.system_fix,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class GateResult:
    gate: str
    passed: bool
    summary: str
    issues: list[GateIssue] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "summary": self.summary,
            "issues": [issue.as_dict() for issue in self.issues],
        }


class QualityGateFailure(RuntimeError):
    def __init__(self, result: GateResult) -> None:
        self.result = result
        super().__init__(human_failure_message(result))


def human_failure_message(result: GateResult) -> str:
    parts = [
        f"Quality gate failed: {result.summary}",
    ]
    for index, issue in enumerate(result.issues, 1):
        parts.append(
            f"{index}. {issue.field}: {issue.message} "
            f"Operator fix: {issue.operator_fix} "
            f"System fix: {issue.system_fix}"
        )
    return " ".join(parts)


def passed_gate(gate: str, summary: str) -> GateResult:
    return GateResult(gate=gate, passed=True, summary=summary)


def failed_gate(gate: str, summary: str, issues: list[GateIssue]) -> GateResult:
    return GateResult(gate=gate, passed=False, summary=summary, issues=issues)


def validate_prompt_contract(
    *,
    gate: str,
    messages: list[dict[str, str]],
    required_phrases: list[str],
    task_name: str,
) -> GateResult:
    text = "\n".join(message.get("content", "") for message in messages)
    issues: list[GateIssue] = []
    for phrase in required_phrases:
        if phrase.lower() not in text.lower():
            issues.append(
                GateIssue(
                    field="messages",
                    message=f"Prompt is missing required instruction: {phrase}",
                    operator_fix=(
                        "Do not run this workflow with the current prompt. Ask the solution "
                        "owner to update the prompt template."
                    ),
                    system_fix=(
                        f"Update the {task_name} prompt builder so it includes this required "
                        "role, boundary, or output instruction."
                    ),
                )
            )
    if issues:
        return failed_gate(gate, f"The {task_name} prompt is incomplete.", issues)
    return passed_gate(gate, f"The {task_name} prompt contains the required control instructions.")


def validate_risk_answer(
    *,
    answer: StructuredRiskAnswer,
    evidence: list[RetrievedEvidence],
    raw_model_output: str,
    question: str,
) -> GateResult:
    issues: list[GateIssue] = []
    source_ids = {f"S{index}" for index, _ in enumerate(evidence, 1)}
    evidence_text = "\n".join(hit.chunk.text for hit in evidence).lower()

    if _looks_like_meta_failure(raw_model_output):
        issues.append(
            GateIssue(
                field="raw_model_output",
                message=(
                    "The model answered with meta-commentary, JSON repair text, or an apology "
                    "instead of the required risk analysis."
                ),
                operator_fix=(
                    "Do not approve this run. Retry after lowering the model freedom or choosing "
                    "a stronger model."
                ),
                system_fix=(
                    "Strengthen the prompt and keep validation retries enabled. If repeated, "
                    "reduce prompt size and send only the validated evidence packet."
                ),
            )
        )

    for field_name in ["threats", "vulnerabilities", "risks", "recommended_controls"]:
        values = getattr(answer, field_name)
        if not values:
            issues.append(
                GateIssue(
                    field=field_name,
                    message=f"The answer did not produce any {field_name.replace('_', ' ')}.",
                    operator_fix=(
                        "Check whether the standards corpus contains evidence for this gap. "
                        "If not, add or ingest relevant source material."
                    ),
                    system_fix=(
                        "If evidence exists, improve retrieval/reranking or the prompt so the "
                        "model must fill this section from the supplied evidence."
                    ),
                )
            )
        for value in values:
            if _is_bad_text(value):
                issues.append(_bad_text_issue(field_name, value))

    if not answer.risk_control_matrix:
        issues.append(
            GateIssue(
                field="risk_control_matrix",
                message="The answer did not produce a risk/control matrix.",
                operator_fix=(
                    "Do not approve the run; the analyst cannot trace gap to risk to "
                    "control."
                ),
                system_fix="Require at least one matrix row in the prompt and validator.",
            )
        )

    for row_index, row in enumerate(answer.risk_control_matrix):
        prefix = f"risk_control_matrix[{row_index}]"
        for field_name in ["gap", "threat", "vulnerability", "risk"]:
            value = getattr(row, field_name)
            if not value or _is_bad_text(value):
                issues.append(_bad_text_issue(f"{prefix}.{field_name}", value))
        if not row.controls:
            issues.append(
                GateIssue(
                    field=f"{prefix}.controls",
                    message="Matrix row has no concrete controls.",
                    operator_fix="Do not approve this risk row until controls are identified.",
                    system_fix=(
                        "Require controls per matrix row and extract controls from evidence."
                    ),
                )
            )
        for citation in row.evidence:
            if citation and citation not in source_ids:
                issues.append(
                    GateIssue(
                        field=f"{prefix}.evidence",
                        message=f"Matrix row cites {citation}, which was not supplied.",
                        operator_fix=(
                            "Do not approve the run because the citation cannot be traced."
                        ),
                        system_fix=(
                            "Reject invented citation IDs and only allow source IDs from "
                            "retrieval."
                        ),
                    )
                )

    for control in answer.recommended_controls:
        normalized = control.lower()
        if normalized not in evidence_text and not _looks_like_framework_label(control):
            issues.append(
                GateIssue(
                    field="recommended_controls",
                    message=f"Control appears unsupported by retrieved evidence: {control}",
                    operator_fix=(
                        "Check the source citations. If the control is truly needed, add source "
                        "material that contains it."
                    ),
                    system_fix=(
                        "Only promote controls extracted from retrieved evidence or validated "
                        "control-label extraction."
                    ),
                    severity="warning",
                )
            )

    if issues:
        return failed_gate(
            "risk_answer_output",
            (
                "The model output for this risk answer does not meet the expected "
                f"risk-analysis contract for: {question[:120]}"
            ),
            issues,
        )
    return passed_gate(
        "risk_answer_output",
        "Risk answer passed schema, content, citation, and evidence checks.",
    )


def validate_final_paragraphs(
    *,
    paragraphs: dict[str, str],
    raw_model_output: str,
    vendor_name: str,
    tier_level: int,
    weakness_summaries: list[str],
    validated_facts: dict[str, Any],
) -> GateResult:
    issues: list[GateIssue] = []
    required = [
        "management_summary",
        "introduction",
        "objective",
        "risk_exposure",
        "conclusion",
    ]
    for key in required:
        value = paragraphs.get(key, "")
        if not value:
            issues.append(
                GateIssue(
                    field=key,
                    message="Required paragraph is missing or empty.",
                    operator_fix="Do not approve the report. The required section was not drafted.",
                    system_fix=(
                        "Reject the model output and retry with the final paragraph repair "
                        "prompt."
                    ),
                )
            )
        elif _is_bad_text(value):
            issues.append(_bad_text_issue(key, value))

    joined = " ".join(paragraphs.values()).lower()
    if vendor_name.lower() not in joined:
        issues.append(
            GateIssue(
                field="paragraphs",
                message=f"Final paragraphs do not mention the assessed vendor: {vendor_name}.",
                operator_fix=(
                    "Do not approve the report; the business owner cannot tell who it is "
                    "about."
                ),
                system_fix="Require the vendor name in the final paragraph prompt and validator.",
            )
        )
    if f"tier {tier_level}" not in joined:
        issues.append(
            GateIssue(
                field="paragraphs",
                message=f"Final paragraphs do not mention Tier {tier_level}.",
                operator_fix="Ask for a corrected run that includes the vendor criticality.",
                system_fix="Require tier context in the final paragraph prompt and validator.",
            )
        )
    if _looks_like_meta_failure(raw_model_output):
        issues.append(
            GateIssue(
                field="raw_model_output",
                message=(
                    "The model discussed JSON/prompt mechanics instead of writing report "
                    "paragraphs."
                ),
                operator_fix="Do not approve this output. It is a failed model response.",
                system_fix="Send only the validated fact packet to the report writer and retry.",
            )
        )

    validated_terms = _validated_terms(validated_facts)
    if validated_terms and not any(term in joined for term in validated_terms):
        issues.append(
            GateIssue(
                field="risk_exposure",
                message=(
                    "Paragraphs do not mention any concrete validated risk/control terms from "
                    "the fact packet."
                ),
                operator_fix=(
                    "Do not approve the report; it is too generic. Review retrieval quality or "
                    "retry the report-writing step."
                ),
                system_fix=(
                    "Require final paragraphs to reference at least one validated weakness, risk, "
                    "or control term."
                ),
            )
        )

    if issues:
        return failed_gate(
            "final_paragraph_output",
            "Final report paragraphs failed quality checks.",
            issues,
        )
    return passed_gate(
        "final_paragraph_output",
        "Final report paragraphs passed schema, content, and consistency checks.",
    )


def repair_prompt(
    *,
    original_messages: list[dict[str, str]],
    gate_result: GateResult,
) -> list[dict[str, str]]:
    repair_instructions = {
        "task": "Repair the previous answer only.",
        "validation_errors": gate_result.as_dict(),
        "rules": [
            "Use the same trusted input and evidence as the original prompt.",
            "Do not add new facts, controls, citations, or assumptions.",
            "Return only the required JSON object.",
            "Do not explain JSON, critique the prompt, or include markdown.",
        ],
    }
    return [
        *original_messages,
        {
            "role": "user",
            "content": (
                "Your previous answer failed quality gates. Correct the answer using only the "
                "same trusted source material.\n\n"
                f"{json.dumps(repair_instructions, indent=2, ensure_ascii=True)}"
            ),
        },
    ]


def _looks_like_meta_failure(text: str) -> bool:
    lower = text.lower()
    phrases = [
        "invalid json",
        "corrected json",
        "json structure",
        "key issues",
        "next steps",
        "let me know",
        "as an ai",
        "i cannot",
        "i'm sorry",
    ]
    return any(phrase in lower for phrase in phrases)


def _is_bad_text(value: str) -> bool:
    clean = str(value or "").strip()
    if not clean:
        return True
    lower = clean.lower()
    if lower in {"see retrieved evidence", "n/a", "none", "unknown"}:
        return True
    if "see retrieved evidence" in lower:
        return True
    if re.search(r"\b(\w+)(?:\s+\1\b){5,}", lower):
        return True
    if len(clean) > 20 and sum(ch.isalpha() for ch in clean) / max(len(clean), 1) < 0.45:
        return True
    return False


def _bad_text_issue(field: str, value: object) -> GateIssue:
    preview = str(value or "")[:120]
    return GateIssue(
        field=field,
        message=f"Value is missing, placeholder-like, malformed, or not usable: {preview}",
        operator_fix="Do not approve this output. Re-run after improving evidence or model choice.",
        system_fix="Reject placeholder/malformed values and retry with a repair prompt.",
    )


def _looks_like_framework_label(control: str) -> bool:
    return bool(
        re.search(
            r"\b(CIS|NIST|ISO|SCF|PR\.|RC\.|ID\.|DE\.|RS\.|GV\.|BCD-|END-)\b",
            control,
            re.I,
        )
    )


def _validated_terms(validated_facts: dict[str, Any]) -> set[str]:
    raw = str(validated_facts).lower()
    terms: set[str] = set()
    for candidate in [
        "anti-malware",
        "malware",
        "ransomware",
        "endpoint",
        "recovery",
        "disaster",
        "business continuity",
        "resilience",
    ]:
        if candidate in raw:
            terms.add(candidate)
    return terms
