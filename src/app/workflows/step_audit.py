from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.quality_gates import (
    validate_workflow_step_contract,
    validate_workflow_step_payload,
)


@dataclass(frozen=True)
class WorkflowAuditIssue:
    step_index: int
    step_name: str
    severity: str
    message: str
    fix: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "step_name": self.step_name,
            "severity": self.severity,
            "message": self.message,
            "fix": self.fix,
        }


def audit_workflow_run(run: dict[str, Any]) -> dict[str, Any]:
    steps = run.get("steps") or run.get("partial_steps") or []
    issues: list[WorkflowAuditIssue] = []
    previous_output: Any = None
    previous_name = ""
    for index, step in enumerate(steps, 1):
        name = str(step.get("name") or "")
        input_payload = step.get("input")
        output_payload = step.get("output")
        issues.extend(_gate_issues(index, name, input_payload, output_payload))
        if previous_output is not None:
            chain_issue = _chain_issue(
                index=index,
                name=name,
                previous_name=previous_name,
                previous_output=previous_output,
                input_payload=input_payload,
            )
            if chain_issue is not None:
                issues.append(chain_issue)
        issues.extend(_semantic_issues(index, name, input_payload, output_payload))
        previous_output = output_payload
        previous_name = name
    blocking = [issue for issue in issues if issue.severity == "blocking"]
    return {
        "passed": not blocking,
        "step_count": len(steps),
        "blocking_issue_count": len(blocking),
        "warning_issue_count": len(issues) - len(blocking),
        "issues": [issue.as_dict() for issue in issues],
    }


def _gate_issues(
    index: int,
    name: str,
    input_payload: Any,
    output_payload: Any,
) -> list[WorkflowAuditIssue]:
    issues: list[WorkflowAuditIssue] = []
    for gate in [
        validate_workflow_step_payload(
            step_name=name,
            input_payload=input_payload,
            output_payload=output_payload,
        ),
        validate_workflow_step_contract(
            step_name=name,
            input_payload=input_payload,
            output_payload=output_payload,
        ),
    ]:
        if gate.passed:
            continue
        for issue in gate.issues:
            issues.append(
                WorkflowAuditIssue(
                    step_index=index,
                    step_name=name,
                    severity=issue.severity,
                    message=f"{issue.field}: {issue.message}",
                    fix=issue.system_fix,
                )
            )
    return issues


def _chain_issue(
    *,
    index: int,
    name: str,
    previous_name: str,
    previous_output: Any,
    input_payload: Any,
) -> WorkflowAuditIssue | None:
    if input_payload == previous_output:
        return None
    if name == "Ask model to draft report paragraphs":
        packet = _dict(input_payload).get("validated_fact_packet_from_previous_step")
        if packet == previous_output:
            return None
    if name == "Prepare final result for the application":
        paragraphs = _dict(previous_output).get("parsed_paragraphs")
        if paragraphs and _dict(input_payload).get("paragraphs_from_previous_step") == paragraphs:
            return None
    return WorkflowAuditIssue(
        step_index=index,
        step_name=name,
        severity="blocking",
        message=(
            f"Step input is not the previous step output from '{previous_name}', and the "
            "auditor does not recognize a controlled derivation."
        ),
        fix=(
            "Make the previous step output the next step input, or add an explicit visible "
            "handoff packet that explains and contains the controlled derivation."
        ),
    )


def _semantic_issues(
    index: int,
    name: str,
    input_payload: Any,
    output_payload: Any,
) -> list[WorkflowAuditIssue]:
    issues: list[WorkflowAuditIssue] = []
    if name.startswith("Search standards evidence for "):
        prompt_sources = (
            _dict(_dict(output_payload).get("prompt_evidence")).get("prompt_sources") or []
        )
        source_has_control = any(
            _source_mentions_control(source) for source in prompt_sources
        )
        if prompt_sources and not source_has_control:
            issues.append(
                WorkflowAuditIssue(
                    step_index=index,
                    step_name=name,
                    severity="blocking",
                    message=(
                        "Selected prompt evidence does not appear to include a concrete "
                        "control or framework reference."
                    ),
                    fix=(
                        "Improve evidence selection so the model receives source excerpts with "
                        "control IDs, safeguard names, SCF labels, or framework references."
                    ),
                )
            )
    if name.startswith("Prepare model prompt for "):
        prompt_summary = _dict(_dict(output_payload).get("model_prompt"))
        previews = " ".join(
            str(message.get("preview") or "")
            for message in prompt_summary.get("messages", [])
            if isinstance(message, dict)
        )
        contract_preview = str(prompt_summary.get("output_contract_preview") or "")
        if "Return exactly this JSON shape" not in f"{previews}\n{contract_preview}":
            issues.append(
                WorkflowAuditIssue(
                    step_index=index,
                    step_name=name,
                    severity="blocking",
                    message="Prompt preview does not show the output contract.",
                    fix="Keep the prompt output contract visible in the compact prompt preview.",
                )
            )
    if name.startswith("Ask model to write risk answer for "):
        answer = _dict(output_payload).get("structured_answer") or {}
        if not _dict(answer).get("recommended_controls"):
            issues.append(
                WorkflowAuditIssue(
                    step_index=index,
                    step_name=name,
                    severity="blocking",
                    message="Accepted risk answer has no recommended controls.",
                    fix="Tighten retrieval, deterministic control extraction, or output gates.",
                )
            )
    return issues


def _source_mentions_control(source: object) -> bool:
    item = _dict(source)
    text = " ".join(
        [
            str(item.get("text_preview") or ""),
            str(_dict(item.get("metadata")).get("control_id") or ""),
            str(_dict(item.get("metadata")).get("framework") or ""),
        ]
    ).lower()
    return any(
        marker in text
        for marker in [
            "control",
            "safeguard",
            "scf",
            "nist",
            "iso",
            "cis",
            "pr.",
            "rc.",
            "end-",
        ]
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
