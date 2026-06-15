from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from app.schemas import RetrievedEvidence, StructuredRiskAnswer

GateSeverity = Literal["blocking", "warning"]

RISK_LABEL_MAX_WORDS = 12
CONTROL_LABEL_MAX_WORDS = 18
MATRIX_CELL_MAX_WORDS = 16
STORYLINE_FIELD_MAX_WORDS = 55
FINAL_PARAGRAPH_MAX_WORDS = 90


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
    if result.gate == "gap_storyline_output":
        return (
            "The workflow stopped while explaining one questionnaire gap in business terms. "
            "The model did not connect the gap to a threat, vulnerability, risk, named control, "
            "and residual concern using the validated chain. Do not approve this run; retry with "
            "a stronger model or ask the solution owner to tighten the storyline prompt and "
            "validator."
        )
    if result.gate == "risk_answer_output":
        fields = {issue.field for issue in result.issues}
        missing_matrix = any(field.startswith("risk_control_matrix") for field in fields)
        if missing_matrix:
            return (
                "The workflow stopped while building a risk/control matrix. Standards evidence "
                "was retrieved, but the selected model did not return complete gap, threat, "
                "vulnerability, risk, control, and evidence values after the repair attempts. "
                "Do not approve this run; retry with a stronger model or ask the solution owner "
                "to improve graph filtering and the repair prompt."
            )
        if "raw_model_output" in fields:
            return (
                "The workflow stopped because the selected model did not return usable structured "
                "risk analysis. Do not approve this run; retry with a stronger model or reduce "
                "the prompt context."
            )
        return (
            "The workflow stopped because the selected model response did not meet the required "
            "risk-analysis structure. Do not approve this run until the failed step is corrected."
        )
    if result.gate == "final_paragraph_output":
        return (
            "The workflow stopped while drafting the business report. The model response did not "
            "match the required report sections or did not use the validated facts. Do not approve "
            "this run until the report-writing step is corrected."
        )
    if result.gate == "workflow_step_payload":
        return (
            "The workflow stopped because one step tried to pass data that was too large or too "
            "technical to the next step. This usually means debug details, full prompts, or raw "
            "model output leaked into the normal workflow handoff. The solution owner must compact "
            "that step so it passes only business facts, short source summaries, and clear status."
        )
    if result.gate == "workflow_step_contract":
        return (
            "The workflow stopped because one step did not produce the type of business handoff "
            "the next step needs. Do not approve this run. The solution owner must fix the step "
            "logic, filters, prompt builder, or validator before running it again."
        )
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


def validate_prompt_quality(
    *,
    gate: str,
    messages: list[dict[str, str]],
    task_name: str,
    max_total_chars: int,
    max_user_chars: int,
    max_source_markers: int,
) -> GateResult:
    text = "\n".join(message.get("content", "") for message in messages)
    user_text = "\n".join(
        message.get("content", "") for message in messages if message.get("role") == "user"
    )
    issues: list[GateIssue] = []
    if len(text) > max_total_chars:
        issues.append(
            GateIssue(
                field="messages",
                message=(
                    f"The {task_name} prompt is too large "
                    f"({len(text):,} characters > {max_total_chars:,})."
                ),
                operator_fix=(
                    "Do not run this prompt. It asks the model to process too much context for "
                    "one step."
                ),
                system_fix=(
                    "Curate the evidence packet before prompt construction and send only the "
                    "best source excerpts needed for this one decision."
                ),
            )
        )
    if len(user_text) > max_user_chars:
        issues.append(
            GateIssue(
                field="user_message",
                message=(
                    f"The {task_name} user instruction is too large "
                    f"({len(user_text):,} characters > {max_user_chars:,})."
                ),
                operator_fix="Do not approve this run; the model input is not focused enough.",
                system_fix=(
                    "Reduce retrieved excerpts, remove repeated plan text, and split the step "
                    "if one model call needs too much material."
                ),
            )
        )
    source_markers = len(re.findall(r"(?m)^\[S\d+\]", text))
    if source_markers > max_source_markers:
        issues.append(
            GateIssue(
                field="retrieved_evidence",
                message=(
                    f"The prompt includes too many cited source excerpts "
                    f"({source_markers} > {max_source_markers})."
                ),
                operator_fix=(
                    "Do not run this prompt; the analyst cannot reasonably inspect why the "
                    "model received this much evidence for one weak answer."
                ),
                system_fix=(
                    "Use the prompt evidence selector to keep only the most relevant compact "
                    "source excerpts."
                ),
            )
        )
    noisy_fragments = [
        '"sub_questions"',
        '"retrieval_queries"',
        "messages_sent_to_model",
        "raw_model_response",
        "prompt_messages",
    ]
    leaked = [fragment for fragment in noisy_fragments if fragment in text]
    if leaked:
        issues.append(
            GateIssue(
                field="messages",
                message=(
                    "The prompt contains workflow/debug structures instead of a clean analyst "
                    f"instruction: {', '.join(leaked)}."
                ),
                operator_fix="Do not run this prompt; it is not readable enough for review.",
                system_fix=(
                    "Format the prompt as role, objective, trusted inputs, and output contract. "
                    "Do not paste planner/debug JSON into the model request."
                ),
            )
        )
    if issues:
        return failed_gate(
            gate,
            f"The {task_name} prompt is not focused enough for a controlled model call.",
            issues,
        )
    return passed_gate(gate, f"The {task_name} prompt is focused and bounded.")


def combine_gate_results(gate: str, results: list[GateResult]) -> GateResult:
    issues = [issue for result in results for issue in result.issues]
    failed = [result for result in results if not result.passed]
    if failed:
        return failed_gate(
            gate,
            "One or more required quality checks failed.",
            issues,
        )
    if issues:
        return GateResult(
            gate=gate,
            passed=True,
            summary="All required quality checks passed with warnings.",
            issues=issues,
        )
    return passed_gate(gate, "All required quality checks passed.")


def validate_workflow_step_payload(
    *,
    step_name: str,
    input_payload: Any,
    output_payload: Any,
    max_payload_chars: int = 60_000,
    max_model_prompt_chars: int = 12_000,
) -> GateResult:
    issues: list[GateIssue] = []
    input_size = _json_size(input_payload)
    output_size = _json_size(output_payload)
    if input_size > max_payload_chars:
        issues.append(
            GateIssue(
                field="input",
                message=(
                    f"Step input is too large for a readable workflow handoff "
                    f"({input_size:,} characters)."
                ),
                operator_fix=(
                    "Do not approve this run. The workflow is carrying too much internal data "
                    "between steps."
                ),
                system_fix=(
                    "Pass a compact business object to the next step and move full debug data "
                    "to an explicit debug log or preview artifact."
                ),
            )
        )
    if output_size > max_payload_chars:
        issues.append(
            GateIssue(
                field="output",
                message=(
                    f"Step output is too large for a readable workflow handoff "
                    f"({output_size:,} characters)."
                ),
                operator_fix=(
                    "Do not approve this run. The workflow produced an oversized intermediate "
                    "result."
                ),
                system_fix=(
                    "Compact the step output before it becomes input to the next step. Keep "
                    "only business facts, source IDs, and short previews."
                ),
            )
        )
    forbidden_keys = _find_forbidden_debug_keys(output_payload)
    if forbidden_keys:
        issues.append(
            GateIssue(
                field="output",
                message=(
                    "Step output contains debug-only fields that should not travel through the "
                    f"normal workflow: {', '.join(sorted(forbidden_keys))}."
                ),
                operator_fix=(
                    "Do not approve this run. The workflow is mixing debug material with "
                    "business handoff data."
                ),
                system_fix=(
                    "Remove debug, prompt_messages, messages_sent_to_model, and full raw model "
                    "responses from normal step outputs. Store them only in debug logs or "
                    "explicit full-detail previews."
                ),
            )
        )
    prompt_chars = _max_model_prompt_chars(output_payload)
    if prompt_chars > max_model_prompt_chars:
        issues.append(
            GateIssue(
                field="output.model_prompt",
                message=(
                    f"Model prompt is too large ({prompt_chars:,} characters)."
                ),
                operator_fix=(
                    "Do not approve this run. The model is being asked to process too much "
                    "context for this one step."
                ),
                system_fix=(
                    "Use compact search focus and source excerpts. Do not include full planner "
                    "JSON, full chunks, or repeated debug data in the prompt."
                ),
            )
        )
    if issues:
        return failed_gate(
            "workflow_step_payload",
            f"Workflow step '{step_name}' is not clean enough to pass to the next step.",
            issues,
        )
    return passed_gate(
        "workflow_step_payload",
        f"Workflow step '{step_name}' passed payload hygiene checks.",
    )


def validate_workflow_step_contract(
    *,
    step_name: str,
    input_payload: Any,
    output_payload: Any,
) -> GateResult:
    issues: list[GateIssue] = []
    if not step_name.strip():
        issues.append(_step_issue("name", "Step name is missing."))

    if step_name.startswith("Plan searches for "):
        if "search_plan" in _dict(output_payload) or "retrieval_queries" in _dict(output_payload):
            issues.append(
                _step_issue(
                    "output",
                    (
                        "Planning output exposes internal planner JSON instead of a readable "
                        "search focus."
                    ),
                    system_fix=(
                        "Return compact search_focus entries and keep full query strings inside "
                        "the retriever."
                    ),
                )
            )
        focus = _dict(output_payload).get("search_focus")
        if not isinstance(focus, list) or len(focus) < 4:
            issues.append(
                _step_issue(
                    "output.search_focus",
                    "Search focus is missing or incomplete.",
                )
            )

    if step_name.startswith("Search standards evidence for "):
        output = _dict(output_payload)
        prompt_evidence = _dict(output.get("prompt_evidence"))
        prompt_count = int(prompt_evidence.get("prompt_source_count") or 0)
        retrieved_count = int(prompt_evidence.get("retrieved_source_count") or 0)
        if prompt_count <= 0:
            issues.append(
                _step_issue(
                    "output.prompt_evidence",
                    "Retrieval did not select any compact evidence for the model prompt.",
                    operator_fix=(
                        "Do not approve a run that asks the model to answer without selected "
                        "source evidence."
                    ),
                    system_fix=(
                        "Improve retrieval or evidence selection. If no source is available, "
                        "return insufficient evidence instead of calling the model."
                    ),
                )
            )
        if prompt_count > 5:
            issues.append(
                _step_issue(
                    "output.prompt_evidence",
                    f"Too many excerpts selected for the model prompt ({prompt_count} > 5).",
                    system_fix="Keep only the highest-value excerpts for the current weak answer.",
                )
            )
        if prompt_count > retrieved_count:
            issues.append(
                _step_issue(
                    "output.prompt_evidence",
                    "Selected prompt evidence count is larger than retrieved evidence count.",
                )
            )
        for index, source in enumerate(prompt_evidence.get("prompt_sources") or []):
            preview = str(_dict(source).get("text_preview") or "")
            if len(preview) > 700:
                issues.append(
                    _step_issue(
                        f"output.prompt_evidence.prompt_sources[{index}].text_preview",
                        "Selected evidence preview is too long for a focused handoff.",
                    )
                )

    if step_name.startswith("Prepare model prompt for "):
        output = _dict(output_payload)
        prompt = _dict(output.get("model_prompt"))
        chars = int(prompt.get("total_characters_sent_to_model") or 0)
        if chars <= 0:
            issues.append(_step_issue("output.model_prompt", "Model prompt summary is missing."))
        elif chars > 8_500:
            issues.append(
                _step_issue(
                    "output.model_prompt",
                    f"Risk-answer prompt is too large ({chars:,} characters > 8,500).",
                    system_fix=(
                        "Select fewer evidence excerpts, shorten excerpts, or split the model "
                        "task into narrower calls."
                    ),
                )
            )
        if _dict(output.get("quality_gate")).get("passed") is not True:
            issues.append(
                _step_issue(
                    "output.quality_gate",
                    "Prompt quality gate did not pass before the model call.",
                )
            )

    if step_name.startswith("Ask model to write risk answer for "):
        output = _dict(output_payload)
        if "raw_model_response" in output:
            issues.append(
                _step_issue(
                    "output.raw_model_response",
                    "Full raw model output leaked into the visible workflow handoff.",
                )
            )
        if _dict(output.get("quality_gate")).get("passed") is not True:
            issues.append(
                _step_issue(
                    "output.quality_gate",
                    "Risk answer was not accepted by the output quality gate.",
                )
            )
        structured = _dict(output.get("structured_answer"))
        if not structured.get("risk_control_matrix"):
            issues.append(
                _step_issue(
                    "output.structured_answer.risk_control_matrix",
                    "Risk answer does not contain a risk/control matrix.",
                )
            )

    if step_name == "Build validated fact packet for report drafting":
        facts = _dict(output_payload).get("validated_risk_facts")
        if not isinstance(facts, list):
            issues.append(
                _step_issue(
                    "output.validated_risk_facts",
                    "Validated fact packet does not contain validated risk facts.",
                )
            )

    if step_name == "Prepare final report model prompt":
        output = _dict(output_payload)
        prompt = _dict(output.get("model_prompt"))
        chars = int(prompt.get("total_characters_sent_to_model") or 0)
        if chars <= 0:
            issues.append(_step_issue("output.model_prompt", "Final report prompt is missing."))
        elif chars > 12_000:
            issues.append(
                _step_issue(
                    "output.model_prompt",
                    f"Final report prompt is too large ({chars:,} characters > 12,000).",
                    system_fix=(
                        "Compact the report fact packet and use the storyline report instead "
                        "of full risk-chain/debug data."
                    ),
                )
            )
        if _dict(output.get("quality_gate")).get("passed") is not True:
            issues.append(
                _step_issue(
                    "output.quality_gate",
                    "Final report prompt contract gate did not pass before the model call.",
                )
            )
        if _dict(output.get("prompt_quality_gate")).get("passed") is not True:
            issues.append(
                _step_issue(
                    "output.prompt_quality_gate",
                    "Final report prompt quality gate did not pass before the model call.",
                )
            )

    if step_name == "Build risk assessment chains and added-value delta":
        chains = _dict(output_payload).get("risk_assessment_chains")
        delta = _dict(output_payload).get("toolchain_delta")
        if not isinstance(chains, list) or not chains:
            issues.append(
                _step_issue(
                    "output.risk_assessment_chains",
                    "Risk assessment chains are missing.",
                    system_fix=(
                        "Build one risk assessment chain per weak questionnaire answer before "
                        "report drafting."
                    ),
                )
            )
        if not isinstance(delta, dict) or not delta.get("added_by_rag"):
            issues.append(
                _step_issue(
                    "output.toolchain_delta",
                    "Added-value delta does not show what RAG added beyond SQL.",
                    system_fix=(
                        "Compare initial assessment gaps with standards-backed risk facts and "
                        "store the added controls, risks, and resilience findings."
                    ),
                )
            )

    if step_name.startswith("Prepare storyline model prompt for "):
        output = _dict(output_payload)
        prompt = _dict(output.get("model_prompt"))
        chars = int(prompt.get("total_characters_sent_to_model") or 0)
        if chars <= 0:
            issues.append(
                _step_issue("output.model_prompt", "Storyline prompt summary is missing.")
            )
        elif chars > 7_000:
            issues.append(
                _step_issue(
                    "output.model_prompt",
                    f"Storyline prompt is too large ({chars:,} characters > 7,000).",
                    system_fix=(
                        "Send only the selected risk chain, compact controls, and compact "
                        "source references for this one gap."
                    ),
                )
            )
        if _dict(output.get("quality_gate")).get("passed") is not True:
            issues.append(
                _step_issue(
                    "output.quality_gate",
                    "Storyline prompt contract gate did not pass before the model call.",
                )
            )

    if step_name.startswith("Ask model to explain risk chain for "):
        output = _dict(output_payload)
        if _dict(output.get("quality_gate")).get("passed") is not True:
            issues.append(
                _step_issue(
                    "output.quality_gate",
                    "Business storyline was not accepted by the output quality gate.",
                )
            )
        storyline = _dict(output.get("business_storyline"))
        required = [
            "gap_story",
            "business_meaning",
            "risk_logic",
            "control_logic",
            "resilience_logic",
            "residual_conclusion",
        ]
        missing = [field for field in required if not str(storyline.get(field) or "").strip()]
        if missing:
            issues.append(
                _step_issue(
                    "output.business_storyline",
                    f"Business storyline is missing fields: {', '.join(missing)}.",
                )
            )

    if step_name == "Ask model to draft report paragraphs":
        prompt = _dict(_dict(input_payload).get("model_prompt"))
        chars = int(prompt.get("total_characters_sent_to_model") or 0)
        if chars > 16_000:
            issues.append(
                _step_issue(
                    "input.model_prompt",
                    f"Final report prompt is too large ({chars:,} characters > 16,000).",
                    system_fix=(
                        "Compact the validated fact packet before report drafting and avoid "
                        "sending retrieval/debug detail."
                    ),
                )
            )
        if _dict(output_payload).get("parsed_paragraphs") is None:
            issues.append(
                _step_issue(
                    "output.parsed_paragraphs",
                    "Report model output did not parse into the required paragraph fields.",
                )
            )
        if _dict(_dict(output_payload).get("quality_gate")).get("passed") is not True:
            issues.append(
                _step_issue(
                    "output.quality_gate",
                    "Final paragraph quality gate did not pass.",
                )
            )

    if issues:
        return failed_gate(
            "workflow_step_contract",
            f"Workflow step '{step_name}' failed its business handoff contract.",
            issues,
        )
    return passed_gate(
        "workflow_step_contract",
        f"Workflow step '{step_name}' passed its business handoff contract.",
    )


def validate_gap_storyline(
    *,
    storyline: dict[str, Any],
    risk_chain: dict[str, Any],
    raw_model_output: str,
) -> GateResult:
    issues: list[GateIssue] = []
    required_fields = [
        "question_id",
        "gap_story",
        "business_meaning",
        "risk_logic",
        "control_logic",
        "resilience_logic",
        "residual_conclusion",
    ]
    for field_name in required_fields:
        value = str(storyline.get(field_name) or "").strip()
        if not value or _is_bad_text(value):
            issues.append(_storyline_issue(field_name, "Storyline field is missing or unusable."))
        elif _word_count(value) > STORYLINE_FIELD_MAX_WORDS:
            issues.append(
                _storyline_issue(
                    field_name,
                    (
                        "Storyline field is too long for a reviewer to inspect quickly "
                        f"({_word_count(value)} words > {STORYLINE_FIELD_MAX_WORDS})."
                    ),
                    system_fix=(
                        "Constrain the storyline prompt to one or two concise sentences per "
                        "field, with no methodology explanation."
                    ),
                )
            )

    expected_question_id = str(risk_chain.get("question_id") or "").strip()
    if (
        expected_question_id
        and str(storyline.get("question_id") or "").strip() != expected_question_id
    ):
        issues.append(
            _storyline_issue(
                "question_id",
                "Storyline question_id does not match the selected risk chain.",
                system_fix="Force the model to copy the exact question_id from the input chain.",
            )
        )

    raw_lower = raw_model_output.lower()
    if _looks_like_meta_failure(raw_model_output):
        issues.append(
            _storyline_issue(
                "raw_model_output",
                "The model answered with meta-commentary instead of the storyline contract.",
                system_fix=(
                    "Retry with a repair prompt or use the deterministic storyline renderer."
                ),
            )
        )
    forbidden = _unsupported_urgency_terms(raw_lower, _json_lower(risk_chain))
    if forbidden:
        issues.append(
            _storyline_issue(
                "business_storyline",
                f"Storyline uses unsupported urgency or severity wording: {', '.join(forbidden)}.",
                system_fix=(
                    "Reject unsupported urgency words unless the validated chain explicitly "
                    "contains them."
                ),
            )
        )

    combined = " ".join(
        str(storyline.get(field) or "")
        for field in required_fields
        if field != "question_id"
    ).lower()
    required_groups = {
        "confirmed gap": risk_chain.get("confirmed_gaps") or [],
        "threat": risk_chain.get("threat_scenarios") or [],
        "vulnerability": risk_chain.get("vulnerabilities") or [],
        "risk": [_dict(risk_chain.get("inherent_risk")).get("risk_statement")],
        "control": [
            _dict(requirement).get("control") or requirement
            for requirement in (risk_chain.get("standards_requirements_added") or [])
        ],
        "resilience or residual concern": [
            *(risk_chain.get("resilience_effects") or []),
            _dict(risk_chain.get("residual_concern")).get("remaining_issue"),
        ],
    }
    for label, candidates in required_groups.items():
        if not _any_meaningful_overlap(combined, candidates):
            issues.append(
                _storyline_issue(
                    label,
                    f"Storyline does not clearly reuse the validated {label} from the chain.",
                    system_fix=(
                        "Build the prompt from compact validated chain fields and require the "
                        f"storyline to explain the selected {label}."
                    ),
                )
            )

    if issues:
        return failed_gate(
            "gap_storyline_output",
            "A business storyline did not meet the gap-to-risk explanation contract.",
            issues,
        )
    return passed_gate(
        "gap_storyline_output",
        "The business storyline connects the gap to risk, controls, and residual concern.",
    )


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _step_issue(
    field: str,
    message: str,
    *,
    operator_fix: str = (
        "Do not approve this run. The workflow step did not produce a clean handoff."
    ),
    system_fix: str = (
        "Fix the step implementation so it passes the expected compact business object to "
        "the next step."
    ),
) -> GateIssue:
    return GateIssue(
        field=field,
        message=message,
        operator_fix=operator_fix,
        system_fix=system_fix,
    )


def _storyline_issue(
    field: str,
    message: str,
    *,
    operator_fix: str = (
        "Do not approve this storyline. Review the selected gap and rerun after evidence, "
        "prompt, or model settings are corrected."
    ),
    system_fix: str = (
        "Fix the storyline prompt, parser, or validator so the model explains only the "
        "validated risk chain."
    ),
) -> GateIssue:
    return GateIssue(
        field=field,
        message=message,
        operator_fix=operator_fix,
        system_fix=system_fix,
    )


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
    support_text = f"{question}\n{evidence_text}".lower()

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

    if _word_count(answer.executive_summary) > 45:
        issues.append(
            GateIssue(
                field="executive_summary",
                message=(
                    "Executive summary is too long for the structured risk answer "
                    f"({_word_count(answer.executive_summary)} words > 45)."
                ),
                operator_fix="Do not approve verbose structured risk summaries.",
                system_fix=(
                    "Repair the model answer into one compact sentence that preserves the "
                    "validated gap, risk, and control direction."
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
            if field_name in {"threats", "vulnerabilities", "risks"} and not _has_term_support(
                value,
                support_text,
            ):
                issues.append(_unsupported_label_issue(field_name, value))
            max_words = (
                CONTROL_LABEL_MAX_WORDS
                if field_name == "recommended_controls"
                else RISK_LABEL_MAX_WORDS
            )
            if _word_count(value) > max_words:
                issues.append(
                    GateIssue(
                        field=field_name,
                        message=(
                            "Value is too verbose for a structured risk field "
                            f"({_word_count(value)} words > {max_words}). Use a short "
                            f"label or phrase: {str(value)[:120]}"
                        ),
                        operator_fix="Do not approve verbose structured fields.",
                        system_fix=(
                            "Tighten the prompt or add a compression pass so structured fields "
                            "use concise labels instead of prose."
                        ),
                    )
                )
        max_items = 5 if field_name == "recommended_controls" else 3
        if len(values) > max_items:
            issues.append(
                GateIssue(
                    field=field_name,
                    message=(
                        f"Too many items for a concise risk answer "
                        f"({len(values)} > {max_items})."
                    ),
                    operator_fix="Do not approve this run; the section is too broad to review.",
                    system_fix=(
                        "Limit the prompt and validator to the highest-value items only."
                    ),
                )
            )

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
            elif field_name in {"threat", "vulnerability", "risk"} and not _has_term_support(
                value,
                support_text,
            ):
                issues.append(_unsupported_label_issue(f"{prefix}.{field_name}", value))
            elif _word_count(value) > MATRIX_CELL_MAX_WORDS:
                issues.append(
                    GateIssue(
                        field=f"{prefix}.{field_name}",
                        message=(
                            "Matrix value is too verbose "
                            f"({_word_count(value)} words > {MATRIX_CELL_MAX_WORDS}). "
                            f"Use a short business phrase: {str(value)[:120]}"
                        ),
                        operator_fix="Do not approve rows that bury the finding in prose.",
                        system_fix=(
                            "Compress matrix cells to concise phrases before passing them "
                            "forward."
                        ),
                    )
                )
        if len(answer.risk_control_matrix) > 3:
            issues.append(
                GateIssue(
                    field="risk_control_matrix",
                    message="Risk/control matrix has too many rows for this step.",
                    operator_fix="Do not approve overly broad matrix output.",
                    system_fix="Keep only the top three most relevant rows for this weakness.",
                )
            )
            break
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

    blocking = [issue for issue in issues if issue.severity == "blocking"]
    if blocking:
        return failed_gate(
            "risk_answer_output",
            (
                "The model output for this risk answer does not meet the expected "
                f"risk-analysis contract for: {question[:120]}"
            ),
            issues,
        )
    if issues:
        return GateResult(
            gate="risk_answer_output",
            passed=True,
            summary=(
                "Risk answer passed required checks with warnings that should be reviewed."
            ),
            issues=issues,
        )
    return passed_gate(
        "risk_answer_output",
        "Risk answer passed schema, content, citation, and evidence checks.",
    )


def validate_risk_assessment_chains(value: dict[str, Any]) -> GateResult:
    issues: list[GateIssue] = []
    chains = value.get("risk_assessment_chains")
    delta = value.get("toolchain_delta")
    if not isinstance(chains, list) or not chains:
        issues.append(
            GateIssue(
                field="risk_assessment_chains",
                message="No risk assessment chains were produced.",
                operator_fix=(
                    "Do not approve this run. The workflow has not translated gaps into a "
                    "risk assessment model."
                ),
                system_fix=(
                    "Build one chain per weak answer with known facts, standards additions, "
                    "risks, controls, resilience, residual concern, and missing evidence."
                ),
            )
        )
    for index, chain in enumerate(chains or []):
        prefix = f"risk_assessment_chains[{index}]"
        required = [
            "known_from_assessment",
            "standards_requirements_added",
            "confirmed_gaps",
            "threat_scenarios",
            "vulnerabilities",
            "inherent_risk",
            "recommended_controls_by_function",
            "control_effects",
            "residual_concern",
            "added_value_summary",
        ]
        for field_name in required:
            if not _dict(chain).get(field_name):
                issues.append(
                    GateIssue(
                        field=f"{prefix}.{field_name}",
                        message=(
                            "Risk assessment chain is missing required business analysis "
                            f"field: {field_name}."
                        ),
                        operator_fix=(
                            "Do not approve this run. The final report would not have enough "
                            "material to answer the risk question."
                        ),
                        system_fix=(
                            "Populate this field before report drafting, or fail with "
                            "insufficient evidence."
                        ),
                    )
                )
        if not _dict(_dict(chain).get("inherent_risk")).get("risk_statement"):
            issues.append(
                GateIssue(
                    field=f"{prefix}.inherent_risk.risk_statement",
                    message="Inherent risk statement is missing.",
                    operator_fix="Do not approve a risk chain with no risk statement.",
                    system_fix="Carry the validated risk from the RAG answer into the risk chain.",
                )
            )
    if not isinstance(delta, dict) or not delta.get("added_by_rag"):
        issues.append(
            GateIssue(
                field="toolchain_delta.added_by_rag",
                message="The workflow did not record what RAG added beyond the SQL facts.",
                operator_fix=(
                    "Do not approve a final report that cannot show the analysis added by the "
                    "toolchain."
                ),
                system_fix=(
                    "Create a deterministic delta comparing known assessment facts with "
                    "standards-backed controls, risks, resilience findings, and uncertainty."
                ),
            )
        )
    if issues:
        return failed_gate(
            "risk_assessment_chain",
            "Risk assessment chain did not meet the required product-analysis contract.",
            issues,
        )
    return passed_gate(
        "risk_assessment_chain",
        (
            "Risk assessment chains include known facts, added RAG value, risk, "
            "controls, and residual concern."
        ),
    )


def prune_unsupported_risk_answer(
    *,
    answer: StructuredRiskAnswer,
    evidence: list[RetrievedEvidence],
    question: str,
) -> StructuredRiskAnswer:
    support_text = f"{question}\n" + "\n".join(hit.chunk.text for hit in evidence).lower()
    threats = _supported_values(answer.threats, support_text)
    vulnerabilities = _supported_values(answer.vulnerabilities, support_text)
    risks = _supported_values(answer.risks, support_text)
    rows = []
    for row in answer.risk_control_matrix:
        if not all(
            _has_term_support(value, support_text)
            for value in [row.threat, row.vulnerability, row.risk]
        ):
            continue
        rows.append(row)
    return answer.model_copy(
        update={
            "threats": threats or answer.threats,
            "vulnerabilities": vulnerabilities or answer.vulnerabilities,
            "risks": risks or answer.risks,
            "risk_control_matrix": rows or answer.risk_control_matrix,
        }
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
        elif _word_count(value) > FINAL_PARAGRAPH_MAX_WORDS:
            issues.append(
                GateIssue(
                    field=key,
                    message=(
                        "Report paragraph is too long for the business summary contract "
                        f"(maximum {FINAL_PARAGRAPH_MAX_WORDS} words)."
                    ),
                    operator_fix=(
                        "Do not approve this report section; it is too verbose for review."
                    ),
                    system_fix=(
                        "Constrain final report paragraphs to the most important facts first "
                        "and remove background explanation."
                    ),
                )
            )

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

    if "acceptable risk" in joined and "acceptable risk" not in str(validated_facts).lower():
        issues.append(
            GateIssue(
                field="risk_exposure",
                message=(
                    "Final paragraphs mention acceptable risk thresholds, but the validated "
                    "fact packet does not define an acceptance threshold."
                ),
                operator_fix=(
                    "Do not approve the report; it makes a risk-acceptance statement that was "
                    "not established by the assessment facts."
                ),
                system_fix=(
                    "Forbid threshold or acceptance statements unless the validated fact packet "
                    "contains the threshold and decision basis."
                ),
            )
        )

    validated_text = str(validated_facts).lower()
    unsupported_emphasis = [
        term
        for term in ["critical", "immediate", "unacceptable", "severe"]
        if term in joined and term not in validated_text
    ]
    if unsupported_emphasis:
        issues.append(
            GateIssue(
                field="paragraphs",
                message=(
                    "Final paragraphs use urgency or severity wording that is not present "
                    f"in the validated facts: {', '.join(unsupported_emphasis)}."
                ),
                operator_fix=(
                    "Do not approve this report; the wording overstates what the evidence "
                    "supports."
                ),
                system_fix=(
                    "Calibrate final report wording to the validated facts. Use high/medium/low "
                    "only when those ratings are in the fact packet, and avoid urgency words "
                    "unless the packet contains them."
                ),
            )
        )

    delta = _dict(validated_facts.get("toolchain_delta"))
    added_by_rag = [str(item) for item in delta.get("added_by_rag") or []]
    added_by_resilience = [
        str(item) for item in delta.get("added_by_resilience_analysis") or []
    ]
    added_terms = _important_terms([*added_by_rag, *added_by_resilience])
    if added_terms and not any(term in joined for term in added_terms):
        issues.append(
            GateIssue(
                field="paragraphs",
                message=(
                    "Final paragraphs do not explain the standards/RAG value added beyond "
                    "the questionnaire gaps."
                ),
                operator_fix=(
                    "Do not approve this report; it repeats assessment gaps without showing "
                    "the actual risk analysis."
                ),
                system_fix=(
                    "Use toolchain_delta and risk_assessment_chains in the report prompt, and "
                    "require at least one standards/control/risk/resilience term in the final "
                    "paragraphs."
                ),
            )
        )

    framework_terms = _framework_terms(added_by_rag)
    risk_or_conclusion = " ".join(
        [
            paragraphs.get("risk_exposure", ""),
            paragraphs.get("conclusion", ""),
        ]
    ).lower()
    if framework_terms and not any(term in risk_or_conclusion for term in framework_terms):
        issues.append(
            GateIssue(
                field="risk_exposure",
                message=(
                    "Risk exposure or conclusion does not name any standards/control reference "
                    "that RAG added."
                ),
                operator_fix=(
                    "Do not approve this report; it hides the concrete control mapping behind "
                    "generic wording."
                ),
                system_fix=(
                    "Require risk_exposure or conclusion to name at least one added framework "
                    "or control reference such as CIS, SCF, NIST, END-04, or BCD-01."
                ),
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
            "Keep every important validated fact, but rewrite it shorter and more directly.",
            "Remove filler, methodology explanation, duplicated reasoning, and broad background.",
            "Use natural professional language: factual, calm, specific, and concise.",
            "If a field failed for length, summarize it without changing its meaning.",
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


def _word_count(value: object) -> int:
    return len(re.findall(r"\b\w+\b", str(value or "")))


def _bad_text_issue(field: str, value: object) -> GateIssue:
    preview = str(value or "")[:120]
    return GateIssue(
        field=field,
        message=f"Value is missing, placeholder-like, malformed, or not usable: {preview}",
        operator_fix="Do not approve this output. Re-run after improving evidence or model choice.",
        system_fix="Reject placeholder/malformed values and retry with a repair prompt.",
    )


def _unsupported_label_issue(field: str, value: object) -> GateIssue:
    return GateIssue(
        field=field,
        message=(
            "Value does not reuse meaningful terminology from the assessment question or "
            f"selected source evidence: {str(value)[:120]}"
        ),
        operator_fix=(
            "Do not approve this output. The finding may be plausible, but it is not grounded "
            "in the selected evidence for this step."
        ),
        system_fix=(
            "Repair the answer so threats, vulnerabilities, and risks use terminology present "
            "in the selected source excerpts or assessment question."
        ),
    )


def _supported_values(values: list[str], support_text: str) -> list[str]:
    return [value for value in values if _has_term_support(value, support_text)]


def _has_term_support(value: object, support_text: str) -> bool:
    terms = [
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{3,}", str(value).lower())
        if term
        not in {
            "risk",
            "risks",
            "threat",
            "threats",
            "vulnerability",
            "vulnerabilities",
            "lack",
            "missing",
            "unknown",
            "high",
            "medium",
            "low",
            "with",
            "from",
            "during",
        }
    ]
    if not terms:
        return False
    return any(_term_supported(term, support_text) for term in terms)


def _any_meaningful_overlap(text: str, candidates: list[Any]) -> bool:
    for candidate in candidates:
        if _has_term_support(candidate, text):
            return True
        candidate_text = str(candidate or "").lower()
        if _looks_like_framework_label(candidate_text):
            tokens = [
                token
                for token in re.findall(r"[a-z0-9]+", candidate_text)
                if len(token) >= 2
            ]
            if tokens and all(token in text for token in tokens[:2]):
                return True
    return False


def _json_lower(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str).lower()


def _unsupported_urgency_terms(output_text: str, support_text: str) -> list[str]:
    return [
        term
        for term in ["critical", "immediate", "unacceptable", "severe"]
        if term in output_text and term not in support_text
    ]


def _term_supported(term: str, support_text: str) -> bool:
    if term in support_text:
        return True
    if term.endswith("ment") and term[:-4] in support_text:
        return True
    if term.endswith("ed") and term[:-2] in support_text:
        return True
    if term.endswith("ing") and term[:-3] in support_text:
        return True
    if term.endswith("s") and term[:-1] in support_text:
        return True
    return False


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


def _important_terms(values: list[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        lower = value.lower()
        for match in re.findall(
            r"\b(?:cis|scf|iso|nist|end-\d+(?:\.\d+)?|bcd-\d+(?:\.\d+)?)\b",
            lower,
        ):
            terms.add(match)
        for candidate in [
            "business continuity",
            "recovery",
            "resilience",
            "detective",
            "preventative",
            "response",
        ]:
            if candidate in lower:
                terms.add(candidate)
    return terms


def _framework_terms(values: list[str]) -> set[str]:
    terms: set[str] = set()
    for value in values:
        lower = value.lower()
        for match in re.findall(
            r"\b(?:cis|scf|iso|nist|end-\d+(?:\.\d+)?|bcd-\d+(?:\.\d+)?|"
            r"pr\.[a-z]{2}-\d+|\d+\.\d+)\b",
            lower,
        ):
            terms.add(match)
    return terms


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str))
    except TypeError:
        return len(str(value))


def _find_forbidden_debug_keys(value: Any, path: str = "") -> set[str]:
    forbidden = {
        "debug",
        "prompt_messages",
        "messages_sent_to_model",
        "raw_model_response",
    }
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            next_path = f"{path}.{key_text}" if path else key_text
            if key_text in forbidden:
                found.add(next_path)
            found.update(_find_forbidden_debug_keys(item, next_path))
    elif isinstance(value, list):
        for index, item in enumerate(value[:20]):
            found.update(_find_forbidden_debug_keys(item, f"{path}[{index}]"))
    return found


def _max_model_prompt_chars(value: Any) -> int:
    if isinstance(value, dict):
        own = 0
        prompt = value.get("model_prompt")
        if isinstance(prompt, dict):
            own = int(prompt.get("total_characters_sent_to_model") or 0)
        return max([own, *[_max_model_prompt_chars(item) for item in value.values()]])
    if isinstance(value, list):
        return max([0, *[_max_model_prompt_chars(item) for item in value[:20]]])
    return 0
