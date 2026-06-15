from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.assessment.findings import classify_findings, sanitize_packet
from app.assessment.schemas import (
    AssessmentFinding,
    FoundationAssessmentPacket,
)
from app.assessment.token_estimator import MODEL_PRICES_PER_MILLION, USD_TO_EUR_RATE
from app.generation.clients import ChatModel, OpenAIChatClient
from app.pipeline import GraphRagPipeline
from app.quality_gates import (
    GateIssue,
    QualityGateFailure,
    failed_gate,
    human_failure_message,
    repair_prompt,
    validate_final_paragraphs,
    validate_prompt_contract,
    validate_workflow_step_contract,
    validate_workflow_step_payload,
)
from app.schemas import GraphRagAnswer
from app.workflows.run_store import WorkflowRunStore
from secure_rag.llm import OllamaChatClient


class ModelSelection(BaseModel):
    provider: Literal["ollama", "openai"] = "ollama"
    model: str = "qwen3:14b"
    openai_api_key: str | None = Field(default=None, exclude=True)
    confirm_external_call: bool = False
    estimated_output_tokens: int = Field(default=1_200, ge=200, le=6_000)
    max_estimated_input_tokens: int = Field(default=60_000, ge=500, le=120_000)
    enforce_token_budget: bool = True
    token_budget_tolerance_percent: int = Field(default=10, ge=0, le=100)


class AssessmentInputSource(BaseModel):
    adapter: Literal["foundation_packet_v1", "simulated_postgres_v1"] = (
        "foundation_packet_v1"
    )
    payload: dict[str, Any]


class CompleteAssessmentRequest(BaseModel):
    packet: FoundationAssessmentPacket | None = None
    input_source: AssessmentInputSource | None = None
    model: ModelSelection = Field(default_factory=ModelSelection)
    top_k: int = Field(default=10, ge=3, le=30)
    debug: bool = True


class WorkflowCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    explanation: str
    tool: str
    input: Any
    process: str
    output: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "explanation": self.explanation,
            "tool": self.tool,
            "input": self.input,
            "process": self.process,
            "output": self.output,
        }


@dataclass
class TokenUsageTracker:
    preflight_estimated_total_tokens: int
    tolerance_percent: int
    provider: str
    model: str
    enforced: bool = True
    calls: list[dict[str, Any]] | None = None
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    estimated_fallback_calls: int = 0

    @property
    def allowed_total_tokens(self) -> int:
        return math.ceil(
            self.preflight_estimated_total_tokens
            * (1 + (self.tolerance_percent / 100))
        )

    @property
    def actual_total_tokens(self) -> int:
        return self.actual_input_tokens + self.actual_output_tokens

    def assert_can_send(self, call_name: str, estimated_input_tokens: int) -> None:
        if not self.enforced:
            return
        predicted_total = self.actual_total_tokens + estimated_input_tokens
        if predicted_total > self.allowed_total_tokens:
            raise ValueError(
                "Token budget guard blocked a model call before sending it. "
                f"Call '{call_name}' would bring the run to about {predicted_total} "
                f"tokens, above the allowed {self.allowed_total_tokens}."
            )

    def record_call(
        self,
        *,
        call_name: str,
        messages: list[dict[str, str]],
        response_text: str,
        chat_model: object,
    ) -> dict[str, Any]:
        estimated_input = _rough_tokens(json.dumps(messages, ensure_ascii=True))
        estimated_output = _rough_tokens(response_text)
        usage = _model_usage(chat_model)
        actual_input = usage.get("input_tokens") if usage else estimated_input
        actual_output = usage.get("output_tokens") if usage else estimated_output
        if not isinstance(actual_input, int):
            actual_input = estimated_input
        if not isinstance(actual_output, int):
            actual_output = estimated_output
        if usage is None:
            self.estimated_fallback_calls += 1
        self.actual_input_tokens += actual_input
        self.actual_output_tokens += actual_output
        record = {
            "call_name": call_name,
            "provider": self.provider,
            "model": self.model,
            "estimated_input_tokens": estimated_input,
            "estimated_output_tokens": estimated_output,
            "actual_input_tokens": actual_input,
            "actual_output_tokens": actual_output,
            "actual_total_tokens": actual_input + actual_output,
            "source": "provider_reported" if usage is not None else "rough_estimate",
            "running_total_tokens": self.actual_total_tokens,
            "allowed_total_tokens": self.allowed_total_tokens,
            "within_budget_after_call": self.actual_total_tokens <= self.allowed_total_tokens,
        }
        if self.calls is None:
            self.calls = []
        self.calls.append(record)
        if self.enforced and self.actual_total_tokens > self.allowed_total_tokens:
            raise ValueError(
                "Token budget guard stopped the workflow after a model call. "
                f"Actual usage is {self.actual_total_tokens} tokens, above the "
                f"allowed {self.allowed_total_tokens}."
            )
        return record

    def as_dict(self) -> dict[str, Any]:
        difference = self.actual_total_tokens - self.preflight_estimated_total_tokens
        difference_percent = (
            round(difference / self.preflight_estimated_total_tokens * 100, 2)
            if self.preflight_estimated_total_tokens
            else 0
        )
        return {
            "provider": self.provider,
            "model": self.model,
            "enforced": self.enforced,
            "preflight_estimated_total_tokens": self.preflight_estimated_total_tokens,
            "tolerance_percent": self.tolerance_percent,
            "allowed_total_tokens": self.allowed_total_tokens,
            "actual_input_tokens": self.actual_input_tokens,
            "actual_output_tokens": self.actual_output_tokens,
            "actual_total_tokens": self.actual_total_tokens,
            "difference_tokens": difference,
            "difference_percent": difference_percent,
            "within_budget": self.actual_total_tokens <= self.allowed_total_tokens,
            "actual_cost_estimate": _token_cost_estimate(
                provider=self.provider,
                model=self.model,
                input_tokens=self.actual_input_tokens,
                output_tokens=self.actual_output_tokens,
                estimate_basis="actual_tokens_reported_or_measured_for_this_run",
            ),
            "usage_source_note": (
                "Provider-reported token counts are used when available. "
                "Calls without provider counts use the same conservative rough estimate "
                "as preflight."
            ),
            "estimated_fallback_calls": self.estimated_fallback_calls,
            "calls": self.calls or [],
        }


class CompleteAssessmentWorkflow:
    def __init__(
        self,
        *,
        pipeline: GraphRagPipeline,
        run_store: WorkflowRunStore,
    ) -> None:
        self.pipeline = pipeline
        self.run_store = run_store

    def run(
        self,
        request: CompleteAssessmentRequest,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[WorkflowStep], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        _validate_model_selection(request.model, require_external_confirmation=True)
        packet, input_source = _resolve_input_source(request)
        preflight = estimate_complete_assessment_preflight(request)
        created_at = datetime.now(UTC).isoformat()
        run_id = f"run-{created_at.replace(':', '').replace('.', '')}-{uuid4().hex[:8]}"
        steps: list[WorkflowStep] = []
        token_tracker = TokenUsageTracker(
            preflight_estimated_total_tokens=preflight["estimated_total_tokens"],
            tolerance_percent=request.model.token_budget_tolerance_percent,
            provider=request.model.provider,
            model=request.model.model,
            enforced=request.model.enforce_token_budget,
        )
        selected_chat_model = _chat_model(
            request.model,
            self.pipeline.settings.ollama_base_url,
            self.pipeline.settings.openai_api_key,
            cancel_event,
        )
        original_pipeline_chat_model = self.pipeline.chat_model
        self.pipeline.chat_model = selected_chat_model

        def add_step(step: WorkflowStep) -> None:
            payload_gate = validate_workflow_step_payload(
                step_name=step.name,
                input_payload=step.input,
                output_payload=step.output,
            )
            if not payload_gate.passed:
                failure_step = WorkflowStep(
                    name=f"Quality gate failed for workflow payload {step.name}",
                    explanation=(
                        "Stops because this workflow step produced data that is too large, "
                        "too technical, or polluted with debug details to pass forward safely."
                    ),
                    tool="Python workflow payload quality gate",
                    input={
                        "step_name": step.name,
                        "input_size": len(json.dumps(step.input, default=str)),
                        "output_size": len(json.dumps(step.output, default=str)),
                    },
                    process=(
                        "We check every step handoff before it becomes the next step's input. "
                        "The workflow must pass compact business facts and source summaries, "
                        "not full debug objects or oversized prompts."
                    ),
                    output=payload_gate.as_dict(),
                )
                steps.append(failure_step)
                if progress_callback is not None:
                    progress_callback(failure_step)
                raise QualityGateFailure(payload_gate)
            contract_gate = validate_workflow_step_contract(
                step_name=step.name,
                input_payload=step.input,
                output_payload=step.output,
            )
            if not contract_gate.passed:
                failure_step = WorkflowStep(
                    name=f"Quality gate failed for workflow step {step.name}",
                    explanation=(
                        "Stops because this workflow step did not produce the clear business "
                        "handoff that the next step needs."
                    ),
                    tool="Python workflow business contract gate",
                    input={
                        "step_name": step.name,
                        "input_size": len(json.dumps(step.input, default=str)),
                        "output_size": len(json.dumps(step.output, default=str)),
                    },
                    process=(
                        "We check whether the step output matches the step responsibility. "
                        "For example, retrieval must select compact evidence, prompt building "
                        "must stay focused, and model output must pass the expected structure."
                    ),
                    output=contract_gate.as_dict(),
                )
                steps.append(failure_step)
                if progress_callback is not None:
                    progress_callback(failure_step)
                raise QualityGateFailure(contract_gate)
            steps.append(step)
            if progress_callback is not None:
                progress_callback(step)

        def set_status(name: str) -> None:
            if status_callback is not None:
                status_callback(name)

        def record_model_call(
            call_name: str,
            messages: list[dict[str, str]],
            response_text: str,
            chat_model: object,
        ) -> dict[str, Any]:
            return token_tracker.record_call(
                call_name=call_name,
                messages=messages,
                response_text=response_text,
                chat_model=chat_model,
            )

        def assert_model_call_budget(
            call_name: str,
            messages: list[dict[str, str]],
        ) -> None:
            token_tracker.assert_can_send(
                call_name,
                _rough_tokens(json.dumps(messages, ensure_ascii=True)),
            )

        try:
            _raise_if_cancelled(cancel_event)
            sql_query = _simulated_sql(packet.assessment_id)
            add_step(
                WorkflowStep(
                    name="Read assessment input",
                    explanation=(
                        "Reads the vendor assessment data and converts it into the common "
                        "format used by the rest of the workflow."
                    ),
                    tool=f"{input_source.adapter} + PostgreSQL (simulated)",
                    input={
                        "adapter": input_source.adapter,
                        "sql": sql_query,
                        "payload": input_source.payload,
                    },
                    process=(
                        "We start with the simulated SQL result. This step keeps the database "
                        "shape separate from the workflow, so production PostgreSQL can later "
                        "send a different shape through another adapter."
                    ),
                    output={"normalized_packet": packet.model_dump(mode="json")},
                )
            )

            _raise_if_cancelled(cancel_event)
            normalized_output = {"normalized_packet": packet.model_dump(mode="json")}
            sanitized = sanitize_packet(packet)
            findings = classify_findings(sanitized)
            classified_output = {
                "sanitized_packet": sanitized.model_dump(mode="json"),
                "findings": _findings_dump(findings),
            }
            add_step(
                WorkflowStep(
                    name="Clean and sort questionnaire answers",
                    explanation=(
                        "Cleans the input text and sorts answers into strengths, weaknesses, "
                        "and answers that still need clarification."
                    ),
                    tool="Python deterministic workflow",
                    input=normalized_output,
                    process=(
                        "We take the normalized packet from the previous step, clean the free "
                        "text, and classify every questionnaire answer. Full compliance becomes "
                        "a strength. Partial or no compliance becomes a weakness to evaluate."
                    ),
                    output=classified_output,
                )
            )

            retrieval_input = _risk_queries(sanitized, findings)
            add_step(
                WorkflowStep(
                    name="Create risk questions from weak answers",
                    explanation=(
                        "Turns each weak questionnaire answer into a clear question that can "
                        "be searched in the standards library."
                    ),
                    tool="Python deterministic workflow",
                    input=classified_output,
                    process=(
                        "We use only the weaknesses from the previous step. For each one, we "
                        "build a search question that includes the vendor tier, the linked "
                        "control, the vendor response, and the analyst comment."
                    ),
                    output={"risk_questions": retrieval_input},
                )
            )
            rag_answers: list[GraphRagAnswer] = []
            risk_chain_output: dict[str, Any] = {"risk_questions": retrieval_input}
            for item_index, item in enumerate(retrieval_input):
                _raise_if_cancelled(cancel_event)
                trace_label = f"{item['question_id']} / {item['context']['control']['control_id']}"
                selection_output = {
                    "selected_risk_question": item,
                    "risk_answers_so_far": [_rag_answer_dump(answer) for answer in rag_answers],
                    "remaining_risk_question_count": len(retrieval_input) - item_index - 1,
                }
                add_step(
                    WorkflowStep(
                        name=f"Select next weak answer to evaluate for {trace_label}",
                        explanation=(
                            "Chooses one weak questionnaire answer from the queue so it can be "
                            "checked against the standards library."
                        ),
                        tool="Python deterministic workflow",
                        input=risk_chain_output,
                        process=(
                            "The previous step output is the current work package. This step "
                            "moves the workflow cursor to the next weak answer while keeping "
                            "the risk answers already produced."
                        ),
                        output=selection_output,
                    )
                )
                risk_chain_output = selection_output
                answer, trace_steps = self.pipeline.query_with_trace(
                    item["retrieval_question"],
                    top_k=request.top_k,
                    debug=True,
                    trace_input=selection_output,
                    trace_label=trace_label,
                    model_label=f"{request.model.provider}:{request.model.model}",
                    token_usage_callback=record_model_call,
                    token_budget_guard=assert_model_call_budget,
                    status_callback=set_status,
                    trace_step_callback=lambda trace_step: add_step(
                        WorkflowStep(**trace_step)
                    ),
                )
                for trace_step in trace_steps:
                    _raise_if_cancelled(cancel_event)
                    risk_chain_output = trace_step["output"]
                rag_answers.append(answer)
                stored_answer_output = {
                    "latest_risk_answer": _rag_answer_dump(answer),
                    "risk_answers_so_far": [
                        _rag_answer_dump(rag_answer) for rag_answer in rag_answers
                    ],
                    "remaining_risk_question_count": len(retrieval_input) - item_index - 1,
                }
                add_step(
                    WorkflowStep(
                        name=f"Store risk answer for {trace_label}",
                        explanation=(
                            "Keeps the risk answer so later steps can draft the final report "
                            "from all evaluated weak answers."
                        ),
                        tool="Python deterministic workflow",
                        input=risk_chain_output,
                        process=(
                            "We take the model answer from the previous step, store it in the "
                            "workflow result list, and pass that accumulated list forward."
                        ),
                        output=stored_answer_output,
                    )
                )
                risk_chain_output = stored_answer_output
            retrieval_output = [_rag_answer_dump(answer) for answer in rag_answers]
            compact_risk_evidence = _compact_rag_evidence(retrieval_output)
            rag_evidence_output = {"retrieved_risk_evidence": compact_risk_evidence}
            add_step(
                WorkflowStep(
                    name="Collect risk answers for report drafting",
                    explanation=(
                        "Collects the risk answers from all weak questionnaire answers so the "
                        "report-writing step has one clean evidence package."
                    ),
                    tool="Python deterministic workflow",
                    input=risk_chain_output,
                    process=(
                        "The previous steps produced one answer per weak control. Here we keep "
                        "the useful parts, trim oversized debug text, and prepare the evidence "
                        "package that will be sent to the report-writing model."
                    ),
                    output=rag_evidence_output,
                )
            )

            _raise_if_cancelled(cancel_event)
            validated_fact_packet = _validated_fact_packet(
                sanitized,
                findings,
                compact_risk_evidence,
            )
            add_step(
                WorkflowStep(
                    name="Build validated fact packet for report drafting",
                    explanation=(
                        "Creates the clean facts that the final report writer is allowed to use."
                    ),
                    tool="Python deterministic workflow + quality gate",
                    input=rag_evidence_output,
                    process=(
                        "We remove raw debug material and keep only vendor context, tier context, "
                        "weaknesses, validated risk facts, controls, source mappings, and missing "
                        "information. The final model may phrase these facts but may not invent "
                        "new facts."
                    ),
                    output=validated_fact_packet,
                )
            )

            _raise_if_cancelled(cancel_event)
            cost_estimate = _cost_estimate_from_preflight(request.model, preflight)
            paragraph_messages = _paragraph_prompt(
                validated_fact_packet,
            )
            paragraph_prompt_gate = validate_prompt_contract(
                gate="final_paragraph_prompt",
                messages=paragraph_messages,
                required_phrases=[
                    "Role:",
                    "Objective:",
                    "Trusted fact boundary:",
                    "Forbidden behavior:",
                    "Output contract:",
                    "Do not repair JSON",
                ],
                task_name="final paragraph",
            )
            if not paragraph_prompt_gate.passed:
                add_step(
                    WorkflowStep(
                        name="Quality gate failed for final paragraph prompt",
                        explanation=(
                            "Stops before calling the report model because the prompt is "
                            "incomplete."
                        ),
                        tool="Python quality gate",
                        input={"model_prompt": _compact_messages(paragraph_messages)},
                        process=(
                            "We check that the prompt explains the model role, task, trusted "
                            "fact boundary, forbidden behavior, and output contract."
                        ),
                        output=paragraph_prompt_gate.as_dict(),
                    )
                )
                raise QualityGateFailure(paragraph_prompt_gate)
            paragraph_input_tokens = _rough_tokens(
                json.dumps(paragraph_messages, ensure_ascii=True)
            )
            if paragraph_input_tokens > request.model.max_estimated_input_tokens:
                raise ValueError(
                    "Paragraph prompt exceeds max_estimated_input_tokens "
                    f"({paragraph_input_tokens} > {request.model.max_estimated_input_tokens})"
                )
            token_tracker.assert_can_send(
                "Report paragraph model call",
                paragraph_input_tokens,
            )
            paragraph_input = {
                "validated_fact_packet_from_previous_step": validated_fact_packet,
                "model_prompt": _compact_messages(paragraph_messages),
                "quality_gate": paragraph_prompt_gate.as_dict(),
                "api_key": "[hidden]",
            }
            _raise_if_cancelled(cancel_event)
            paragraph_attempts: list[dict[str, Any]] = []
            paragraph_gate = None
            paragraph_token_usage = None
            paragraphs: dict[str, str] | None = None
            raw_paragraphs = ""
            current_paragraph_messages = paragraph_messages
            for attempt in range(1, 4):
                call_name = (
                    "Report paragraph model call"
                    if attempt == 1
                    else f"Report paragraph model call repair attempt {attempt}"
                )
                token_tracker.assert_can_send(
                    call_name,
                    _rough_tokens(json.dumps(current_paragraph_messages, ensure_ascii=True)),
                )
                set_status(f"Waiting for model to draft report paragraphs (attempt {attempt})")
                raw_paragraphs = selected_chat_model.chat(current_paragraph_messages)
                set_status(f"Validating model-drafted report paragraphs (attempt {attempt})")
                paragraph_token_usage = record_model_call(
                    call_name,
                    current_paragraph_messages,
                    raw_paragraphs,
                    selected_chat_model,
                )
                try:
                    paragraphs = _parse_paragraphs_strict(raw_paragraphs)
                    paragraph_gate = validate_final_paragraphs(
                        paragraphs=paragraphs,
                        raw_model_output=raw_paragraphs,
                        vendor_name=sanitized.vendor.name,
                        tier_level=sanitized.tier.level,
                        weakness_summaries=[
                            finding.summary for finding in findings["weaknesses"]
                        ],
                        validated_facts=validated_fact_packet,
                    )
                except Exception as exc:
                    paragraph_gate = failed_gate(
                        "final_paragraph_output",
                        "Final report paragraphs could not be parsed as the required JSON.",
                        [
                            GateIssue(
                                field="raw_model_output",
                                message=str(exc),
                                operator_fix=(
                                    "Do not approve this report. The model did not return the "
                                    "required paragraph contract."
                                ),
                                system_fix=(
                                    "Retry with the repair prompt, reduce prompt size, or use a "
                                    "stronger model."
                                ),
                            )
                        ],
                    )
                paragraph_attempts.append(
                    {
                        "attempt": attempt,
                        "raw_model_output_preview": _preview_text(raw_paragraphs, 1_500),
                        "parsed_paragraphs": paragraphs,
                        "token_usage": paragraph_token_usage,
                        "quality_gate": paragraph_gate.as_dict(),
                    }
                )
                if paragraph_gate.passed and paragraphs is not None:
                    break
                if attempt < 3:
                    current_paragraph_messages = repair_prompt(
                        original_messages=paragraph_messages,
                        gate_result=paragraph_gate,
                    )
            _raise_if_cancelled(cancel_event)
            if paragraph_gate is None or not paragraph_gate.passed or paragraphs is None:
                add_step(
                    WorkflowStep(
                        name="Quality gate failed for final report paragraphs",
                        explanation=(
                            "Stops because the report model did not produce trustworthy "
                            "business paragraphs after repair retries."
                        ),
                        tool="Python quality gate",
                        input=paragraph_input,
                        process=(
                            "We validate the final report text for required fields, vendor/tier "
                            "context, validated risk facts, and forbidden JSON-repair behavior. "
                            "The workflow stops instead of showing generic fallback text."
                        ),
                        output={
                            "attempts": paragraph_attempts,
                            "quality_gate": paragraph_gate.as_dict()
                            if paragraph_gate
                            else None,
                            "human_explanation": human_failure_message(paragraph_gate)
                            if paragraph_gate
                            else "The quality gate did not produce a result.",
                        },
                    )
                )
                raise QualityGateFailure(paragraph_gate) if paragraph_gate else RuntimeError(
                    "Final paragraph quality gate failed without a gate result."
                )
            add_step(
                WorkflowStep(
                    name="Ask model to draft report paragraphs",
                    explanation=(
                        "Asks the selected model to turn the assessment data and retrieved "
                        "evidence into readable report text."
                    ),
                    tool=f"{request.model.provider}:{request.model.model}",
                    input=paragraph_input,
                    process=(
                        "We send the evidence package from the previous step plus the cleaned "
                        "assessment data. The model is instructed to write only the requested "
                        "paragraphs and not decide scores, schema, or database fields."
                    ),
                    output={
                        "raw_model_output_preview": _preview_text(raw_paragraphs, 1_500),
                        "parsed_paragraphs": paragraphs,
                        "attempts": paragraph_attempts,
                        "quality_gate": paragraph_gate.as_dict(),
                        "token_usage": paragraph_token_usage,
                    },
                )
            )

            _raise_if_cancelled(cancel_event)
            final_result = _final_result(
                packet,
                paragraphs,
                findings,
                compact_risk_evidence,
            )
            add_step(
                WorkflowStep(
                    name="Prepare final result for the application",
                    explanation=(
                        "Packages the paragraphs, strengths, weaknesses, and risk answers into "
                        "the structured result your application can store later."
                    ),
                    tool="Python deterministic workflow",
                    input={
                        "paragraphs_from_previous_step": paragraphs,
                        "classified_findings": _findings_dump(findings),
                        "risk_evidence_package": compact_risk_evidence,
                    },
                    process=(
                        "This step is plain Python. It takes the model-written paragraphs from "
                        "the previous step and places them into the fixed output contract. The "
                        "AI model does not decide the database shape."
                    ),
                    output=final_result,
                )
            )

            run = {
                "run_id": run_id,
                "created_at": created_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "assessment_id": packet.assessment_id,
                "vendor_id": packet.vendor.vendor_id,
                "provider": request.model.provider,
                "model": request.model.model,
                "cost_estimate": cost_estimate,
                "preflight": preflight,
                "token_budget": token_tracker.as_dict(),
                "steps": [step.as_dict() for step in steps],
                "final_result": final_result,
            }
            run["duration_seconds"] = round(
                (
                    datetime.fromisoformat(run["completed_at"])
                    - datetime.fromisoformat(created_at)
                ).total_seconds(),
                3,
            )
            path = self.run_store.save(run)
            run["run_path"] = str(path)
            return run
        finally:
            self.pipeline.chat_model = original_pipeline_chat_model


def estimate_complete_assessment_preflight(
    request: CompleteAssessmentRequest,
) -> dict[str, Any]:
    _validate_model_selection(request.model, require_external_confirmation=False)
    packet, input_source = _resolve_input_source(request)
    sanitized = sanitize_packet(packet)
    findings = classify_findings(sanitized)
    retrieval_input = _risk_queries(sanitized, findings)
    packet_tokens = _rough_tokens(
        json.dumps(sanitized.model_dump(mode="json"), ensure_ascii=True)
    )
    finding_tokens = _rough_tokens(json.dumps(_findings_dump(findings), ensure_ascii=True))
    risk_question_count = len(retrieval_input)
    llm_call_count = max(1, risk_question_count) + 1
    input_breakdown = _conservative_preflight_input_breakdown(
        packet_tokens=packet_tokens,
        finding_tokens=finding_tokens,
        risk_question_count=risk_question_count,
        top_k=request.top_k,
    )
    estimated_input_tokens = input_breakdown["estimated_input_tokens"]
    estimated_output_tokens = request.model.estimated_output_tokens * llm_call_count
    estimated_total_tokens = estimated_input_tokens + estimated_output_tokens
    allowed_total_tokens = math.ceil(
        estimated_total_tokens
        * (1 + (request.model.token_budget_tolerance_percent / 100))
    )
    price = MODEL_PRICES_PER_MILLION.get(request.model.model)
    cost_estimate = _token_cost_estimate(
        provider=request.model.provider,
        model=request.model.model,
        input_tokens=estimated_input_tokens,
        output_tokens=estimated_output_tokens,
        estimate_basis="conservative_preflight_for_one_complete_workflow_run",
    )
    return {
        "adapter": input_source.adapter,
        "assessment_id": packet.assessment_id,
        "vendor_id": packet.vendor.vendor_id,
        "provider": request.model.provider,
        "model": request.model.model,
        "weakness_count": len(findings["weaknesses"]),
        "retrieval_query_count": len(retrieval_input),
        "top_k": request.top_k,
        "llm_call_count": llm_call_count,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_tokens": estimated_total_tokens,
        "estimate_policy": "conservative_workflow_reserve",
        "estimate_breakdown": input_breakdown,
        "max_estimated_input_tokens": request.model.max_estimated_input_tokens,
        "enforce_token_budget": request.model.enforce_token_budget,
        "token_budget_tolerance_percent": request.model.token_budget_tolerance_percent,
        "allowed_total_tokens": allowed_total_tokens,
        "estimated_cost_usd": cost_estimate["estimated_cost_usd"],
        "estimated_cost_eur": cost_estimate["estimated_cost_eur"],
        "usd_to_eur_rate": cost_estimate["usd_to_eur_rate"],
        "price_per_million_tokens": price if request.model.provider == "openai" else None,
        "pricing_note": cost_estimate["pricing_note"],
        "will_exceed_guard": estimated_input_tokens > request.model.max_estimated_input_tokens,
        "note": (
            "Preflight does not query Qdrant, BM25, Neo4j, Ollama, or OpenAI. "
            "It reserves conservatively for all planned model calls before GPU/API work starts."
        ),
    }


def _validate_model_selection(
    model: ModelSelection,
    *,
    require_external_confirmation: bool = True,
) -> None:
    allowed_local = {"qwen3:14b", "gemma3:4b"}
    allowed_openai = {"gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-4.1-mini"}
    if model.provider == "ollama" and model.model not in allowed_local:
        raise ValueError(f"Unsupported local model: {model.model}")
    if model.provider == "openai":
        if model.model not in allowed_openai:
            raise ValueError(f"Unsupported OpenAI model: {model.model}")
        if require_external_confirmation and not model.confirm_external_call:
            raise ValueError("OpenAI calls require confirm_external_call=true")


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise WorkflowCancelled("Workflow job was cancelled")


def _resolve_input_source(
    request: CompleteAssessmentRequest,
) -> tuple[FoundationAssessmentPacket, AssessmentInputSource]:
    if request.input_source is None and request.packet is None:
        raise ValueError("Request must include either packet or input_source")
    if request.input_source is None:
        assert request.packet is not None
        source = AssessmentInputSource(
            adapter="foundation_packet_v1",
            payload=request.packet.model_dump(mode="json"),
        )
        return request.packet, source
    source = request.input_source
    payload = source.payload
    if source.adapter == "foundation_packet_v1":
        return FoundationAssessmentPacket.model_validate(payload), source
    if source.adapter == "simulated_postgres_v1":
        candidate = payload.get("packet") or payload.get("row") or payload
        rows = payload.get("rows")
        if rows:
            candidate = rows[0]
        return FoundationAssessmentPacket.model_validate(candidate), source
    raise ValueError(f"Unsupported input adapter: {source.adapter}")


def _chat_model(
    model: ModelSelection,
    ollama_base_url: str,
    configured_openai_api_key: str | None,
    cancel_event: threading.Event | None = None,
) -> ChatModel:
    if model.provider == "openai":
        return OpenAIChatClient(
            api_key=model.openai_api_key or configured_openai_api_key,
            model=model.model,
            max_output_tokens=model.estimated_output_tokens,
        )
    return OllamaChatClient(
        model=model.model,
        base_url=ollama_base_url,
        max_output_tokens=model.estimated_output_tokens,
        keep_alive="0s",
        cancel_event=cancel_event,
    )


def _risk_queries(
    packet: Any,
    findings: dict[str, list[AssessmentFinding]],
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    result_by_id = {result.question_id: result for result in packet.questionnaire_results}
    for finding in findings["weaknesses"]:
        result = result_by_id.get(finding.question_id)
        question_text = result.question_text if result else ""
        vendor_comment = result.sanitized_vendor_comment if result else ""
        analyst_comment = result.sanitized_analyst_comment if result else ""
        queries.append(
            {
                "question_id": finding.question_id,
                "question": (
                    f"Vendor {packet.vendor.name} is Tier {packet.tier.level}. "
                    f"Control {finding.control.framework} {finding.control.control_id} "
                    f"({finding.control.title}) is weak: {finding.summary}. "
                    "Using standards evidence, identify threats, vulnerabilities, risks, "
                    "preventative, detective, corrective, recovery, and response "
                    "controls, plus resilience impact."
                ),
                "retrieval_question": (
                    f"{finding.control.title}. {question_text}. "
                    f"{vendor_comment} {analyst_comment} {finding.summary}. "
                    "Identify threats, vulnerabilities, risks, "
                    "and security controls from standards evidence."
                ),
                "context": {
                    "tier": packet.tier.model_dump(mode="json"),
                    "vendor_type": packet.vendor.vendor_type,
                    "question_id": finding.question_id,
                    "control": finding.control.model_dump(mode="json"),
                    "compliance": finding.compliance.value,
                    "maturity": finding.maturity.value,
                },
            }
        )
    return queries


def _paragraph_prompt(
    validated_fact_packet: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Role:\n"
                "You are a business-facing TPRM report writer.\n\n"
                "Objective:\n"
                "Turn the validated fact packet into concise report paragraphs for a "
                "business owner.\n\n"
                "Trusted fact boundary:\n"
                "Use only the supplied validated fact packet. The risk analysis has already been "
                "performed and validated. Do not add new risks, controls, citations, evidence, "
                "certifications, or assumptions.\n\n"
                "Forbidden behavior:\n"
                "- Do not repair JSON.\n"
                "- Do not critique the input.\n"
                "- Do not re-analyze the standards evidence.\n"
                "- Do not include markdown.\n"
                "- Do not invent facts outside the validated fact packet.\n\n"
                "- Do not mention acceptable risk, thresholds, approval, or rejection unless "
                "the validated fact packet explicitly contains that decision basis.\n\n"
                "Output contract:\n"
                "Return only valid JSON with exactly these keys: management_summary, "
                "introduction, objective, risk_exposure, conclusion.\n\n"
                "Output discipline:\n"
                "Use controlled prose. Each paragraph must be 2-4 sentences. Put the most "
                "important finding first. Avoid filler, adjectives, methodology explanations, "
                "and long background education."
            ),
        },
        {
            "role": "user",
            "content": (
                "Task:\n"
                "Write only the five requested paragraph values. Each paragraph must be "
                "business-readable, concise, and grounded in the validated facts.\n\n"
                "Required output JSON:\n"
                "{\n"
                '  "management_summary": "",\n'
                '  "introduction": "",\n'
                '  "objective": "",\n'
                '  "risk_exposure": "",\n'
                '  "conclusion": ""\n'
                "}\n\n"
                "Validated fact packet:\n"
                f"{json.dumps(validated_fact_packet, ensure_ascii=True, indent=2)}"
            ),
        },
    ]


def _cost_estimate_from_preflight(
    model: ModelSelection,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    cost = _token_cost_estimate(
        provider=model.provider,
        model=model.model,
        input_tokens=int(preflight["estimated_input_tokens"]),
        output_tokens=int(preflight["estimated_output_tokens"]),
        estimate_basis="conservative_preflight_for_one_complete_workflow_run",
    )
    return {
        "model": model.model,
        "provider": model.provider,
        "llm_call_count": preflight["llm_call_count"],
        "estimated_input_tokens": preflight["estimated_input_tokens"],
        "estimated_output_tokens": preflight["estimated_output_tokens"],
        "estimated_total_tokens": preflight["estimated_total_tokens"],
        "estimate_policy": preflight["estimate_policy"],
        "estimate_breakdown": preflight["estimate_breakdown"],
        **cost,
    }


def _token_cost_estimate(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    estimate_basis: str,
) -> dict[str, Any]:
    price = MODEL_PRICES_PER_MILLION.get(model)
    if provider != "openai" or price is None:
        return {
            "estimated_cost_usd": 0.0,
            "estimated_cost_eur": 0.0,
            "usd_to_eur_rate": USD_TO_EUR_RATE,
            "estimate_basis": estimate_basis,
            "pricing_note": (
                "Internal/local model run: token use is tracked, but API price is $0."
            ),
        }
    input_cost = input_tokens * price["input"] / 1_000_000
    output_cost = output_tokens * price["output"] / 1_000_000
    estimated_cost_usd = round(input_cost + output_cost, 6)
    return {
        "estimated_cost_usd": estimated_cost_usd,
        "estimated_cost_eur": round(estimated_cost_usd * USD_TO_EUR_RATE, 6),
        "usd_to_eur_rate": USD_TO_EUR_RATE,
        "estimate_basis": estimate_basis,
        "pricing_note": (
            "External OpenAI estimate from configured per-million input/output token prices. "
            "EUR uses the configured USD_TO_EUR_RATE."
        ),
    }


def _conservative_preflight_input_breakdown(
    *,
    packet_tokens: int,
    finding_tokens: int,
    risk_question_count: int,
    top_k: int,
) -> dict[str, int | float]:
    risk_call_count = max(1, risk_question_count)
    prompt_instruction_reserve = risk_call_count * 1_600
    retrieved_chunk_reserve = risk_call_count * top_k * 900
    graph_context_reserve = risk_call_count * 1_000
    final_report_prompt_reserve = packet_tokens + finding_tokens + 1_500
    final_report_evidence_reserve = risk_question_count * 2_500
    subtotal = (
        packet_tokens
        + finding_tokens
        + prompt_instruction_reserve
        + retrieved_chunk_reserve
        + graph_context_reserve
        + final_report_prompt_reserve
        + final_report_evidence_reserve
    )
    safety_multiplier = 1.35
    estimated_input_tokens = math.ceil(subtotal * safety_multiplier)
    return {
        "packet_tokens": packet_tokens,
        "finding_tokens": finding_tokens,
        "risk_call_count": risk_call_count,
        "top_k": top_k,
        "prompt_instruction_reserve": prompt_instruction_reserve,
        "retrieved_chunk_reserve": retrieved_chunk_reserve,
        "graph_context_reserve": graph_context_reserve,
        "final_report_prompt_reserve": final_report_prompt_reserve,
        "final_report_evidence_reserve": final_report_evidence_reserve,
        "subtotal_before_safety_margin": subtotal,
        "safety_multiplier": safety_multiplier,
        "estimated_input_tokens": estimated_input_tokens,
    }


def _parse_paragraphs_strict(raw: str) -> dict[str, str]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Model output did not contain a JSON object.")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Model output JSON is not an object.")
    keys = ["management_summary", "introduction", "objective", "risk_exposure", "conclusion"]
    missing = [key for key in keys if not isinstance(data.get(key), str) or not data[key].strip()]
    if missing:
        raise ValueError(f"Model output is missing required paragraph fields: {missing}")
    return {key: str(data[key]).strip() for key in keys}


def _validated_fact_packet(
    packet: FoundationAssessmentPacket,
    findings: dict[str, list[AssessmentFinding]],
    rag_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    validated_risk_facts: list[dict[str, Any]] = []
    for answer in rag_answers:
        risk_answer = answer.get("answer") or {}
        validated_risk_facts.append(
            {
                "executive_summary": risk_answer.get("executive_summary"),
                "threats": risk_answer.get("threats") or [],
                "vulnerabilities": risk_answer.get("vulnerabilities") or [],
                "risks": risk_answer.get("risks") or [],
                "recommended_controls": risk_answer.get("recommended_controls") or [],
                "risk_control_matrix": risk_answer.get("risk_control_matrix") or [],
                "missing_information": risk_answer.get("missing_information") or [],
                "sources": answer.get("sources") or [],
                "insufficient_evidence": answer.get("insufficient_evidence", False),
            }
        )
    return {
        "vendor": {
            "vendor_id": packet.vendor.vendor_id,
            "name": packet.vendor.name,
            "vendor_type": packet.vendor.vendor_type,
            "business_relationship": packet.vendor.business_relationship,
            "services": packet.vendor.services,
        },
        "tier": packet.tier.model_dump(mode="json"),
        "strengths": [finding.model_dump(mode="json") for finding in findings["strengths"]],
        "weaknesses": [finding.model_dump(mode="json") for finding in findings["weaknesses"]],
        "validated_risk_facts": validated_risk_facts,
        "reporting_rules": {
            "do_not_invent_new_facts": True,
            "draft_status": "analyst_review_required",
            "source": "deterministic_packet_from_validated_workflow_steps",
        },
    }


def _final_result(
    packet: FoundationAssessmentPacket,
    paragraphs: dict[str, str],
    findings: dict[str, list[AssessmentFinding]],
    rag_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "assessment_id": packet.assessment_id,
        "vendor_id": packet.vendor.vendor_id,
        "draft_sections": {
            **paragraphs,
            "strengths": [finding.summary for finding in findings["strengths"]],
            "weaknesses": [finding.summary for finding in findings["weaknesses"]],
            "key_findings": [
                *(finding.summary for finding in findings["strengths"][:3]),
                *(finding.summary for finding in findings["weaknesses"][:5]),
            ],
            "missing_information": [
                f"Missing evidence for {finding.question_id}"
                for finding in findings["weaknesses"]
                if not finding.evidence_ids
            ],
        },
        "risk_evaluations": rag_answers,
        "snapshot_ready": False,
        "source_question_ids": [result.question_id for result in packet.questionnaire_results],
    }


def _rag_answer_dump(answer: GraphRagAnswer) -> dict[str, Any]:
    data = answer.model_dump()
    return {
        "answer": _compact_risk_answer(data.get("answer") or {}),
        "insufficient_evidence": data.get("insufficient_evidence", False),
        "sources": [_compact_source(source) for source in (data.get("sources") or [])[:8]],
    }


def _compact_messages(messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "message_count": len(messages),
        "total_characters_sent_to_model": sum(
            len(message.get("content", "")) for message in messages
        ),
        "messages": [
            {
                "role": message.get("role"),
                "characters": len(message.get("content", "")),
                "preview": _preview_text(message.get("content", ""), 2_000),
            }
            for message in messages
        ],
    }


def _preview_text(value: object, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} [...]"


def _compact_rag_evidence(rag_answers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for answer in rag_answers:
        risk_answer = _compact_risk_answer(answer.get("answer") or {})
        sources = answer.get("sources") or []
        retrieved = (answer.get("debug") or {}).get("retrieved_chunks") or []
        compact.append(
            {
                "answer": risk_answer,
                "insufficient_evidence": answer.get("insufficient_evidence", False),
                "sources": [_compact_source(source) for source in sources[:5]],
                "retrieved_chunks": [
                    {
                        "score": item.get("score"),
                        "source": item.get("source"),
                        "retrieval_method": item.get("retrieval_method"),
                        "metadata": _compact_metadata(
                            (item.get("chunk") or {}).get("metadata", {})
                        ),
                        "text": ((item.get("chunk") or {}).get("text") or "")[:350],
                    }
                    for item in retrieved[:5]
                ],
            }
        )
    return compact


def _compact_risk_answer(answer: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "executive_summary",
        "assumptions",
        "threats",
        "vulnerabilities",
        "risks",
        "recommended_controls",
        "risk_control_matrix",
        "missing_information",
        "from_retrieved_evidence",
        "general_model_reasoning",
    ]
    compact = {key: answer.get(key) for key in keys if answer.get(key)}
    if "risk_control_matrix" in compact:
        compact["risk_control_matrix"] = compact["risk_control_matrix"][:5]
    return compact


def _compact_source(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": source.get("id"),
        "source": source.get("source"),
        "score": source.get("score"),
        "retrieval_method": source.get("retrieval_method"),
        "metadata": _compact_metadata(source.get("metadata", {})),
    }


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata.get(key)
        for key in [
            "filename",
            "document_type",
            "page_or_section",
            "framework",
            "control_id",
            "source_path",
        ]
        if metadata.get(key) is not None
    }


def _findings_dump(findings: dict[str, list[AssessmentFinding]]) -> dict[str, list[dict[str, Any]]]:
    return {
        key: [finding.model_dump(mode="json") for finding in value]
        for key, value in findings.items()
    }


def _simulated_sql(assessment_id: str) -> str:
    safe_assessment_id = assessment_id.replace("'", "''")
    return (
        "SELECT vendor, tier_attributes, questionnaire_results, controls, "
        "comments, evidence_descriptions\n"
        "FROM tprm_assessment_view\n"
        f"WHERE assessment_id = '{safe_assessment_id}';"
    )


def _model_usage(chat_model: object) -> dict[str, Any] | None:
    usage = getattr(chat_model, "last_usage", None)
    if not isinstance(usage, dict):
        return None
    return usage


def _rough_tokens(text: str) -> int:
    return max(1, (len(text) + 2) // 3)
