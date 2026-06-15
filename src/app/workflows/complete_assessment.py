from __future__ import annotations

import json
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
    FoundationSummaryDraft,
)
from app.assessment.token_estimator import MODEL_PRICES_PER_MILLION
from app.generation.clients import ChatModel, OpenAIChatClient
from app.pipeline import GraphRagPipeline
from app.schemas import GraphRagAnswer
from app.workflows.run_store import WorkflowRunStore
from secure_rag.llm import OllamaChatClient


class ModelSelection(BaseModel):
    provider: Literal["ollama", "openai"] = "ollama"
    model: str = "qwen3:14b"
    openai_api_key: str | None = Field(default=None, exclude=True)
    confirm_external_call: bool = False
    estimated_output_tokens: int = Field(default=1_200, ge=200, le=6_000)
    max_estimated_input_tokens: int = Field(default=24_000, ge=500, le=60_000)


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
    ) -> dict[str, Any]:
        _validate_model_selection(request.model)
        packet, input_source = _resolve_input_source(request)
        created_at = datetime.now(UTC).isoformat()
        run_id = f"run-{created_at.replace(':', '').replace('.', '')}-{uuid4().hex[:8]}"
        steps: list[WorkflowStep] = []
        selected_chat_model = _chat_model(
            request.model,
            self.pipeline.settings.ollama_base_url,
            self.pipeline.settings.openai_api_key,
            cancel_event,
        )
        original_pipeline_chat_model = self.pipeline.chat_model
        self.pipeline.chat_model = selected_chat_model

        def add_step(step: WorkflowStep) -> None:
            steps.append(step)
            if progress_callback is not None:
                progress_callback(step)

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
            for item in retrieval_input:
                _raise_if_cancelled(cancel_event)
                trace_label = f"{item['question_id']} / {item['context']['control']['control_id']}"
                answer, trace_steps = self.pipeline.query_with_trace(
                    item["retrieval_question"],
                    top_k=request.top_k,
                    debug=True,
                    trace_input={"risk_question_from_previous_step": item},
                    trace_label=trace_label,
                    model_label=f"{request.model.provider}:{request.model.model}",
                )
                for trace_step in trace_steps:
                    _raise_if_cancelled(cancel_event)
                    add_step(WorkflowStep(**trace_step))
                rag_answers.append(answer)
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
                    input={"risk_answer_outputs": retrieval_output},
                    process=(
                        "The previous steps produced one answer per weak control. Here we keep "
                        "the useful parts, trim oversized debug text, and prepare the evidence "
                        "package that will be sent to the report-writing model."
                    ),
                    output=rag_evidence_output,
                )
            )

            _raise_if_cancelled(cancel_event)
            cost_estimate = _estimate_complete_workflow_cost(
                sanitized,
                request.model,
                compact_risk_evidence,
            )
            paragraph_messages = _paragraph_prompt(
                sanitized,
                findings,
                compact_risk_evidence,
            )
            paragraph_input_tokens = _rough_tokens(
                json.dumps(paragraph_messages, ensure_ascii=True)
            )
            if paragraph_input_tokens > request.model.max_estimated_input_tokens:
                raise ValueError(
                    "Paragraph prompt exceeds max_estimated_input_tokens "
                    f"({paragraph_input_tokens} > {request.model.max_estimated_input_tokens})"
                )
            paragraph_input = {
                "evidence_package_from_previous_step": rag_evidence_output,
                "messages_sent_to_model": paragraph_messages,
                "api_key": "[hidden]",
            }
            _raise_if_cancelled(cancel_event)
            raw_paragraphs = selected_chat_model.chat(paragraph_messages)
            _raise_if_cancelled(cancel_event)
            paragraphs = _parse_paragraphs(raw_paragraphs, sanitized, findings)
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
                        "raw_model_output": raw_paragraphs,
                        "parsed_paragraphs": paragraphs,
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
                "assessment_id": packet.assessment_id,
                "vendor_id": packet.vendor.vendor_id,
                "provider": request.model.provider,
                "model": request.model.model,
                "cost_estimate": cost_estimate,
                "steps": [step.as_dict() for step in steps],
                "final_result": final_result,
            }
            path = self.run_store.save(run)
            run["run_path"] = str(path)
            return run
        finally:
            self.pipeline.chat_model = original_pipeline_chat_model


def estimate_complete_assessment_preflight(
    request: CompleteAssessmentRequest,
) -> dict[str, Any]:
    _validate_model_selection(request.model)
    packet, input_source = _resolve_input_source(request)
    sanitized = sanitize_packet(packet)
    findings = classify_findings(sanitized)
    retrieval_input = _risk_queries(sanitized, findings)
    packet_tokens = _rough_tokens(
        json.dumps(sanitized.model_dump(mode="json"), ensure_ascii=True)
    )
    finding_tokens = _rough_tokens(json.dumps(_findings_dump(findings), ensure_ascii=True))
    llm_call_count = max(1, len(retrieval_input)) + 1
    estimated_retrieval_context_tokens = len(retrieval_input) * request.top_k * 180
    estimated_input_tokens = packet_tokens + finding_tokens + estimated_retrieval_context_tokens
    estimated_output_tokens = request.model.estimated_output_tokens * llm_call_count
    price = MODEL_PRICES_PER_MILLION.get(request.model.model)
    estimated_cost = 0.0
    if request.model.provider == "openai" and price is not None:
        input_cost = estimated_input_tokens * price["input"] / 1_000_000
        output_cost = estimated_output_tokens * price["output"] / 1_000_000
        estimated_cost = round(input_cost + output_cost, 6)
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
        "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
        "max_estimated_input_tokens": request.model.max_estimated_input_tokens,
        "estimated_cost_usd": estimated_cost,
        "will_exceed_guard": estimated_input_tokens > request.model.max_estimated_input_tokens,
        "note": (
            "Preflight does not query Qdrant, BM25, Neo4j, Ollama, or OpenAI. "
            "It estimates the workflow before GPU/API work starts."
        ),
    }


def _validate_model_selection(model: ModelSelection) -> None:
    allowed_local = {"qwen3:14b", "gemma3:4b"}
    allowed_openai = {"gpt-5.4-mini", "gpt-5.4", "gpt-5.5", "gpt-4.1-mini"}
    if model.provider == "ollama" and model.model not in allowed_local:
        raise ValueError(f"Unsupported local model: {model.model}")
    if model.provider == "openai":
        if model.model not in allowed_openai:
            raise ValueError(f"Unsupported OpenAI model: {model.model}")
        if not model.confirm_external_call:
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
    packet: Any,
    findings: dict[str, list[AssessmentFinding]],
    rag_answers: list[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You draft business-facing TPRM assessment paragraphs. Use only the supplied "
                "assessment data and retrieved standards evidence. Do not invent citations, "
                "controls, certifications, or facts. Return valid JSON with exactly the "
                "requested keys."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": (
                        "Write only paragraph text. Do not decide schema, scores, "
                        "findings, or persistence."
                    ),
                    "required_keys": [
                        "management_summary",
                        "introduction",
                        "objective",
                        "risk_exposure",
                        "conclusion",
                    ],
                    "assessment": packet.model_dump(mode="json"),
                    "findings": _findings_dump(findings),
                    "retrieved_risk_evidence": rag_answers,
                },
                ensure_ascii=True,
                indent=2,
            ),
        },
    ]


def _estimate_complete_workflow_cost(
    packet: FoundationAssessmentPacket,
    model: ModelSelection,
    rag_answers: list[dict[str, Any]],
) -> dict[str, Any]:
    input_payload = json.dumps(
        {"assessment": packet.model_dump(mode="json"), "retrieved_evidence": rag_answers},
        ensure_ascii=True,
    )
    estimated_input_tokens = _rough_tokens(input_payload)
    llm_call_count = max(1, len(rag_answers)) + 1
    estimated_output_tokens = model.estimated_output_tokens * llm_call_count
    price = MODEL_PRICES_PER_MILLION.get(model.model)
    if model.provider == "openai" and price is not None:
        input_cost = estimated_input_tokens * price["input"] / 1_000_000
        output_cost = estimated_output_tokens * price["output"] / 1_000_000
        estimated_cost = round(input_cost + output_cost, 6)
        pricing_note = (
            "Estimated from configured OpenAI per-million prices for the full workflow."
        )
    else:
        estimated_cost = 0.0
        pricing_note = "Local Ollama workflow is estimated as $0 API cost."
    return {
        "model": model.model,
        "provider": model.provider,
        "llm_call_count": llm_call_count,
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens,
        "estimated_cost_usd": estimated_cost,
        "pricing_note": pricing_note,
    }


def _parse_paragraphs(
    raw: str,
    packet: Any,
    findings: dict[str, list[AssessmentFinding]],
) -> dict[str, str]:
    try:
        data = json.loads(raw[raw.find("{") : raw.rfind("}") + 1])
    except Exception:
        data = {}
    fallback = FoundationSummaryDraft(
        management_summary=(
            f"{packet.vendor.name} is a Tier {packet.tier.level} vendor with "
            f"{len(findings['weaknesses'])} control weaknesses requiring review."
        ),
        introduction=f"This assessment summarizes {packet.vendor.name}'s vendor risk posture.",
        objective="The objective is to support analyst review and business-owner decision making.",
        risk_exposure=(
            "Risk exposure is driven by weak questionnaire responses and retrieved evidence."
        ),
        conclusion="The result should be reviewed before snapshot approval.",
    ).model_dump()
    keys = ["management_summary", "introduction", "objective", "risk_exposure", "conclusion"]
    return {key: str(data.get(key) or fallback[key]) for key in keys}


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
    return answer.model_dump()


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


def _rough_tokens(text: str) -> int:
    return max(1, (len(text) + 2) // 3)
