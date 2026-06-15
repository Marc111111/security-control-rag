from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.assessment.schemas import FoundationAssessmentPacket, FoundationSummaryResponse
from app.assessment.workflow import FoundationAssessmentWorkflow
from app.config import Settings
from app.evaluation.logger import EvaluationLogger
from app.generation.clients import ChatModel, chat_client_for_provider
from app.generation.prompts import build_structured_answer_prompt
from app.generation.structured import (
    insufficient_evidence_answer,
    parse_structured_answer_strict,
)
from app.graph.extractor import HeuristicGraphExtractor
from app.graph.store import GraphStore, MemoryGraphStore, Neo4jGraphStore
from app.ingestion.chunker import load_and_chunk_path
from app.planning.planner import RiskQuestionPlanner
from app.quality_gates import (
    GateIssue,
    QualityGateFailure,
    combine_gate_results,
    failed_gate,
    human_failure_message,
    prune_unsupported_risk_answer,
    repair_prompt,
    validate_prompt_contract,
    validate_prompt_quality,
    validate_risk_answer,
)
from app.retrieval.bm25 import KeywordIndex
from app.retrieval.evidence_packet import prompt_evidence_summary, select_prompt_evidence
from app.retrieval.graph_context import build_graph_context_rows
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector_store import DenseStore, MemoryDenseStore, QdrantDenseStore
from app.schemas import GraphRagAnswer
from secure_rag.embeddings import EmbeddingClient, OllamaEmbeddingClient


class GraphRagPipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        embedding_client: EmbeddingClient | None = None,
        dense_store: DenseStore | None = None,
        graph_store: GraphStore | None = None,
        chat_model: ChatModel | None = None,
        logger: EvaluationLogger | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client or OllamaEmbeddingClient(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
        self.dense_store = dense_store
        self.keyword_index: KeywordIndex | None = None
        self.graph_store = graph_store
        self.chat_model = chat_model or chat_client_for_provider(
            provider=settings.llm_provider,
            ollama_base_url=settings.ollama_base_url,
            ollama_model=settings.generation_model,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )
        self.extractor = HeuristicGraphExtractor()
        self.foundation_workflow = FoundationAssessmentWorkflow(self.chat_model)
        self.planner = RiskQuestionPlanner()
        self.retriever: HybridRetriever | None = None
        self.logger = logger or EvaluationLogger()

    def ingest(
        self,
        source: str | Path,
        *,
        chunk_size: int = 1_500,
        overlap: int = 200,
        batch_size: int = 64,
    ) -> dict[str, int]:
        retriever = self._retriever()
        chunks = load_and_chunk_path(source, chunk_size=chunk_size, overlap=overlap)
        total_entities = 0
        total_relationships = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedding_client.embed([chunk.text for chunk in batch])
            retriever.add_chunks(batch, embeddings)
            for chunk in batch:
                extraction = self.extractor.extract(chunk)
                self._graph_store_instance().upsert(
                    extraction.entities,
                    extraction.relationships,
                )
                total_entities += len(extraction.entities)
                total_relationships += len(extraction.relationships)
        return {
            "indexed_chunks": len(chunks),
            "graph_entities": total_entities,
            "graph_relationships": total_relationships,
        }

    def query(
        self,
        question: str,
        *,
        context: dict[str, Any] | None = None,
        top_k: int | None = None,
        debug: bool | None = None,
    ) -> GraphRagAnswer:
        answer, _steps = self.query_with_trace(
            question,
            context=context,
            top_k=top_k,
            debug=debug,
        )
        return answer

    def query_with_trace(
        self,
        question: str,
        *,
        context: dict[str, Any] | None = None,
        top_k: int | None = None,
        debug: bool | None = None,
        trace_input: dict[str, Any] | None = None,
        trace_label: str = "this gap",
        model_label: str = "Selected AI model",
        token_usage_callback: Callable[
            [str, list[dict[str, str]], str, object],
            dict[str, Any],
        ]
        | None = None,
        token_budget_guard: Callable[[str, list[dict[str, str]]], None] | None = None,
        status_callback: Callable[[str], None] | None = None,
        trace_step_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[GraphRagAnswer, list[dict[str, Any]]]:
        full_question = _merge_question_context(question, context)
        trace_steps: list[dict[str, Any]] = []

        def add_trace_step(step: dict[str, Any]) -> None:
            trace_steps.append(step)
            if trace_step_callback is not None:
                trace_step_callback(step)

        if status_callback is not None:
            status_callback(f"Planning standards searches for {trace_label}")
        plan = self.planner.plan(full_question)
        effective_top_k = top_k or self.settings.top_k
        plan_output = _compact_plan_output(full_question, plan, effective_top_k)
        add_trace_step(
            {
                "name": f"Plan searches for {trace_label}",
                "explanation": (
                    "Turns one weak assessment answer into focused searches that can be sent "
                    "to the standards library."
                ),
                "tool": "Python code",
                "input": trace_input or {"question": full_question},
                "process": (
                    "We take the question from the previous step and split it into smaller "
                    "searches for gaps, threats, vulnerabilities, risks, controls, and standards."
                ),
                "output": plan_output,
            }
        )
        if status_callback is not None:
            status_callback(f"Searching standards evidence for {trace_label}")
        evidence, graph_rows = self._retriever().retrieve(
            plan,
            top_k=effective_top_k,
        )
        prompt_evidence = select_prompt_evidence(full_question, evidence)
        graph_context_rows = build_graph_context_rows(graph_rows, prompt_evidence)
        retrieved_chunks = [hit.model_dump() for hit in evidence]
        retrieval_output = {
            "retrieved_chunks": _compact_retrieved_evidence(evidence),
            "prompt_evidence": prompt_evidence_summary(
                full_question,
                len(evidence),
                prompt_evidence,
            ),
            "graph_context_rows": graph_context_rows,
            "raw_graph_row_count": len(graph_rows),
        }
        add_trace_step(
            {
                "name": f"Search standards evidence for {trace_label}",
                "explanation": (
                    "Looks in the standards text and relationship graph for evidence related "
                    "to this gap."
                ),
                "tool": "Qdrant vector search + BM25 keyword search + Neo4j graph lookup",
                "input": plan_output,
                "process": (
                    "We use the search plan from the previous step to find matching standards "
                    "paragraphs and graph links. This step does not ask the AI model to invent "
                    "an answer; it only collects evidence."
                ),
                "output": retrieval_output,
            }
        )
        if not prompt_evidence:
            answer = insufficient_evidence_answer()
            add_trace_step(
                {
                    "name": f"Handle missing evidence for {trace_label}",
                    "explanation": (
                        "Stops this gap from being answered when the local knowledge base did "
                        "not return useful evidence."
                    ),
                    "tool": "Python code",
                    "input": retrieval_output,
                    "process": (
                        "Because the previous search did not find usable evidence, we return an "
                        "insufficient-evidence result instead of asking the model to guess."
                    ),
                    "output": answer.model_dump(),
                }
            )
            return answer, trace_steps
        if status_callback is not None:
            status_callback(f"Preparing model prompt for {trace_label}")
        messages = build_structured_answer_prompt(
            full_question,
            plan,
            prompt_evidence,
            graph_context_rows,
        )
        prompt_contract_gate = validate_prompt_contract(
            gate="risk_answer_prompt",
            messages=messages,
            required_phrases=[
                "Role:",
                "Objective:",
                "Trusted evidence boundary:",
                "Forbidden behavior:",
                "Output contract:",
                "Do not use general training knowledge",
            ],
            task_name="risk answer",
        )
        prompt_focus_gate = validate_prompt_quality(
            gate="risk_answer_prompt_quality",
            messages=messages,
            task_name="risk answer",
            max_total_chars=8_500,
            max_user_chars=6_900,
            max_source_markers=5,
        )
        prompt_gate = combine_gate_results(
            "risk_answer_prompt",
            [prompt_contract_gate, prompt_focus_gate],
        )
        if not prompt_gate.passed:
            add_trace_step(
                {
                    "name": f"Quality gate failed for risk prompt {trace_label}",
                    "explanation": (
                        "Stops before calling the model because the prompt is incomplete."
                    ),
                    "tool": "Python quality gate",
                    "input": {"messages_sent_to_model": messages},
                    "process": (
                        "We check the generated prompt for the role, task, trusted evidence "
                        "boundary, forbidden behavior, output contract, and failure rules."
                    ),
                    "output": prompt_gate.as_dict(),
                }
            )
            raise QualityGateFailure(prompt_gate)
        prompt_output = {
            "model_prompt": _compact_prompt_messages(messages),
            "selected_prompt_evidence": retrieval_output["prompt_evidence"],
            "graph_context_rows": graph_context_rows,
            "quality_gate": prompt_gate.as_dict(),
        }
        add_trace_step(
            {
                "name": f"Prepare model prompt for {trace_label}",
                "explanation": (
                    "Builds the exact model input from the evidence found in the standards "
                    "library."
                ),
                "tool": "Python deterministic workflow",
                "input": retrieval_output,
                "process": (
                    "We take the retrieved standards evidence from the previous step and place "
                    "it into a strict prompt. This is still plain code; no model has been asked "
                    "to write the risk answer yet."
                ),
                "output": prompt_output,
            }
        )
        call_name = f"Risk answer model call for {trace_label}"
        attempts: list[dict[str, Any]] = []
        current_messages = messages
        final_gate = None
        raw = ""
        structured = None
        token_usage = None
        for attempt in range(1, 4):
            attempt_name = call_name if attempt == 1 else f"{call_name} repair attempt {attempt}"
            if token_budget_guard is not None:
                token_budget_guard(attempt_name, current_messages)
            if status_callback is not None:
                status_callback(
                    f"Waiting for model risk answer for {trace_label} "
                    f"(attempt {attempt})"
                )
            raw = self.chat_model.chat(current_messages)
            if status_callback is not None:
                status_callback(
                    f"Validating model risk answer for {trace_label} "
                    f"(attempt {attempt})"
                )
            token_usage = (
                token_usage_callback(
                    attempt_name,
                    current_messages,
                    raw,
                    self.chat_model,
                )
                if token_usage_callback is not None
                else None
            )
            try:
                structured = parse_structured_answer_strict(raw, prompt_evidence)
                structured = prune_unsupported_risk_answer(
                    answer=structured,
                    evidence=prompt_evidence,
                    question=full_question,
                )
                final_gate = validate_risk_answer(
                    answer=structured,
                    evidence=prompt_evidence,
                    raw_model_output=raw,
                    question=full_question,
                )
            except Exception as exc:
                final_gate = failed_gate(
                    "risk_answer_output",
                    "The model response could not be parsed as the required risk-answer JSON.",
                    [
                        GateIssue(
                            field="raw_model_output",
                            message=str(exc),
                            operator_fix=(
                                "Do not approve this run. The model did not produce usable "
                                "structured risk data."
                            ),
                            system_fix=(
                                "Retry with the repair prompt, reduce prompt size, or use a "
                                "stronger model."
                            ),
                        )
                    ],
                )
            attempts.append(
                {
                    "attempt": attempt,
                    "raw_model_response_preview": _preview_text(raw, 1_500),
                    "token_usage": token_usage,
                    "quality_gate": final_gate.as_dict(),
                }
            )
            if final_gate.passed and structured is not None:
                break
            if attempt < 3:
                current_messages = repair_prompt(
                    original_messages=messages,
                    gate_result=final_gate,
                )
        if final_gate is None or not final_gate.passed or structured is None:
            add_trace_step(
                {
                    "name": f"Quality gate failed for risk answer {trace_label}",
                    "explanation": (
                        "Stops because the model did not produce a trustworthy risk answer "
                        "after repair retries."
                    ),
                    "tool": "Python quality gate",
                    "input": prompt_output,
                    "process": (
                        "We validate schema, content, citations, and evidence support. The "
                        "workflow stops instead of passing bad risk facts forward."
                    ),
                    "output": {
                        "attempts": attempts,
                        "quality_gate": final_gate.as_dict() if final_gate else None,
                        "human_explanation": human_failure_message(final_gate)
                        if final_gate
                        else "The quality gate did not produce a result.",
                    },
                }
            )
            raise QualityGateFailure(final_gate) if final_gate else RuntimeError(
                "Risk answer quality gate failed without a gate result."
            )
        sources = structured.source_citations or [
            {
                "id": f"S{index}",
                "source": hit.source,
                "chunk_id": hit.chunk.id,
                "score": hit.score,
                "metadata": hit.chunk.metadata,
            }
            for index, hit in enumerate(prompt_evidence, 1)
        ]
        model_output = {
            "raw_model_response_preview": _preview_text(raw, 1_500),
            "structured_answer": structured.model_dump(),
            "sources": sources,
            "attempts": attempts,
            "quality_gate": final_gate.as_dict(),
        }
        if token_usage is not None:
            model_output["token_usage"] = token_usage
        add_trace_step(
            {
                "name": f"Ask model to write risk answer for {trace_label}",
                "explanation": (
                    "Asks the selected model to turn the retrieved evidence into a structured "
                    "risk and control answer."
                ),
                "tool": model_label,
                "input": prompt_output,
                "process": (
                    "We send only the assessment question and the retrieved evidence to the "
                    "model. The model must produce threats, vulnerabilities, risks, controls, "
                    "and source references from that evidence."
                ),
                "output": model_output,
            }
        )
        debug_enabled = self.settings.debug if debug is None else debug
        debug_payload = {
            "plan": plan.model_dump(),
            "retrieved_chunks": retrieved_chunks,
            "graph_context_rows": graph_context_rows,
            "raw_graph_row_count": len(graph_rows),
            "prompt_messages": messages,
            "raw_model_response": raw,
        }
        log_payload = (
            debug_payload
            if debug_enabled
            else {"plan": plan.model_dump(), "sources": sources}
        )
        log_path = self.logger.log_query(log_payload)
        response_debug = (
            {**debug_payload, "log_path": str(log_path)}
            if debug_enabled
            else {"log_path": str(log_path)}
        )
        return GraphRagAnswer(
            answer=structured,
            insufficient_evidence=False,
            sources=sources,
            debug=response_debug,
        ), trace_steps

    def retrieve_debug(self, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        plan = self.planner.plan(question)
        evidence, graph_rows = self._retriever().retrieve(
            plan,
            top_k=top_k or self.settings.top_k,
        )
        graph_context_rows = build_graph_context_rows(graph_rows, evidence)
        return {
            "plan": plan.model_dump(),
            "retrieved_chunks": _compact_retrieved_evidence(evidence),
            "graph_context_rows": graph_context_rows,
            "raw_graph_row_count": len(graph_rows),
        }

    def foundation_summary(
        self,
        packet: FoundationAssessmentPacket,
        *,
        debug: bool = False,
    ) -> FoundationSummaryResponse:
        return self.foundation_workflow.summarize(packet, debug=debug)

    def _retriever(self) -> HybridRetriever:
        if self.retriever is None:
            self.retriever = HybridRetriever(
                embedding_client=self.embedding_client,
                dense_store=self._dense_store_instance(),
                keyword_index=self._keyword_index_instance(),
                graph_store=self._graph_store_instance(),
            )
        return self.retriever

    def _dense_store_instance(self) -> DenseStore:
        if self.dense_store is None:
            self.dense_store = self._dense_store(self.settings)
        return self.dense_store

    def _keyword_index_instance(self) -> KeywordIndex:
        if self.keyword_index is None:
            self.keyword_index = KeywordIndex(path=self.settings.keyword_index_path)
        return self.keyword_index

    def _graph_store_instance(self) -> GraphStore:
        if self.graph_store is None:
            self.graph_store = self._graph_store(self.settings)
        return self.graph_store

    @staticmethod
    def _dense_store(settings: Settings) -> DenseStore:
        if settings.vector_backend == "memory":
            return MemoryDenseStore()
        return QdrantDenseStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            vector_size=settings.embedding_dimensions,
        )

    @staticmethod
    def _graph_store(settings: Settings) -> GraphStore:
        if settings.graph_backend == "memory":
            return MemoryGraphStore()
        return Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )


def _merge_question_context(question: str, context: dict[str, Any] | None) -> str:
    if not context:
        return question
    context_lines = "\n".join(f"{key}: {value}" for key, value in sorted(context.items()))
    return f"{question}\n\nStructured context:\n{context_lines}"


def _compact_plan_output(
    full_question: str,
    plan: Any,
    top_k: int,
) -> dict[str, Any]:
    return {
        "question_for_retrieval": full_question,
        "search_focus": [
            {
                "label": sub_question.label,
                "focus": sub_question.focus,
                "plain_goal": _plain_search_goal(sub_question.focus),
            }
            for sub_question in plan.sub_questions
        ],
        "top_k": top_k,
        "handoff_note": (
            "The full internal query strings stay inside the retriever. This visible handoff "
            "shows what we are trying to find without repeating the full question six times."
        ),
    }


def _plain_search_goal(focus: str) -> str:
    goals = {
        "gap": "Confirm the control gap described by the weak answer.",
        "threat": "Find threat language related to this gap.",
        "vulnerability": "Find what weakness or missing capability can be exploited.",
        "risk": "Find business or compliance impact language.",
        "control": "Find concrete controls that reduce the gap or risk.",
        "compliance": "Find framework references that support the controls.",
    }
    return goals.get(focus, f"Find evidence for {focus}.")


def _compact_prompt_messages(messages: list[dict[str, str]]) -> dict[str, Any]:
    total_chars = sum(len(message.get("content", "")) for message in messages)
    full_text = "\n".join(message.get("content", "") for message in messages)
    return {
        "message_count": len(messages),
        "total_characters_sent_to_model": total_chars,
        "output_contract_preview": _extract_output_contract_preview(full_text),
        "messages": [
            {
                "role": message.get("role"),
                "characters": len(message.get("content", "")),
                "preview": _preview_text(message.get("content", ""), 2_000),
            }
            for message in messages
        ],
        "note": (
            "This is a compact preview of the prompt sent to the model. Full prompt text is "
            "stored in the evaluation/debug log for development review."
        ),
    }


def _extract_output_contract_preview(text: str) -> str:
    marker = "Return exactly this JSON shape:"
    start = text.find(marker)
    if start < 0:
        marker = "Output contract:"
        start = text.find(marker)
    if start < 0:
        return ""
    return _preview_text(text[start:], 1_200)


def _compact_retrieved_evidence(evidence: list[Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for index, hit in enumerate(evidence, 1):
        metadata = dict(hit.chunk.metadata)
        compact.append(
            {
                "source_id": f"S{index}",
                "chunk_id": hit.chunk.id,
                "score": round(float(hit.score), 6),
                "source": hit.source,
                "retrieval_method": hit.retrieval_method,
                "sub_question": hit.sub_question,
                "metadata": {
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
                },
                "text_preview": _preview_text(hit.chunk.text, 320),
            }
        )
    return compact


def _preview_text(value: object, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} [...]"
