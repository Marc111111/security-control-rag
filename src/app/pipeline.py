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
from app.generation.structured import insufficient_evidence_answer, parse_structured_answer
from app.graph.extractor import HeuristicGraphExtractor
from app.graph.store import GraphStore, MemoryGraphStore, Neo4jGraphStore
from app.ingestion.chunker import load_and_chunk_path
from app.planning.planner import RiskQuestionPlanner
from app.retrieval.bm25 import KeywordIndex
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
    ) -> tuple[GraphRagAnswer, list[dict[str, Any]]]:
        full_question = _merge_question_context(question, context)
        if status_callback is not None:
            status_callback(f"Planning standards searches for {trace_label}")
        plan = self.planner.plan(full_question)
        effective_top_k = top_k or self.settings.top_k
        plan_output = {
            "question_for_retrieval": full_question,
            "search_plan": plan.model_dump(),
            "top_k": effective_top_k,
        }
        trace_steps: list[dict[str, Any]] = [
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
        ]
        if status_callback is not None:
            status_callback(f"Searching standards evidence for {trace_label}")
        evidence, graph_rows = self._retriever().retrieve(
            plan,
            top_k=effective_top_k,
        )
        retrieved_chunks = [hit.model_dump() for hit in evidence]
        retrieval_output = {
            "retrieved_chunks": retrieved_chunks,
            "graph_rows": graph_rows,
        }
        trace_steps.append(
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
        if not evidence:
            answer = insufficient_evidence_answer()
            trace_steps.append(
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
        messages = build_structured_answer_prompt(full_question, plan, evidence, graph_rows)
        prompt_output = {
            "messages_sent_to_model": messages,
            "retrieved_evidence": retrieval_output,
        }
        trace_steps.append(
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
        if token_budget_guard is not None:
            token_budget_guard(call_name, messages)
        if status_callback is not None:
            status_callback(f"Waiting for model risk answer for {trace_label}")
        raw = self.chat_model.chat(messages)
        if status_callback is not None:
            status_callback(f"Parsing model risk answer for {trace_label}")
        token_usage = (
            token_usage_callback(
                call_name,
                messages,
                raw,
                self.chat_model,
            )
            if token_usage_callback is not None
            else None
        )
        structured = parse_structured_answer(raw, evidence)
        sources = structured.source_citations or [
            {
                "id": f"S{index}",
                "source": hit.source,
                "chunk_id": hit.chunk.id,
                "score": hit.score,
                "metadata": hit.chunk.metadata,
            }
            for index, hit in enumerate(evidence, 1)
        ]
        model_output = {
            "raw_model_response": raw,
            "structured_answer": structured.model_dump(),
            "sources": sources,
        }
        if token_usage is not None:
            model_output["token_usage"] = token_usage
        trace_steps.append(
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
            "graph_rows": graph_rows,
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
        return {
            "plan": plan.model_dump(),
            "retrieved_chunks": [hit.model_dump() for hit in evidence],
            "graph_rows": graph_rows,
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
