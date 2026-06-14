from __future__ import annotations

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
        full_question = _merge_question_context(question, context)
        plan = self.planner.plan(full_question)
        evidence, graph_rows = self._retriever().retrieve(
            plan,
            top_k=top_k or self.settings.top_k,
        )
        if not evidence:
            return insufficient_evidence_answer()
        messages = build_structured_answer_prompt(full_question, plan, evidence, graph_rows)
        raw = self.chat_model.chat(messages)
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
        debug_enabled = self.settings.debug if debug is None else debug
        debug_payload = {
            "plan": plan.model_dump(),
            "retrieved_chunks": [hit.model_dump() for hit in evidence],
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
        )

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
