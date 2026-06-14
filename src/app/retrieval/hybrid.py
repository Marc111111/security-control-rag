from __future__ import annotations

import re

from app.graph.store import GraphStore
from app.retrieval.bm25 import KeywordIndex
from app.retrieval.vector_store import DenseStore
from app.schemas import DocumentChunk, QueryPlan, RetrievedEvidence
from secure_rag.embeddings import EmbeddingClient


class HybridRetriever:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        dense_store: DenseStore,
        keyword_index: KeywordIndex,
        graph_store: GraphStore,
    ) -> None:
        self.embedding_client = embedding_client
        self.dense_store = dense_store
        self.keyword_index = keyword_index
        self.graph_store = graph_store

    def add_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        self.dense_store.add(chunks, embeddings)
        self.keyword_index.add(chunks)

    def retrieve(
        self,
        plan: QueryPlan,
        *,
        top_k: int = 12,
    ) -> tuple[list[RetrievedEvidence], list[dict[str, object]]]:
        candidates: list[RetrievedEvidence] = []
        graph_rows: list[dict[str, object]] = []
        for sub_question in plan.sub_questions:
            query = sub_question.question
            embedding = self.embedding_client.embed([query])[0]
            dense_hits = self.dense_store.search(embedding, top_k=top_k)
            keyword_hits = self.keyword_index.search(query, top_k=top_k)
            for hit in [*dense_hits, *keyword_hits]:
                candidates.append(hit.model_copy(update={"sub_question": sub_question.label}))
            graph_rows.extend(self.graph_store.search_related(query, limit=top_k))
        merged = deduplicate_and_rerank(candidates, plan.original_question)
        return merged[:top_k], graph_rows


def deduplicate_and_rerank(
    candidates: list[RetrievedEvidence],
    question: str,
) -> list[RetrievedEvidence]:
    by_id: dict[str, RetrievedEvidence] = {}
    terms = set(_tokens(question))
    for hit in candidates:
        overlap = len(terms & set(_tokens(hit.chunk.text))) / max(len(terms), 1)
        method_bonus = 0.08 if hit.retrieval_method == "keyword" else 0.0
        score = hit.score + overlap * 0.2 + method_bonus
        existing = by_id.get(hit.chunk.id)
        updated = hit.model_copy(update={"score": score})
        if existing is None or updated.score > existing.score:
            by_id[hit.chunk.id] = updated
    return sorted(by_id.values(), key=lambda item: item.score, reverse=True)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())
