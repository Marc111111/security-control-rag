from __future__ import annotations

import re

from secure_rag.embeddings import EmbeddingClient
from secure_rag.schema import RetrievalHit
from secure_rag.vector_store import VectorStore

SECURITY_EXPANSIONS = {
    "playbook": [
        "incident response plan",
        "incident response process",
        "incident handling",
        "response plan",
        "roles and responsibilities",
        "tabletop exercise",
        "communications",
    ],
    "ransomware": [
        "malware",
        "data recovery",
        "backup",
        "restore",
        "incident response",
        "business continuity",
    ],
    "small company": [
        "small office",
        "implementation group 1",
        "IG1",
        "foundational",
    ],
    "concrete controls": [
        "control",
        "safeguard",
        "control identifier",
        "control text",
    ],
}

GENERIC_SOURCE_PREFIXES = ("tests\\fixtures", "tests/fixtures")


class Retriever:
    def __init__(
        self,
        *,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
        excluded_source_prefixes: tuple[str, ...] = GENERIC_SOURCE_PREFIXES,
    ) -> None:
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.excluded_source_prefixes = excluded_source_prefixes

    def retrieve(self, query: str, *, top_k: int = 8) -> list[RetrievalHit]:
        expanded_query = expand_security_query(query)
        embedding = self.embedding_client.embed([expanded_query])[0]
        candidate_count = max(top_k * 8, 50)
        candidates = self.vector_store.search(embedding, top_k=candidate_count)
        filtered = [hit for hit in candidates if not self._excluded(hit)]
        reranked = rerank_hits(expanded_query, filtered)
        return reranked[:top_k]

    def _excluded(self, hit: RetrievalHit) -> bool:
        source_path = str(hit.chunk.metadata.get("source_path", ""))
        return any(source_path.startswith(prefix) for prefix in self.excluded_source_prefixes)


def expand_security_query(query: str) -> str:
    additions: list[str] = []
    lower_query = query.lower()
    for trigger, terms in SECURITY_EXPANSIONS.items():
        if trigger in lower_query:
            additions.extend(terms)
    if not additions:
        return query
    return f"{query}\nRelated security terms: {', '.join(dict.fromkeys(additions))}"


def rerank_hits(query: str, hits: list[RetrievalHit]) -> list[RetrievalHit]:
    query_terms = _terms(query)
    reranked: list[RetrievalHit] = []
    for hit in hits:
        text_terms = _terms(hit.chunk.text)
        overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
        source_bonus = _source_bonus(str(hit.chunk.metadata.get("source_path", "")))
        score = hit.score + (0.25 * overlap) + source_bonus
        reranked.append(RetrievalHit(chunk=hit.chunk, score=score, vector_score=hit.vector_score))
    return sorted(reranked, key=lambda item: item.score, reverse=True)


def _terms(text: str) -> set[str]:
    terms = set(re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower()))
    return {term for term in terms if term not in {"the", "and", "for", "with", "that", "this"}}


def _source_bonus(source_path: str) -> float:
    if source_path.startswith("standards\\") or source_path.startswith("standards/"):
        return 0.03
    return 0.0
