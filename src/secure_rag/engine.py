from __future__ import annotations

from secure_rag.llm import ChatClient
from secure_rag.prompts import build_grounded_prompt
from secure_rag.schema import ControlAnswer, Metadata, RetrievalHit


class ControlRagEngine:
    def __init__(
        self,
        *,
        retriever: object,
        chat_client: ChatClient,
        min_score: float = 0.6,
    ) -> None:
        self.retriever = retriever
        self.chat_client = chat_client
        self.min_score = min_score

    def answer(self, criteria: str, *, top_k: int = 8) -> ControlAnswer:
        hits: list[RetrievalHit] = self.retriever.retrieve(criteria, top_k=top_k)
        supported_hits = [hit for hit in hits if hit.score >= self.min_score]
        sources = [_source_metadata(hit) for hit in supported_hits]
        if not supported_hits:
            return ControlAnswer(
                answer=(
                    "The local corpus returned no sufficiently relevant evidence for this query. "
                    "Add or index more source material, or broaden the criteria."
                ),
                sources=[],
                insufficient_evidence=True,
            )

        messages = build_grounded_prompt(criteria, supported_hits)
        answer = self.chat_client.chat(messages)
        return ControlAnswer(answer=answer, sources=sources, raw={"hit_count": len(supported_hits)})


def _source_metadata(hit: RetrievalHit) -> Metadata:
    metadata = dict(hit.chunk.metadata)
    metadata["chunk_id"] = hit.chunk.id
    metadata["score"] = round(hit.score, 6)
    return metadata
