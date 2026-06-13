from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from secure_rag.embeddings import OllamaEmbeddingClient
from secure_rag.engine import ControlRagEngine
from secure_rag.indexer import RagIndexer
from secure_rag.llm import OllamaChatClient
from secure_rag.retriever import Retriever
from secure_rag.schema import ControlAnswer, RetrievalHit
from secure_rag.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class RagRuntimeConfig:
    db_path: Path = Path("storage/chroma")
    embedding_model: str = "mxbai-embed-large"
    generation_model: str = "gemma3:4b"
    top_k: int = 8
    min_score: float = 0.6


class RagRuntime:
    def __init__(self, config: RagRuntimeConfig) -> None:
        self.config = config
        self.embedding_client = OllamaEmbeddingClient(model=config.embedding_model)
        self.vector_store = ChromaVectorStore(path=config.db_path)
        self.retriever = Retriever(
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
        )
        self.engine = ControlRagEngine(
            retriever=self.retriever,
            chat_client=OllamaChatClient(model=config.generation_model),
            min_score=config.min_score,
        )

    def ingest(
        self,
        source: str | Path,
        *,
        chunk_size: int = 1_500,
        overlap: int = 200,
        batch_size: int = 64,
    ) -> int:
        indexer = RagIndexer(
            embedding_client=self.embedding_client,
            vector_store=self.vector_store,
            chunk_size=chunk_size,
            overlap=overlap,
            batch_size=batch_size,
        )
        return len(indexer.index_path(source))

    def query(
        self,
        *,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> ControlAnswer:
        criteria = build_criteria(message=message, context=context, history=history)
        return self.engine.answer(criteria, top_k=top_k or self.config.top_k)

    def retrieve(
        self,
        *,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievalHit]:
        criteria = build_criteria(message=message, context=context, history=history)
        return self.retriever.retrieve(criteria, top_k=top_k or self.config.top_k)


def build_criteria(
    *,
    message: str,
    context: dict[str, Any] | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    clean_message = message.strip()
    if not clean_message:
        raise ValueError("message cannot be empty")
    parts = [clean_message]
    if history:
        previous_user_turns = [
            turn.get("content", "").strip()
            for turn in history[-6:]
            if turn.get("role") == "user" and turn.get("content", "").strip()
        ]
        if previous_user_turns:
            parts.extend(["", "Conversation context:", *previous_user_turns])
    if context:
        parts.extend(
            [
                "",
                "Structured criteria JSON:",
                json.dumps(context, ensure_ascii=True, sort_keys=True, indent=2),
            ]
        )
    return "\n".join(parts)
