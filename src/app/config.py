from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "ollama"
    generation_model: str = "qwen3:14b"
    planner_model: str = "qwen3:14b"
    embedding_model: str = "mxbai-embed-large"
    embedding_dimensions: int = 1024
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "cyber_grc_chunks"
    vector_backend: str = "qdrant"
    graph_backend: str = "neo4j"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "please-change-me"
    top_k: int = 12
    debug: bool = False


def load_settings() -> Settings:
    return Settings(
        llm_provider=os.getenv("GRAPHRAG_LLM_PROVIDER", "ollama").lower(),
        generation_model=os.getenv("GRAPHRAG_GENERATION_MODEL", "qwen3:14b"),
        planner_model=os.getenv("GRAPHRAG_PLANNER_MODEL", "qwen3:14b"),
        embedding_model=os.getenv("GRAPHRAG_EMBEDDING_MODEL", "mxbai-embed-large"),
        embedding_dimensions=int(os.getenv("GRAPHRAG_EMBEDDING_DIMENSIONS", "1024")),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "cyber_grc_chunks"),
        vector_backend=os.getenv("GRAPHRAG_VECTOR_BACKEND", "qdrant").lower(),
        graph_backend=os.getenv("GRAPHRAG_GRAPH_BACKEND", "neo4j").lower(),
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "please-change-me"),
        top_k=int(os.getenv("GRAPHRAG_TOP_K", "12")),
        debug=os.getenv("GRAPHRAG_DEBUG", "false").lower() in {"1", "true", "yes"},
    )

