from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import Settings, load_settings
from app.pipeline import GraphRagPipeline


class GraphRagService(Protocol):
    settings: Settings

    def ingest(
        self,
        source: str | Path,
        *,
        chunk_size: int = 1_500,
        overlap: int = 200,
        batch_size: int = 64,
    ) -> dict[str, int]:
        ...

    def query(
        self,
        question: str,
        *,
        context: dict[str, Any] | None = None,
        top_k: int | None = None,
        debug: bool | None = None,
    ) -> object:
        ...

    def retrieve_debug(self, question: str, *, top_k: int | None = None) -> dict[str, Any]:
        ...


class IngestRequest(BaseModel):
    source: str = Field(..., min_length=1)
    chunk_size: int = Field(default=1_500, ge=200, le=10_000)
    overlap: int = Field(default=200, ge=0, le=2_000)
    batch_size: int = Field(default=64, ge=1, le=512)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: dict[str, Any] | None = None
    top_k: int | None = Field(default=None, ge=1, le=40)
    debug: bool = False


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., min_length=1)
    chunk_id: str = Field(..., min_length=1)
    relevant: bool
    notes: str = ""


def create_app(service: GraphRagService | None = None) -> FastAPI:
    app = FastAPI(
        title="Cybersecurity GRC GraphRAG API",
        description=(
            "Local-first multi-step RAG and GraphRAG prototype for cybersecurity/GRC "
            "risk documentation."
        ),
        version="0.2.0",
    )
    runtime_service = service

    def get_service() -> GraphRagService:
        nonlocal runtime_service
        if runtime_service is None:
            runtime_service = GraphRagPipeline(load_settings())
        return runtime_service

    @app.get("/api/health")
    def health(current: GraphRagService = Depends(get_service)) -> dict[str, Any]:  # noqa: B008
        return {
            "status": "ok",
            "llm_provider": current.settings.llm_provider,
            "generation_model": current.settings.generation_model,
            "embedding_model": current.settings.embedding_model,
            "vector_backend": current.settings.vector_backend,
            "graph_backend": current.settings.graph_backend,
        }

    @app.post("/api/ingest")
    def ingest(
        request: IngestRequest,
        current: GraphRagService = Depends(get_service),  # noqa: B008
    ) -> dict[str, int]:
        try:
            return current.ingest(
                request.source,
                chunk_size=request.chunk_size,
                overlap=request.overlap,
                batch_size=request.batch_size,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/query")
    def query(
        request: QueryRequest,
        current: GraphRagService = Depends(get_service),  # noqa: B008
    ) -> dict[str, Any]:
        try:
            answer = current.query(
                request.question,
                context=request.context,
                top_k=request.top_k,
                debug=request.debug,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return answer.model_dump()

    @app.post("/api/retrieve")
    def retrieve(
        request: QueryRequest,
        current: GraphRagService = Depends(get_service),  # noqa: B008
    ) -> dict[str, Any]:
        try:
            return current.retrieve_debug(request.question, top_k=request.top_k)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/evaluation/feedback")
    def feedback(
        request: FeedbackRequest,
        current: GraphRagService = Depends(get_service),  # noqa: B008
    ) -> dict[str, Any]:
        logger = getattr(current, "logger", None)
        if logger is None:
            raise HTTPException(
                status_code=400,
                detail="service does not expose an evaluation logger",
            )
        return logger.manual_feedback(
            query_id=request.query_id,
            chunk_id=request.chunk_id,
            relevant=request.relevant,
            notes=request.notes,
        )

    return app


app = create_app()


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.getenv("GRAPHRAG_HOST", "127.0.0.1"),
        port=int(os.getenv("GRAPHRAG_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
