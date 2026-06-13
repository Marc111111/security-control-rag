from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from secure_rag.runtime import RagRuntime, RagRuntimeConfig


class RagService(Protocol):
    config: RagRuntimeConfig

    def ingest(
        self,
        source: str | Path,
        *,
        chunk_size: int = 1_500,
        overlap: int = 200,
        batch_size: int = 64,
    ) -> int:
        ...

    def query(
        self,
        *,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> object:
        ...

    def retrieve(
        self,
        *,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> list[object]:
        ...


class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1)


class QueryRequest(BaseModel):
    message: str = Field(..., min_length=1)
    context: dict[str, Any] | None = None
    history: list[ChatTurn] = Field(default_factory=list)
    top_k: int | None = Field(default=None, ge=1, le=30)


class IngestRequest(BaseModel):
    source: str = Field(..., min_length=1)
    chunk_size: int = Field(default=1_500, ge=200, le=10_000)
    overlap: int = Field(default=200, ge=0, le=2_000)
    batch_size: int = Field(default=64, ge=1, le=512)


class HealthResponse(BaseModel):
    status: str
    db_path: str
    embedding_model: str
    generation_model: str
    mode: str = "strict_corpus_first"


def create_app(service: RagService | None = None) -> FastAPI:
    app = FastAPI(
        title="Security Control RAG API",
        description="Local corpus-grounded API for security-control recommendations.",
        version="0.1.0",
    )

    runtime_service = service

    def get_service() -> RagService:
        nonlocal runtime_service
        if runtime_service is None:
            runtime_service = RagRuntime(load_config_from_env())
        return runtime_service

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def ui() -> str:
        return _static_file("index.html")

    @app.get("/api/health", response_model=HealthResponse)
    def health(current: RagService = Depends(get_service)) -> HealthResponse:  # noqa: B008
        return HealthResponse(
            status="ok",
            db_path=current.config.db_path.as_posix(),
            embedding_model=current.config.embedding_model,
            generation_model=current.config.generation_model,
        )

    @app.post("/api/ingest")
    def ingest(
        request: IngestRequest,
        current: RagService = Depends(get_service),  # noqa: B008
    ) -> dict[str, int]:
        try:
            indexed_chunks = current.ingest(
                request.source,
                chunk_size=request.chunk_size,
                overlap=request.overlap,
                batch_size=request.batch_size,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"indexed_chunks": indexed_chunks}

    @app.post("/api/query")
    def query(
        request: QueryRequest,
        current: RagService = Depends(get_service),  # noqa: B008
    ) -> dict[str, Any]:
        try:
            answer = current.query(
                message=request.message,
                context=request.context,
                history=[turn.model_dump() for turn in request.history],
                top_k=request.top_k,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "answer": answer.answer,
            "insufficient_evidence": answer.insufficient_evidence,
            "sources": answer.sources,
            "raw": answer.raw,
        }

    @app.post("/api/retrieve")
    def retrieve(
        request: QueryRequest,
        current: RagService = Depends(get_service),  # noqa: B008
    ) -> dict[str, Any]:
        try:
            hits = current.retrieve(
                message=request.message,
                context=request.context,
                history=[turn.model_dump() for turn in request.history],
                top_k=request.top_k,
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "hits": [
                {
                    "score": hit.score,
                    "chunk": {
                        "id": hit.chunk.id,
                        "text": hit.chunk.text,
                        "metadata": hit.chunk.metadata,
                    },
                }
                for hit in hits
            ]
        }

    return app


def load_config_from_env() -> RagRuntimeConfig:
    return RagRuntimeConfig(
        db_path=Path(os.getenv("SECURE_RAG_DB_PATH", "storage/chroma")),
        embedding_model=os.getenv("SECURE_RAG_EMBEDDING_MODEL", "mxbai-embed-large"),
        generation_model=os.getenv("SECURE_RAG_GENERATION_MODEL", "gemma3:4b"),
        top_k=int(os.getenv("SECURE_RAG_TOP_K", "8")),
        min_score=float(os.getenv("SECURE_RAG_MIN_SCORE", "0.6")),
    )


def _static_file(name: str) -> str:
    path = Path(__file__).with_name("static") / name
    return path.read_text(encoding="utf-8")


app = create_app()


def main() -> None:
    uvicorn.run(
        "secure_rag.api:app",
        host=os.getenv("SECURE_RAG_HOST", "127.0.0.1"),
        port=int(os.getenv("SECURE_RAG_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
