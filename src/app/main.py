from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Protocol

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.assessment.mock_data import MockFoundationChatModel, sample_foundation_packet
from app.assessment.schemas import FoundationAssessmentPacket
from app.assessment.token_estimator import estimate_foundation_summary_tokens
from app.assessment.workflow import FoundationAssessmentWorkflow
from app.config import Settings, load_settings
from app.generation.clients import OpenAIChatClient
from app.pipeline import GraphRagPipeline
from secure_rag.llm import OllamaChatClient


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

    def foundation_summary(
        self,
        packet: FoundationAssessmentPacket,
        *,
        debug: bool = False,
    ) -> object:
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


class FoundationSummaryRequest(BaseModel):
    packet: FoundationAssessmentPacket
    debug: bool = False


class TokenEstimateRequest(BaseModel):
    packet: FoundationAssessmentPacket
    model: str = "gpt-4.1-mini"
    estimated_output_tokens: int = Field(default=900, ge=100, le=4_000)


class OpenAISmokeTestRequest(TokenEstimateRequest):
    confirm_external_call: bool = False
    max_estimated_input_tokens: int = Field(default=6_000, ge=500, le=20_000)


class FoundationModelRunRequest(TokenEstimateRequest):
    provider: Literal["mock", "ollama", "openai"] = "ollama"
    debug: bool = True
    confirm_external_call: bool = False
    max_estimated_input_tokens: int = Field(default=6_000, ge=500, le=20_000)


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

    @app.get("/mock/foundation", response_class=HTMLResponse, include_in_schema=False)
    def foundation_mock_ui() -> str:
        return _static_file("foundation_mock.html")

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

    @app.post("/api/assessments/foundation-summary")
    def foundation_summary(
        request: FoundationSummaryRequest,
        current: GraphRagService = Depends(get_service),  # noqa: B008
    ) -> dict[str, Any]:
        try:
            response = current.foundation_summary(request.packet, debug=request.debug)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return response.model_dump()

    @app.get("/api/mock/foundation-packet")
    def mock_foundation_packet() -> dict[str, Any]:
        return sample_foundation_packet().model_dump(mode="json")

    @app.post("/api/mock/foundation-summary")
    def mock_foundation_summary(request: FoundationSummaryRequest) -> dict[str, Any]:
        workflow = FoundationAssessmentWorkflow(MockFoundationChatModel())
        return workflow.summarize(request.packet, debug=request.debug).model_dump()

    @app.post("/api/assessments/foundation-summary/token-estimate")
    def foundation_token_estimate(request: TokenEstimateRequest) -> dict[str, Any]:
        return estimate_foundation_summary_tokens(
            request.packet,
            model=request.model,
            estimated_output_tokens=request.estimated_output_tokens,
        ).as_dict()

    @app.post("/api/assessments/foundation-summary/openai-smoke-test")
    def foundation_openai_smoke_test(request: OpenAISmokeTestRequest) -> dict[str, Any]:
        if not request.confirm_external_call:
            raise HTTPException(
                status_code=400,
                detail="Set confirm_external_call=true to send this compact packet to OpenAI.",
            )
        estimate = estimate_foundation_summary_tokens(
            request.packet,
            model=request.model,
            estimated_output_tokens=request.estimated_output_tokens,
        )
        if estimate.estimated_input_tokens > request.max_estimated_input_tokens:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Estimated input tokens exceed the request guard. "
                    f"Estimated {estimate.estimated_input_tokens}, "
                    f"limit {request.max_estimated_input_tokens}."
                ),
            )
        settings = load_settings()
        try:
            workflow = FoundationAssessmentWorkflow(
                OpenAIChatClient(
                    api_key=settings.openai_api_key,
                    model=request.model,
                    max_output_tokens=request.estimated_output_tokens,
                )
            )
            response = workflow.summarize(request.packet, debug=True)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        data = response.model_dump()
        data["token_estimate"] = estimate.as_dict()
        return data

    @app.post("/api/assessments/foundation-summary/model-run")
    def foundation_model_run(request: FoundationModelRunRequest) -> dict[str, Any]:
        estimate = estimate_foundation_summary_tokens(
            request.packet,
            model=request.model,
            estimated_output_tokens=request.estimated_output_tokens,
        )
        if estimate.estimated_input_tokens > request.max_estimated_input_tokens:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Estimated input tokens exceed the request guard. "
                    f"Estimated {estimate.estimated_input_tokens}, "
                    f"limit {request.max_estimated_input_tokens}."
                ),
            )
        settings = load_settings()
        if request.provider == "openai" and not request.confirm_external_call:
            raise HTTPException(
                status_code=400,
                detail="External OpenAI call blocked. Enable the external-call checkbox first.",
            )
        try:
            workflow = FoundationAssessmentWorkflow(
                _chat_model_for_comparison(
                    provider=request.provider,
                    model=request.model,
                    settings=settings,
                    max_output_tokens=request.estimated_output_tokens,
                )
            )
            response = workflow.summarize(request.packet, debug=request.debug)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        data = response.model_dump()
        data["model_run"] = {
            "provider": request.provider,
            "model": request.model,
            "external_call": request.provider == "openai",
            "token_estimate": estimate.as_dict(),
        }
        return data

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


def _static_file(name: str) -> str:
    path = Path(__file__).with_name("static") / name
    return path.read_text(encoding="utf-8")


def _chat_model_for_comparison(
    *,
    provider: Literal["mock", "ollama", "openai"],
    model: str,
    settings: Settings,
    max_output_tokens: int,
) -> object:
    if provider == "mock":
        return MockFoundationChatModel()
    if provider == "openai":
        return OpenAIChatClient(
            api_key=settings.openai_api_key,
            model=model,
            max_output_tokens=max_output_tokens,
        )
    return OllamaChatClient(model=model, base_url=settings.ollama_base_url)


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=os.getenv("GRAPHRAG_HOST", "127.0.0.1"),
        port=int(os.getenv("GRAPHRAG_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
