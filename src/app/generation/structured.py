from __future__ import annotations

import json
import re

from app.schemas import GraphRagAnswer, MatrixRow, RetrievedEvidence, StructuredRiskAnswer


def parse_structured_answer(raw: str, evidence: list[RetrievedEvidence]) -> StructuredRiskAnswer:
    try:
        data = json.loads(_extract_json(raw))
        return StructuredRiskAnswer.model_validate(data)
    except Exception:
        return fallback_answer(raw, evidence)


def fallback_answer(raw: str, evidence: list[RetrievedEvidence]) -> StructuredRiskAnswer:
    citations = _source_citations(evidence)
    controls = [
        _control_label(hit)
        for hit in evidence
        if hit.chunk.metadata.get("control_id") or "control" in hit.chunk.text.lower()
    ]
    return StructuredRiskAnswer(
        executive_summary=(
            raw.strip()[:1_000]
            if raw.strip()
            else "Retrieved evidence was found, but the model did not return structured JSON."
        ),
        recommended_controls=list(dict.fromkeys(item for item in controls if item)),
        risk_control_matrix=[
            MatrixRow(
                gap="Question describes a cybersecurity/GRC gap.",
                threat="See retrieved evidence.",
                vulnerability="See retrieved evidence.",
                risk="See retrieved evidence.",
                controls=list(dict.fromkeys(item for item in controls if item))[:5],
                evidence=[f"S{index}" for index, _ in enumerate(evidence[:5], 1)],
            )
        ]
        if evidence
        else [],
        source_citations=citations,
        from_retrieved_evidence=(
            "The answer is based on the retrieved chunks listed in source_citations."
        ),
        general_model_reasoning=(
            "The model response was not valid JSON, so the service returned a conservative "
            "fallback structure."
        ),
    )


def insufficient_evidence_answer() -> GraphRagAnswer:
    return GraphRagAnswer(
        insufficient_evidence=True,
        sources=[],
        answer=StructuredRiskAnswer(
            executive_summary=(
                "The local corpus returned no sufficiently relevant evidence for this question."
            ),
            missing_information=["Index more source material or broaden the query criteria."],
            from_retrieved_evidence="No retrieved evidence was available.",
            general_model_reasoning=(
                "No general reasoning was used to invent controls or citations."
            ),
        ),
    )


def _extract_json(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _source_citations(evidence: list[RetrievedEvidence]) -> list[dict[str, object]]:
    citations: list[dict[str, object]] = []
    for index, hit in enumerate(evidence, 1):
        metadata = dict(hit.chunk.metadata)
        citations.append(
            {
                "id": f"S{index}",
                "chunk_id": hit.chunk.id,
                "score": round(hit.score, 6),
                "retrieval_method": hit.retrieval_method,
                "source": metadata.get("source_path") or metadata.get("filename") or hit.source,
                "metadata": metadata,
            }
        )
    return citations


def _control_label(hit: RetrievedEvidence) -> str:
    metadata = hit.chunk.metadata
    framework = metadata.get("framework")
    control_id = metadata.get("control_id")
    if framework and control_id:
        return f"{framework} {control_id}"
    if control_id:
        return str(control_id)
    return ""
