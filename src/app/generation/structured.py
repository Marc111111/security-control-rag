from __future__ import annotations

import json
import re

from app.schemas import GraphRagAnswer, MatrixRow, RetrievedEvidence, StructuredRiskAnswer


def parse_structured_answer(raw: str, evidence: list[RetrievedEvidence]) -> StructuredRiskAnswer:
    try:
        return parse_structured_answer_strict(raw, evidence)
    except Exception:
        return fallback_answer(raw, evidence)


def parse_structured_answer_strict(
    raw: str,
    evidence: list[RetrievedEvidence],
) -> StructuredRiskAnswer:
    data = json.loads(_extract_json(raw))
    data = _normalize_model_json(data, evidence)
    answer = StructuredRiskAnswer.model_validate(data)
    return augment_answer_with_evidence_controls(answer, evidence)


def fallback_answer(raw: str, evidence: list[RetrievedEvidence]) -> StructuredRiskAnswer:
    citations = _source_citations(evidence)
    controls = _extract_control_labels(evidence)
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


def augment_answer_with_evidence_controls(
    answer: StructuredRiskAnswer,
    evidence: list[RetrievedEvidence],
) -> StructuredRiskAnswer:
    extracted = _extract_control_labels(evidence)
    evidence_text = "\n".join(hit.chunk.text for hit in evidence).lower()
    supported_existing = [
        control
        for control in answer.recommended_controls
        if control.lower() in evidence_text
    ]
    controls = list(dict.fromkeys([*extracted, *supported_existing]))[:5]
    rows = [
        row.model_copy(update={"controls": list(dict.fromkeys([*controls, *row.controls]))[:5]})
        for row in answer.risk_control_matrix
    ]
    return answer.model_copy(
        update={
            "recommended_controls": controls or answer.recommended_controls,
            "risk_control_matrix": rows,
        }
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


def _normalize_model_json(
    data: object,
    evidence: list[RetrievedEvidence],
) -> dict[str, object]:
    if not isinstance(data, dict):
        return {}
    normalized = dict(data)
    for key in ["assumptions", "threats", "vulnerabilities", "risks", "recommended_controls"]:
        normalized[key] = _list_of_strings(normalized.get(key))
    normalized["missing_information"] = _list_of_strings(
        normalized.get("missing_information")
    )
    rows = normalized.get("risk_control_matrix")
    if isinstance(rows, list):
        normalized["risk_control_matrix"] = [_normalize_matrix_row(row) for row in rows]
    else:
        normalized["risk_control_matrix"] = []
    citations = normalized.get("source_citations")
    if isinstance(citations, list) and citations and all(
        isinstance(item, str) for item in citations
    ):
        citation_map = {citation["id"]: citation for citation in _source_citations(evidence)}
        normalized["source_citations"] = [
            citation_map.get(item, {"id": item}) for item in citations
        ]
    elif not isinstance(citations, list):
        normalized["source_citations"] = []
    return normalized


def _normalize_matrix_row(row: object) -> dict[str, object]:
    if not isinstance(row, dict):
        return {}
    normalized = dict(row)
    for key in ["gap", "threat", "vulnerability", "risk", "likelihood", "impact"]:
        normalized[key] = _string_value(normalized.get(key))
    normalized["controls"] = _list_of_strings(normalized.get("controls"))
    normalized["evidence"] = _list_of_strings(normalized.get("evidence"))
    return normalized


def _list_of_strings(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    return [_string_value(item) for item in value if _string_value(item)]


def _string_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ["name", "title", "control_id", "id", "value"]:
            item = value.get(key)
            if item:
                return str(item)
    return str(value)


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


def _extract_control_labels(evidence: list[RetrievedEvidence]) -> list[str]:
    controls: list[str] = []
    for hit in evidence:
        text = hit.chunk.text
        controls.extend(_cis_safeguards(text))
        controls.extend(_cis_controls(text))
        controls.extend(_scf_controls(text))
    return list(dict.fromkeys(controls))[:8]


def _cis_safeguards(text: str) -> list[str]:
    return [
        f"CIS Safeguard {match.group(1)} - {_clean_title(match.group(2))}"
        for match in re.finditer(
            r"Safeguard\s+(\d+(?:\.\d+)?):\s*(.+?)(?=\s+Asset Type|\s+\| |\n|$)",
            text,
            re.I | re.S,
        )
    ]


def _cis_controls(text: str) -> list[str]:
    return [
        f"CIS Control {match.group(1)} - {_clean_title(match.group(2))}"
        for match in re.finditer(
            r"Control\s+(\d+):\s*([A-Z][^.\n]{3,90})",
            text,
        )
    ]


def _scf_controls(text: str) -> list[str]:
    labels: list[str] = []
    for match in re.finditer(
        r"SCF Control:\s*(.+?)\s+SCF #:\s*([A-Z]+-\d+(?:\.\d+)?)",
        text,
        re.I | re.S,
    ):
        labels.append(f"SCF {match.group(2)} - {_clean_title(match.group(1))}")
    for match in re.finditer(
        r"SCF Control Name:\s*(.+?)\s+SCF Control #:\s*([A-Z]+-\d+(?:\.\d+)?)",
        text,
        re.I | re.S,
    ):
        labels.append(f"SCF {match.group(2)} - {_clean_title(match.group(1))}")
    return labels


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .:-|")
