from __future__ import annotations

import re
from pathlib import Path

from app.schemas import DocumentChunk
from secure_rag.chunking import chunk_documents
from secure_rag.loaders import load_path
from secure_rag.schema import stable_id

FRAMEWORK_PATTERNS = {
    "NIST SP 800-53": re.compile(
        r"\b(?:AC|AT|AU|CA|CM|CP|IA|IR|MA|MP|PE|PL|PM|PS|PT|RA|SA|SC|SI|SR)-\d+\b"
    ),
    "NIST CSF": re.compile(r"\b(?:GV|ID|PR|DE|RS|RC)\.[A-Z]{2}-\d+\b"),
    "CIS Controls": re.compile(r"\b(?:CIS\s*)?(?:Safeguard\s*)?\d{1,2}(?:\.\d+)?\b", re.I),
    "ISO 27001": re.compile(r"\bA\.\d+(?:\.\d+){1,2}\b"),
}


def load_and_chunk_path(
    path: str | Path,
    *,
    chunk_size: int = 1_500,
    overlap: int = 200,
) -> list[DocumentChunk]:
    documents = load_path(path)
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    return [
        DocumentChunk(
            id=chunk.id,
            text=chunk.text,
            metadata=_enrich_metadata(chunk.metadata, chunk.text),
        )
        for chunk in chunks
    ]


def _enrich_metadata(metadata: dict[str, object], text: str) -> dict[str, object]:
    enriched = dict(metadata)
    source_path = str(enriched.get("source_path", ""))
    filename = Path(source_path).name if source_path else enriched.get("file_name")
    enriched.setdefault("filename", filename)
    enriched.setdefault("document_type", enriched.get("file_type", "unknown"))
    section = enriched.get("page") or enriched.get("worksheet") or enriched.get("record_index")
    if section is not None:
        enriched.setdefault("page_or_section", section)
    framework, control_id = detect_framework_control(text, source_path)
    if framework:
        enriched.setdefault("framework", framework)
    if control_id:
        enriched.setdefault("control_id", control_id)
    return enriched


def detect_framework_control(text: str, source_path: str = "") -> tuple[str | None, str | None]:
    lower_path = source_path.lower()
    preferred: list[tuple[str, re.Pattern[str]]] = []
    for framework, pattern in FRAMEWORK_PATTERNS.items():
        path_matches_framework = framework.lower().split()[0] in lower_path
        path_matches_cis = "cis" in lower_path and framework == "CIS Controls"
        if path_matches_framework or path_matches_cis:
            preferred.append((framework, pattern))
    preferred.extend(item for item in FRAMEWORK_PATTERNS.items() if item not in preferred)
    for framework, pattern in preferred:
        match = pattern.search(text)
        if match:
            return framework, match.group(0)
    return None, None


def chunk_id_for_text(path: str | Path, text: str, index: int) -> str:
    return stable_id(Path(path).as_posix(), index, text)
