from __future__ import annotations

from secure_rag.schema import RetrievalHit

SYSTEM_PROMPT = """You are a local information-security control assistant.
Use only the retrieved source excerpts as evidence for the answer.
If the excerpts do not support a recommendation, say that the corpus has insufficient evidence.
Prefer concrete controls, implementation guidance, and cited source identifiers.
Only include a recommendation when the retrieved excerpt explicitly supports that measure for the
user's criteria or contains matching risk/control metadata.
Do not introduce controls, frameworks, technologies, or implementation details that are not present
in the retrieved source excerpts. For gaps, describe what evidence is missing without naming
uncited controls, technologies, or implementation examples. Keep rationale tied to source text and
metadata, not general security theory."""


def build_grounded_prompt(criteria: str, hits: list[RetrievalHit]) -> list[dict[str, str]]:
    sources = "\n\n".join(_format_hit(index, hit) for index, hit in enumerate(hits, start=1))
    user_prompt = f"""Input criteria:
{criteria}

Retrieved source excerpts:
{sources}

Return:
1. Recommended security measures or controls.
2. Why each measure applies to the input criteria.
3. Source citations using the shown source identifiers.
4. Whether the corpus evidence is sufficient or weak. Do not list uncited example controls,
technologies, or implementation details."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _format_hit(index: int, hit: RetrievalHit) -> str:
    metadata = hit.chunk.metadata
    source_path = metadata.get("source_path") or metadata.get("file_name") or "unknown-source"
    page = metadata.get("page")
    location = f"{source_path}" + (f"#page={page}" if page else "")
    return (
        f"[S{index}] score={hit.score:.3f} source={location} chunk_id={hit.chunk.id}\n"
        f"{hit.chunk.text}"
    )
