from __future__ import annotations

from secure_rag.schema import RetrievalHit

SYSTEM_PROMPT = """You are a local information-security control assistant.
Use only the retrieved source excerpts as evidence for the answer.
If the excerpts do not support a recommendation, say that the corpus has insufficient evidence.
Prefer concrete controls, implementation guidance, and cited source identifiers.
Do not rely on general model knowledge unless you clearly label it as an uncited gap."""


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
4. Gaps or assumptions where the corpus is weak."""
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

