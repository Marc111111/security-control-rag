from __future__ import annotations

from secure_rag.schema import RetrievalHit

SYSTEM_PROMPT = """You are a local information-security control assistant.
Use only the retrieved source excerpts as evidence for the answer.
If the excerpts do not support a recommendation, say that the corpus has insufficient evidence.
Prefer concrete controls with framework/control IDs, control names, and cited source identifiers.
Only include a recommendation when the retrieved excerpt explicitly supports that measure for the
user's criteria or contains matching risk/control metadata.
Do not collapse distinct source controls into a single recommendation. If multiple retrieved
excerpts contain concrete control identifiers or safeguard numbers, list each relevant control as
its own recommendation.
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
Write a clean, minimal answer in this exact style:

Recommended controls:
- <framework/control id and name>: <one sentence explaining why it fits>. [S#]
- <framework/control id and name>: <one sentence explaining why it fits>. [S#]

Notes:
- <only if needed, one short evidence limitation or practical note>

Rules:
- No preamble such as "Here is" or "Based on the excerpts".
- No numbered sections.
- No markdown bold markers.
- No risk essay unless the user asks for risks.
- Keep each bullet to one or two lines.
- Use source citations inline as [S1], [S2], etc.
- Do not list uncited example controls, technologies, or implementation details."""
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
