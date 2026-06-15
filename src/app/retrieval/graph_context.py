from __future__ import annotations

import re

from app.schemas import RetrievedEvidence


def build_graph_context_rows(
    graph_rows: list[dict[str, object]],
    evidence: list[RetrievedEvidence],
    *,
    max_rows: int = 8,
) -> list[dict[str, object]]:
    """Return graph hints that are safe and readable enough for an LLM prompt.

    Raw graph rows are intentionally treated as hints, not facts. They are kept
    only when they can be traced back to a retrieved chunk already shown to the
    model as source evidence.
    """

    source_ids_by_chunk = {hit.chunk.id: f"S{index}" for index, hit in enumerate(evidence, 1)}
    cleaned: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in graph_rows:
        chunk_id = str(row.get("source_chunk_id") or "")
        source_id = source_ids_by_chunk.get(chunk_id)
        if not source_id:
            continue
        source = _entity(row.get("source"))
        target = _entity(row.get("target"))
        relationship = str(row.get("relationship") or "").strip()
        if not source or not target or not relationship:
            continue
        if _is_noisy_name(source["name"]) or _is_noisy_name(target["name"]):
            continue
        key = (source["name"].lower(), relationship, target["name"].lower(), source_id)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "evidence": source_id,
                "relationship": relationship,
                "source_type": source["type"],
                "source": source["name"],
                "target_type": target["type"],
                "target": target["name"],
            }
        )
        if len(cleaned) >= max_rows:
            break
    return cleaned


def graph_context_prompt_text(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "No validated graph hints were retained. Use the retrieved text evidence only."
    lines = []
    for row in rows:
        lines.append(
            "- "
            f"{row['evidence']}: {row['source_type']} '{row['source']}' "
            f"{row['relationship']} {row['target_type']} '{row['target']}'"
        )
    return "\n".join(lines)


def _entity(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    name = _clean_name(value.get("name"))
    entity_type = _clean_name(value.get("type"))
    if not name or not entity_type:
        return None
    return {"name": name, "type": entity_type}


def _clean_name(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip(" .:-|")
    return text


def _is_noisy_name(value: str) -> bool:
    clean = value.strip()
    if len(clean) < 3 or len(clean) > 90:
        return True
    lower = clean.lower()
    if re.search(r"\b(\w+)(?:\s+\1\b){3,}", lower):
        return True
    alpha_ratio = sum(ch.isalpha() for ch in clean) / max(len(clean), 1)
    if alpha_ratio < 0.45:
        return True
    if re.search(r"\brisk[-:]?\w*\b", lower) and re.search(
        r"\b(?:[a-z]{1,3}\d|\d+[a-z])\b",
        lower,
    ):
        return True
    if lower.startswith("based program that supports"):
        return True
    return False
