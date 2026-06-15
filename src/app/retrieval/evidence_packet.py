from __future__ import annotations

import re
from collections import Counter

from app.schemas import DocumentChunk, RetrievedEvidence

STOP_WORDS = {
    "and",
    "are",
    "assessment",
    "company",
    "control",
    "controls",
    "currently",
    "evidence",
    "from",
    "identify",
    "maturity",
    "medium",
    "nist",
    "question",
    "risk",
    "risks",
    "security",
    "solution",
    "standards",
    "threat",
    "threats",
    "vendor",
    "vulnerabilities",
    "vulnerability",
    "with",
}

MAX_PROMPT_EVIDENCE_ITEMS = 5
MAX_PROMPT_EVIDENCE_CHARS = 380


def select_prompt_evidence(
    question: str,
    evidence: list[RetrievedEvidence],
    *,
    max_items: int = MAX_PROMPT_EVIDENCE_ITEMS,
    max_chars: int = MAX_PROMPT_EVIDENCE_CHARS,
) -> list[RetrievedEvidence]:
    """Select the compact source packet that is safe to send to the LLM.

    Retrieval is allowed to be broad enough for recall. Prompt evidence is not:
    it should contain only the most relevant, readable, source-linked excerpts
    for the current weak answer.
    """

    if not evidence:
        return []
    scored = [
        (_prompt_relevance_score(question, hit), index, hit)
        for index, hit in enumerate(evidence)
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    selected: list[RetrievedEvidence] = []
    seen_sources: Counter[str] = Counter()
    for score, _index, hit in scored:
        if score <= 0:
            continue
        if _should_exclude_from_prompt(question, hit):
            continue
        source_key = str(
            hit.chunk.metadata.get("source_path")
            or hit.chunk.metadata.get("filename")
            or hit.source
        )
        if seen_sources[source_key] >= 3 and len(selected) >= 2:
            continue
        selected.append(_trim_evidence(hit, max_chars=max_chars))
        seen_sources[source_key] += 1
        if len(selected) >= max_items:
            break
    if selected:
        return selected
    fallback = evidence[: min(max_items, len(evidence))]
    return [_trim_evidence(hit, max_chars=max_chars) for hit in fallback]


def prompt_evidence_summary(
    question: str,
    retrieved_count: int,
    prompt_evidence: list[RetrievedEvidence],
) -> dict[str, object]:
    return {
        "question_focus_terms": sorted(_topic_terms(question))[:12],
        "retrieved_source_count": retrieved_count,
        "prompt_source_count": len(prompt_evidence),
        "selection_policy": (
            "Use retrieved evidence for recall, then send only the most relevant compact "
            "source excerpts to the model."
        ),
        "prompt_sources": [
            {
                "source_id": f"S{index}",
                "chunk_id": hit.chunk.id,
                "score": round(float(hit.score), 6),
                "source": hit.source,
                "metadata": {
                    key: hit.chunk.metadata.get(key)
                    for key in [
                        "filename",
                        "document_type",
                        "page_or_section",
                        "framework",
                        "control_id",
                        "source_path",
                    ]
                    if hit.chunk.metadata.get(key) is not None
                },
                "why_selected": _why_selected(question, hit),
                "text_preview": hit.chunk.text,
            }
            for index, hit in enumerate(prompt_evidence, 1)
        ],
    }


def _prompt_relevance_score(question: str, hit: RetrievedEvidence) -> float:
    text = " ".join(str(hit.chunk.text or "").split()).lower()
    terms = _topic_terms(question)
    overlap = len(terms & set(_tokens(text))) / max(len(terms), 1)
    score = float(hit.score) + overlap * 0.75
    if _has_control_anchor(text, hit):
        score += 0.35
    if _is_generic_governance_text(text):
        score -= 0.45
    if _is_wrong_scope(question, text):
        score -= 0.55
    if _topic_terms(question) and not terms & set(_tokens(text)):
        score -= 0.25
    return score


def _should_exclude_from_prompt(question: str, hit: RetrievedEvidence) -> bool:
    text = " ".join(str(hit.chunk.text or "").split()).lower()
    return _is_wrong_scope(question, text) or _is_generic_governance_text(text)


def _trim_evidence(hit: RetrievedEvidence, *, max_chars: int) -> RetrievedEvidence:
    clean = " ".join(str(hit.chunk.text or "").split())
    if len(clean) > max_chars:
        clean = f"{clean[:max_chars].rstrip()} [...]"
    chunk = DocumentChunk(
        id=hit.chunk.id,
        text=clean,
        metadata=dict(hit.chunk.metadata),
    )
    return hit.model_copy(update={"chunk": chunk})


def _why_selected(question: str, hit: RetrievedEvidence) -> str:
    terms = sorted(_topic_terms(question) & set(_tokens(hit.chunk.text)))[:5]
    anchors = []
    text = hit.chunk.text.lower()
    if _has_control_anchor(text, hit):
        anchors.append("contains a control reference")
    if terms:
        anchors.append(f"matches: {', '.join(terms)}")
    return "; ".join(anchors) if anchors else "highest ranked retrieved evidence"


def _topic_terms(text: str) -> set[str]:
    raw = set(_tokens(text))
    expanded = set(raw)
    if {"anti-malware", "antimalware", "malware"} & raw:
        expanded.update({"malware", "anti-malware", "antimalware", "endpoint", "malicious"})
    if {"recovery", "ransomware", "backup", "backups", "disaster"} & raw:
        expanded.update({"recovery", "restore", "restoration", "backup", "backups", "disaster"})
    if {"endpoint", "endpoints"} & raw:
        expanded.update({"endpoint", "endpoints", "malicious", "malware"})
    return {term for term in expanded if term not in STOP_WORDS and len(term) >= 3}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", text.lower())


def _has_control_anchor(text: str, hit: RetrievedEvidence) -> bool:
    metadata = hit.chunk.metadata
    if metadata.get("control_id"):
        return True
    return bool(
        re.search(
            r"\b(safeguard\s+\d|control\s+\d|scf\s+#|scf control|pr\.|rc\.|de\.|rs\.)",
            text,
            re.I,
        )
    )


def _is_generic_governance_text(text: str) -> bool:
    generic_phrases = [
        "business stakeholders and process owners",
        "associated with end domain capabilities",
        "statutory, regulatory and/or contractual obligations",
        "reasonably implement cybersecurity and data protection controls",
    ]
    return any(phrase in text for phrase in generic_phrases)


def _is_wrong_scope(question: str, text: str) -> bool:
    question_lower = question.lower()
    if "endpoint" in question_lower and "email server anti-malware" in text:
        return True
    return False
