from __future__ import annotations

import json

from app.schemas import QueryPlan, RetrievedEvidence

SYSTEM_PROMPT = """You are a cybersecurity and GRC risk documentation assistant.
Use retrieved evidence first. Never invent citations. If a row is based on general reasoning,
leave evidence empty and explain the distinction in general_model_reasoning.
Return only valid JSON with the requested keys."""


def build_structured_answer_prompt(
    question: str,
    plan: QueryPlan,
    evidence: list[RetrievedEvidence],
    graph_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    evidence_text = "\n\n".join(
        _format_evidence(index, hit) for index, hit in enumerate(evidence, 1)
    )
    graph_text = json.dumps(graph_rows[:20], ensure_ascii=True, indent=2)
    user_prompt = f"""Question:
{question}

Plan:
{plan.model_dump_json(indent=2)}

Retrieved evidence:
{evidence_text}

Graph traversal evidence:
{graph_text}

Return exactly this JSON shape:
{{
  "executive_summary": "",
  "assumptions": [],
  "threats": [],
  "vulnerabilities": [],
  "risks": [],
  "recommended_controls": [],
  "risk_control_matrix": [
    {{
      "gap": "",
      "threat": "",
      "vulnerability": "",
      "risk": "",
      "likelihood": "low|medium|high|unknown",
      "impact": "low|medium|high|unknown",
      "controls": [],
      "evidence": ["S1"]
    }}
  ],
  "missing_information": [],
  "source_citations": [],
  "from_retrieved_evidence": "",
  "general_model_reasoning": ""
}}

Rules:
- Cite evidence as S1, S2, etc. only when the cited source is listed above.
- Put standards/control IDs in recommended_controls when evidence contains them.
- If evidence does not support threats, vulnerabilities, risks, or controls, say what is missing.
- Keep the answer concise and suitable for direct insertion into a GRC/risk record."""
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]


def _format_evidence(index: int, hit: RetrievedEvidence) -> str:
    metadata = hit.chunk.metadata
    source = metadata.get("source_path") or metadata.get("filename") or hit.source
    location = (
        metadata.get("page_or_section")
        or metadata.get("page")
        or metadata.get("record_index")
    )
    suffix = f" location={location}" if location is not None else ""
    return (
        f"[S{index}] score={hit.score:.3f} method={hit.retrieval_method} "
        f"source={source}{suffix} chunk_id={hit.chunk.id}\n{hit.chunk.text}"
    )
