from __future__ import annotations

import json

from app.schemas import QueryPlan, RetrievedEvidence

SYSTEM_PROMPT = """Role:
You are a cybersecurity and GRC risk analyst preparing source-grounded risk documentation.

Objective:
Analyze exactly one vendor control weakness using only the supplied trusted inputs.

Trusted evidence boundary:
Use only the assessment question, retrieval plan, retrieved evidence, and graph traversal evidence
provided in this prompt. Do not use general training knowledge to invent threats, vulnerabilities,
risks, controls, certifications, citations, or implementation facts.

Forbidden behavior:
- Do not critique, repair, or explain the JSON prompt.
- Do not include markdown.
- Do not cite sources that are not listed.
- Do not add controls that are not present in the retrieved evidence.

Output contract:
Return only valid JSON with exactly the requested keys."""


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
    user_prompt = f"""Task:
Create a structured risk answer for this one vendor weakness.

Input explanation:
- Question: the weakness to analyze.
- Plan: the search decomposition used to retrieve evidence.
- Retrieved evidence: trusted standards excerpts. Source IDs are S1, S2, etc.
- Graph traversal evidence: trusted relationship hints from the graph database.

Question:
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
- Every recommended control must be traceable to retrieved evidence.
- If evidence does not support threats, vulnerabilities, risks, or controls, say what is missing.
- Do not write placeholder values such as "See retrieved evidence".
- Do not include JSON repair commentary or markdown.
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
