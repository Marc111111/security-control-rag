from __future__ import annotations

from app.retrieval.graph_context import graph_context_prompt_text
from app.schemas import QueryPlan, RetrievedEvidence

MAX_EVIDENCE_CHARS = 520

SYSTEM_PROMPT = """Role:
You are a cybersecurity and GRC risk analyst preparing source-grounded risk documentation.

Objective:
Analyze exactly one vendor control weakness using only the supplied trusted inputs.

Trusted evidence boundary:
Use only the assessment question, retrieval plan, retrieved text evidence, and filtered graph hints
provided in this prompt. Retrieved text evidence is authoritative. Graph hints are secondary and
must be used only when they point to one of the listed source IDs.
Do not use general training knowledge to invent threats, vulnerabilities, risks, controls,
certifications, citations, or implementation facts.

Forbidden behavior:
- Do not critique, repair, or explain the JSON prompt.
- Do not include markdown.
- Do not cite sources that are not listed.
- Do not add controls that are not present in the retrieved evidence.
- Do not write background education or methodology commentary.

Output contract:
Return only valid JSON with exactly the requested keys.

Output discipline:
Be concise and surgical. Use short noun phrases and compact bullet-like JSON values. Do not explain
your method, do not add background education, and do not use adjectives unless they change the risk
meaning."""


def build_structured_answer_prompt(
    question: str,
    plan: QueryPlan,
    evidence: list[RetrievedEvidence],
    graph_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    evidence_text = "\n\n".join(
        _format_evidence(index, hit) for index, hit in enumerate(evidence, 1)
    )
    graph_text = graph_context_prompt_text(graph_rows)
    user_prompt = f"""Task:
Create a structured risk answer for this one vendor weakness.

Inputs:
- Question: one weak assessment answer to analyze.
- Search focus: what the retrieval layer looked for.
- Retrieved evidence: the selected trusted standards excerpts. Source IDs are S1, S2, etc.
- Filtered graph hints: secondary hints. Use them only when they cite S1, S2, etc.

Insufficient-evidence behavior:
If the selected evidence does not support a field, leave that field empty and explain the missing
source in missing_information. Do not fill gaps from general knowledge.

Question:
{question}

Search focus:
{_format_plan_focus(plan)}

Retrieved evidence:
{evidence_text}

Filtered graph hints:
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
- Keep threats, vulnerabilities, risks, and assumptions to maximum 3 items each.
- Keep recommended_controls to maximum 5 items.
- Keep risk_control_matrix to maximum 3 rows.
- Prefer 1-2 matrix rows when they cover the issue.
- Keep each matrix cell short: preferably under 18 words.
- Use crisp labels, not prose paragraphs, for threats, vulnerabilities, risks, gaps, and controls.
- Threat, vulnerability, and risk labels must reuse meaningful words from the question or selected
  evidence. Do not add plausible security labels that are not present in the source text.
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
    excerpt = _truncate_evidence(hit.chunk.text)
    return (
        f"[S{index}] score={hit.score:.3f} method={hit.retrieval_method} "
        f"source={source}{suffix} chunk_id={hit.chunk.id}\n{excerpt}"
    )


def _format_plan_focus(plan: QueryPlan) -> str:
    labels = []
    for sub_question in plan.sub_questions:
        labels.append(f"- {sub_question.label}: {sub_question.focus}")
    return "\n".join(labels)


def _truncate_evidence(text: str) -> str:
    clean = " ".join(str(text or "").split())
    if len(clean) <= MAX_EVIDENCE_CHARS:
        return clean
    return f"{clean[:MAX_EVIDENCE_CHARS].rstrip()} [...]"
