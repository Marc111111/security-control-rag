# Quality Gates For LLM Workflow Outputs

## Purpose

The complete assessment workflow must not treat LLM output as valid merely because it exists or
looks like JSON. Every LLM-generated output must be checked before it can become workflow state,
feed a later step, or be written to an application-facing result.

This document records the required design after the 2026-06-15 review of workflow step 17, where
the final paragraph model call returned irrelevant JSON-repair commentary and the parser silently
replaced it with generic fallback text. That behavior is not acceptable for cybersecurity, GRC, or
TPRM risk documentation.

Implementation status as of 2026-06-15:

- complete-assessment risk-answer LLM calls use prompt validation, strict parsing, output gates,
  and bounded repair retries;
- complete-assessment final-paragraph LLM calls use a deterministic validated fact packet,
  prompt validation, strict parsing, output gates, and bounded repair retries;
- final-paragraph prompts are built from a compact report fact packet, not the full internal
  validated object, and must include an explicit 120-word maximum per paragraph;
- final reports must name concrete standards/control references added by RAG when available, and
  must not hide them behind generic counts such as `7 controls`;
- final reports must avoid urgency or severity terms such as `critical`, `immediate`,
  `unacceptable`, or `severe` unless those terms are present in the validated fact packet;
- per-gap business storylines are now gated with `gap_storyline_output` so the workflow proves how
  each questionnaire gap maps to threats, vulnerabilities, risks, named controls, resilience, and
  residual concern before final report drafting;
- when final-paragraph model retries fail, the workflow may continue only through a visible
  rejection step followed by a deterministic report renderer that uses the same validated facts and
  must pass the same final paragraph gate;
- failed workflow/job gates are shown in the browser through a modal with operator and system-owner
  remediation guidance;
- OpenAI and Ollama generation temperature defaults are set to `0` to reduce variation between
  identical inputs;
- future LLM-powered workflows must adopt the same gate pattern before being considered reliable.

## Non-Negotiable Rules

- No silent fallback may be presented as a successful result.
- A deterministic fallback renderer is allowed only after the failed model drafts are visibly
  rejected and persisted in the workflow trace, and only if the renderer output passes the same
  final report gate.
- LLMs may help draft or structure text, but they must not decide hard facts unsupported by trusted
  input, RAG evidence, GraphRAG relationships, or deterministic Python extraction.
- Every LLM call needs a dedicated professional prompt-builder step.
- Every LLM call needs schema, content, relevance, and evidence quality gates.
- Failed gates must be visible in the workflow UI with validation errors and the offending output.
- Retries must be explicit and bounded. If the output still fails after retries, the workflow step
  must end as failed, not generic-successful.
- Final business paragraphs must be generated from a clean validated fact packet, not raw retrieval
  dumps, debug output, or previous malformed model responses.

## Controlled LLM Transaction Pattern

Every LLM call follows the same pattern:

1. Build a clean prompt from trusted structured inputs.
2. Validate the prompt contract before calling the model.
3. Call the selected model with low generation freedom.
4. Parse the model response.
5. Run quality gates.
6. Retry with a repair prompt that includes validation errors when the output is fixable.
7. Stop the workflow or mark the step failed when retries are exhausted.

The workflow UI must show these elements for every gated model call:

- input
- prompt generated
- raw model output
- parsed output
- quality gate result
- retry count
- final accepted output or failure reason

## Prompt Builder Requirements

A prompt builder is not a raw JSON dump. It must produce a clear instruction package with:

- role: the model's role for this call
- objective: the exact task
- input explanation: what the model receives and what each section means
- trusted evidence boundary: what may be used
- forbidden behavior: what must not be done
- output contract: exact required keys and types
- citation rules: how source IDs may be used
- failure behavior: what to return when evidence is insufficient

Model prompts must instruct the model to use only supplied assessment data, retrieved standards
evidence, graph facts, and validated fact packets. The model must not invent controls, threats,
risks, citations, certifications, or implementation facts from general training data.

## Required Gates By Workflow Area

### 1. Input Normalization Gate

Applies to simulated SQL input and future PostgreSQL adapters.

Checks:

- vendor context exists
- tier level and tier attributes exist
- questionnaire answers exist
- every answer has a linked control
- compliance and maturity values are valid enum values
- evidence descriptions are sanitized
- human-generated text cannot become model/system instructions

Failure behavior: stop before retrieval.

### 2. Finding Classification Gate

Applies after deterministic classification of questionnaire answers.

Checks:

- full compliance maps to strengths
- partial/no compliance maps to weaknesses
- every weakness preserves question ID, control ID, maturity, vendor comment, analyst comment, and
  tier context
- no weak finding loses its source question

Failure behavior: stop before query generation.

### 3. Risk Query Builder Gate

Applies before RAG/GraphRAG retrieval.

Checks each risk query includes:

- vendor tier
- question ID
- linked control ID and title
- weak answer summary
- vendor and analyst comments
- required analysis dimensions: gap, threat, vulnerability, risk, preventative controls,
  detective controls, corrective controls, recovery/response controls, and resilience impact

Failure behavior: rebuild deterministically or stop.

### 4. Retrieval Quality Gate

Applies after Qdrant, BM25, and graph traversal.

Checks:

- enough chunks were retrieved for the question
- chunks are relevant to the weakness
- source metadata includes source path and chunk ID
- evidence is not only generic framework boilerplate
- graph rows are filtered before prompt construction:
  - keep only rows whose `source_chunk_id` maps to one of the retrieved chunks
  - drop malformed entity labels and partial framework/risk-code fragments
  - present graph output as secondary hints, not authoritative text evidence
- source IDs are stable for later citation checks

Failure behavior: return an explicit insufficient-evidence result. Do not ask the model to guess.

### 5. Risk Answer Prompt Gate

Applies before each risk-answer LLM call.

Checks:

- prompt contains a role, objective, input explanation, trusted evidence boundary, forbidden
  behavior, output contract, citation rules, and insufficient-evidence behavior
- prompt includes only the selected weak finding and trusted evidence for that finding
- prompt does not include raw debug dumps unrelated to the task

Failure behavior: stop before the model call.

### 5a. Workflow Payload Hygiene Gate

Applies to every visible workflow step before the step is accepted as part of the chain.

Checks:

- step input and output are bounded enough to be readable and safe to pass forward
- normal workflow handoff does not contain full debug objects
- normal workflow handoff does not contain full prompt messages
- normal workflow handoff does not contain full raw model responses
- model prompt summaries stay under a configured size ceiling

Failure behavior: stop immediately with a human-readable explanation. The operator should not
approve the run. The solution owner must compact the failed step so it passes only business facts,
source IDs, short source previews, status, and quality-gate summaries. Full prompts, full raw
responses, and retrieval debug data belong in explicit debug logs or full-detail views, not in the
step-to-step data contract.

### 6. Risk Answer Output Gate

Applies after each risk-answer LLM call.

Schema checks:

- valid JSON
- required keys present
- arrays are arrays
- matrix rows include gap, threat, vulnerability, risk, likelihood, impact, controls, and evidence

Content checks:

- respects the risk-answer style contract: concise labels and phrases, not prose essays
- threats, vulnerabilities, risks, and assumptions stay within configured item limits
- matrix rows stay within the configured row limit
- matrix cells are short business phrases
- no placeholder final values such as `See retrieved evidence`
- no malformed fragments or repeated garbage text
- no JSON-repair commentary, markdown essay, or meta-analysis
- threats, vulnerabilities, risks, and controls reference the actual vendor gap
- controls are concrete and actionable

Evidence checks:

- every citation ID exists in retrieved evidence
- every recommended control appears in retrieved evidence or deterministic control extraction
- no invented frameworks, controls, or citations

Failure behavior: retry with a repair prompt containing validation errors. If still invalid after
the retry limit, mark the risk answer step failed.

### 7. Validated Fact Packet Gate

Applies after all accepted risk answers.

This deterministic Python step creates the only packet allowed to feed final reporting.

Checks:

- every fact links back to a question ID
- every risk links back to a validated weakness
- every control links back to evidence
- failed or insufficient-evidence model output is not promoted to fact
- matrix rows and summary lists are internally consistent

Failure behavior: stop before final report drafting.

### 8. Per-Gap Business Storyline Gate

Applies after deterministic risk-chain construction and before final report drafting.

Prompt checks:

- prompt receives exactly one compact validated risk chain;
- prompt includes role, objective, trusted fact boundary, forbidden behavior, output contract, and
  short-field limits;
- prompt forbids new threats, vulnerabilities, risks, controls, citations, assumptions, and
  unsupported urgency wording;
- prompt asks for only: question ID, gap story, business meaning, risk logic, control logic,
  resilience logic, and residual conclusion.

Output checks:

- valid JSON with exactly the required storyline fields;
- question ID matches the selected chain;
- every field is concise and reviewer-readable;
- storyline reuses the validated gap, threat, vulnerability, risk, named control, and
  resilience/residual concern;
- no generic "security should improve" wording passes without the validated chain content;
- no unsupported urgency or severity language.

Failure behavior: retry with validation errors. If still invalid, visibly reject the model drafts.
The workflow may render a deterministic storyline from the validated chain only, and that output
must pass this same gate before the storyline can feed final reporting.

### 9. Final Paragraph Prompt Gate

Applies before the final report-writing LLM call.

Checks:

- prompt is written for a business-facing TPRM report writer
- prompt explains that facts have already been validated
- prompt forbids JSON repair, re-analysis, new controls, new risks, new citations, and new evidence
- prompt includes only the compact validated report fact packet plus minimal vendor/tier context
- output contract requires exactly: management summary, introduction, objective, risk exposure,
  and conclusion
- style contract requires 2-4 sentences and at most 120 words per paragraph
- prompt requires risk exposure or conclusion to name the most important standards/control
  references from the risk chains or toolchain delta
- prompt requires evidence-calibrated wording: distinguish a missing control from missing evidence,
  and avoid urgency/severity claims unless the fact packet supports them

Failure behavior: stop before the model call.

### 10. Final Paragraph Output Gate

Applies after the final report-writing LLM call.

Schema checks:

- valid JSON
- exactly the required paragraph keys
- every value is a non-empty string

Content checks:

- paragraphs stay within the configured 120-word length limit
- mentions the actual vendor
- mentions the tier level where relevant
- mentions the real weaknesses
- risk exposure reflects validated risks
- conclusion aligns with validated facts
- no generic fallback text
- no JSON critique or repair commentary

Consistency checks:

- no new risks outside the validated fact packet
- no new controls outside the validated fact packet
- no unsupported assurance statement such as `acceptable risk` unless validated facts support it
- no unsupported urgency or severity wording
- risk exposure or conclusion names at least one standards/control reference added by RAG when
  such references exist

Failure behavior: retry with validation errors. If still invalid, visibly reject the model drafts.
The workflow may then render paragraphs deterministically from the validated fact packet. The
deterministic output must pass this same gate; otherwise the workflow fails and no successful final
result is produced.

## Output Budgeting And Style

Every model call must have both:

- a generation cap enforced by the model provider where possible (`max_output_tokens` for OpenAI,
  `num_predict` for Ollama);
- a prompt-level style contract that says what kind of output is expected.

Risk-analysis calls should be surgical:

- compact JSON only
- short labels and phrases
- maximum-value facts first
- no explanation of the method
- no background education
- no adjective-heavy prose

Final report calls may use prose, but only controlled prose:

- 2-4 sentences per paragraph
- maximum 120 words per paragraph
- most important finding first
- name concrete added controls where available
- use evidence-calibrated language
- no filler
- no methodology narration

Token gates prevent cost/runaway model-call size. Payload hygiene gates prevent oversized step
handoffs. Output style gates prevent technically valid but useless rambling.

## API-Driven Step Audit

The workflow now has a repeatable API audit command:

```powershell
$env:PYTHONPATH="D:\projects\mike-test\src"
python scripts\audit_complete_assessment_workflow.py --base-url http://127.0.0.1:8000 --provider ollama --model qwen3:14b --top-k 8
```

The auditor starts the real async HTTP job unless `--latest` or `--run-id` is supplied. It then
checks every visible workflow step for:

- clean previous-output to next-input handoff, or a recognized controlled derivation;
- payload hygiene and absence of debug/raw prompt leakage;
- compact selected prompt evidence after broad retrieval;
- visible prompt output contract;
- accepted structured risk answers with controls and matrix rows;
- completed workflow status. A failed job is an audit failure even if the partial steps are clean.

This is the operator/developer safety net for the exact issue observed in development: the user
should not have to inspect a bloated step manually to discover prompt drift or garbage handoffs.

## Evidence Curation Before LLM Calls

Retrieval and prompting are deliberately separated:

- Qdrant/BM25/Neo4j retrieval may return a broader set for recall and debug review.
- `select_prompt_evidence` chooses only the compact, most relevant, source-linked excerpts for the
  model.
- Wrong-scope and generic governance snippets are excluded from prompt evidence when better direct
  evidence exists.
- Prompt evidence is capped to a small number of excerpts, and prompt-quality gates reject oversized
  risk prompts before a model call.

This keeps the model call focused while preserving debug visibility into what retrieval found.

## Grounding And Pruning

Risk-answer validation now checks that threat, vulnerability, and risk labels reuse meaningful
terms from the assessment question or selected source evidence. The workflow also performs a
deterministic pruning pass before validation: unsupported extra labels or matrix rows are removed,
but the answer must still retain enough supported content to pass. This avoids accepting plausible
but unsupported security language while also avoiding a full workflow failure for one extra model
label when the core answer is grounded.

Final report validation also rejects unsupported acceptance-threshold language, for example
`acceptable risk`, unless the validated fact packet explicitly contains the threshold and decision
basis.

## Retry Policy

Default retry policy:

- attempt 1: normal professional prompt
- attempt 2: repair prompt with validation errors and the same trusted input
- optional attempt 3: strict repair-only prompt with smaller output scope

Retries must not add new evidence or broaden the model's freedom. They may only ask the model to
correct the response according to the same source material and validation errors.

If final report repair attempts still fail, the model output must be rejected in a visible workflow
step. A deterministic renderer may then phrase the final sections from the already validated risk
model. This is not a generic fallback: it must use the vendor, tier, confirmed gaps, standards
requirements, risk chains, toolchain delta, residual concern, and missing information already
present in the validated fact packet, and it must pass the final paragraph output gate.

## Runtime Quality Gate Result Shape

Quality gates should return structured results similar to:

```json
{
  "passed": false,
  "severity": "blocking",
  "gate": "final_paragraph_output",
  "errors": [
    {
      "field": "risk_exposure",
      "message": "Paragraph is generic and does not mention the validated anti-malware or disaster recovery weaknesses."
    }
  ]
}
```

These results must be persisted in the workflow run and visible in the UI.

## Step 17 Failure Lesson

The 2026-06-15 run showed that final report drafting was fed a messy package containing malformed
risk-answer content and debug-like evidence. The model responded by critiquing and repairing JSON
instead of drafting business paragraphs. The parser then silently substituted generic fallback
paragraphs.

That exact failure mode must be prevented by:

- clean validated fact packet before final report drafting
- compact report fact packet before final report prompting
- stronger final paragraph prompt
- strict paragraph output gate
- no silent parser fallback
- visible failed status when validation fails
