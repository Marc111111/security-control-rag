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
- failed workflow/job gates are shown in the browser through a modal with operator and system-owner
  remediation guidance;
- OpenAI and Ollama generation temperature defaults are set to `0` to reduce variation between
  identical inputs;
- future LLM-powered workflows must adopt the same gate pattern before being considered reliable.

## Non-Negotiable Rules

- No silent fallback may be presented as a successful result.
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
- graph rows are relevant to the same topic before being sent to a prompt
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

### 6. Risk Answer Output Gate

Applies after each risk-answer LLM call.

Schema checks:

- valid JSON
- required keys present
- arrays are arrays
- matrix rows include gap, threat, vulnerability, risk, likelihood, impact, controls, and evidence

Content checks:

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

### 8. Final Paragraph Prompt Gate

Applies before the final report-writing LLM call.

Checks:

- prompt is written for a business-facing TPRM report writer
- prompt explains that facts have already been validated
- prompt forbids JSON repair, re-analysis, new controls, new risks, new citations, and new evidence
- prompt includes only the validated fact packet plus minimal vendor/tier context
- output contract requires exactly: management summary, introduction, objective, risk exposure,
  and conclusion

Failure behavior: stop before the model call.

### 9. Final Paragraph Output Gate

Applies after the final report-writing LLM call.

Schema checks:

- valid JSON
- exactly the required paragraph keys
- every value is a non-empty string

Content checks:

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

Failure behavior: retry with validation errors. If still invalid, mark the final paragraph step
failed and do not produce a successful final result.

## Retry Policy

Default retry policy:

- attempt 1: normal professional prompt
- attempt 2: repair prompt with validation errors and the same trusted input
- optional attempt 3: strict repair-only prompt with smaller output scope

Retries must not add new evidence or broaden the model's freedom. They may only ask the model to
correct the response according to the same source material and validation errors.

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
- stronger final paragraph prompt
- strict paragraph output gate
- no silent parser fallback
- visible failed status when validation fails
