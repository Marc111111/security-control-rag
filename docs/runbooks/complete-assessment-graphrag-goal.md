# Complete Assessment GraphRAG Goal Runbook

Last updated: 2026-06-15.

## Goal

Implement a real local-first cybersecurity/GRC workflow for the Foundation/Risk Assessment
prototype. The only simulated part should be the initial PostgreSQL input. Everything after that
must be real application logic: source normalization, sanitization, deterministic classification,
GraphRAG retrieval, Qdrant, BM25, Neo4j, LLM calls, structured result assembly, run persistence,
tests, and a workflow UI/API.

## Current Architecture Direction

- UI route: `GET /mock/foundation`.
- Browser workflow path:
  - `POST /api/workflows/complete-assessment/preflight`
  - `POST /api/workflows/complete-assessment/jobs`
  - `GET /api/workflows/complete-assessment/jobs/{job_id}`
  - `POST /api/workflows/complete-assessment/jobs/{job_id}/cancel`
- Compatibility/debug workflow endpoint: `POST /api/workflows/complete-assessment/run`.
- Saved run list: `GET /api/workflows/complete-assessment/runs`.
- Saved run detail: `GET /api/workflows/complete-assessment/runs/{run_id}`.
- Source input boundary: `input_source.adapter + input_source.payload`.
- Current adapters:
  - `foundation_packet_v1`: accepts the canonical assessment packet directly.
  - `simulated_postgres_v1`: accepts a simulated PostgreSQL row, `rows[0]`, or `packet` wrapper.
- Normalized internal contract: `FoundationAssessmentPacket`.
- Retrieval stack: Qdrant dense retrieval, persistent BM25 keyword index, Neo4j graph traversal.
- Qdrant image: `qdrant/qdrant:v1.18.0`, aligned with `qdrant-client==1.18.0`.
- Local generation model: `qwen3:14b` through Ollama.
- Embeddings: `mxbai-embed-large` through Ollama.
- Guarded external models: `gpt-5.4-mini`, `gpt-5.4`, `gpt-5.5`, `gpt-4.1-mini`.

## What Is Real

- Document ingestion through `GraphRagPipeline.ingest`.
- Dense vector store through Qdrant when `GRAPHRAG_VECTOR_BACKEND=qdrant`.
- Keyword retrieval through a persisted BM25 JSONL index at `third_party/keyword_index/chunks.jsonl`.
- Graph extraction and graph storage through Neo4j when `GRAPHRAG_GRAPH_BACKEND=neo4j`.
- Planner-driven sub-question retrieval for each weak questionnaire answer.
- Selected model is used for both GraphRAG answer synthesis and final business prose drafting.
- Result contract assembly is deterministic Python code, not LLM-controlled JSON structure.
- The browser workflow must show a transparent step chain. Do not collapse RAG planning, standards
  retrieval, graph lookup, or model calls into one broad tab. Each visible tab should show input,
  plain-language process explanation, and output.
- The job status endpoint returns `partial_steps` while the workflow is running so the UI can show
  which steps finished and which step is currently active.
- The optional simulated DB input form on `/mock/foundation` edits the canonical packet JSON before
  the first adapter step. It supports applying changes, browser-local scenario save, and reset to
  the initial sample. Field meanings are documented in `docs/business-context.md` and served at
  `/mock/foundation/business-context`.
- Exact recommended controls are post-processed from retrieved evidence text, because local models
  can misread framework version numbers as control IDs.
- Run output is saved under `data/workflow_runs`.
- Request-scoped OpenAI API key is accepted but never persisted or returned.
- Browser runs are async. The UI preflights token/cost estimates before model calls, starts a
  background job, polls status, and can cancel running jobs.
- Preflight now also defines a workflow token cap. The estimate is cumulative for one complete
  workflow run across all LLM calls and is deliberately conservative. Default behavior is to enforce
  the estimate plus `token_budget_tolerance_percent` (10% by default), compare it with actual
  provider usage where available, and return per-call token usage in `token_budget.calls`.
- Local Ollama models show token counts with zero API price. External OpenAI preflight/run data may
  include USD and EUR metadata using the configured model price table and USD-to-EUR rate, but the
  Configuration panel shows USD only to keep the estimate compact.
- UI workflow steps must remain a readable handoff chain: previous output becomes next input.
  Multi-answer loops need explicit selection and storage steps so the chain does not appear to jump.
- Each Input, Process, and Output preview has an Expand button that opens a separate inspection
  window. Closing that window must not affect the workflow run or tab open/closed state.
- The Final Result Contract panel has its own Expand button plus a development-only "Open Codex
  review packet" button. The packet contains initial input, run metadata, workflow summaries, final
  result, and a review rubric. It is not an automatic Codex API call; it is a handoff packet for the
  Codex desktop agent unless a real authorized review endpoint is added later.
- The workflow UI has a manual "Estimate cost" button at the bottom of the Configuration panel. It
  preflights the current edited input and selected model without starting a run, calling a model, or
  requiring an API key. Dirty optional-form edits are synced into the request before the estimate is
  generated. The estimate display must include input tokens, output tokens, total tokens, token cap,
  LLM calls, and USD cost where applicable.
- Workflow step tabs should be created closed by default during polling. Preserve user open/closed
  state across polling updates. Do not auto-open the current or newest step.
- Running jobs show ETA from the median duration of previous completed saved runs, preferring the
  same provider/model when available. If there is no duration history, show that honestly.
- Cancelling an Ollama job calls `ollama stop <model>` immediately, reports `cancelling` while any
  in-flight call unwinds, then calls `ollama stop <model>` again before reporting `cancelled`.
- Job-scoped Ollama calls stream responses and pass `keep_alive=0s`; this lets cancellation be
  observed between chunks and avoids keeping the generator resident after successful calls.

## What Is Simulated

- The first PostgreSQL read is simulated. This must remain isolated behind the source adapter so
  production SQL/view shapes can replace it later without changing the rest of the chain.
- The current UI starts from a sample JSON payload returned by `GET /api/mock/foundation-packet`.

## Commands

Install/update Python dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Start third-party services:

```powershell
docker compose up -d
```

Pull Ollama models:

```powershell
ollama pull qwen3:14b
ollama pull mxbai-embed-large
```

Ingest standards into Qdrant/BM25/Neo4j:

```powershell
python scripts\ingest_graphrag.py standards
```

Run tests:

```powershell
python -m ruff check .
python -m pytest
```

Start API/UI:

```powershell
grc-graphrag-api
```

Open:

```text
http://127.0.0.1:8000/mock/foundation
```

## Quality Rules

- Never invent citations.
- Separate retrieved evidence from general model reasoning.
- If evidence is missing, say what is missing.
- Keep the LLM out of persistence and schema decisions.
- Every LLM call must have a dedicated professional prompt-builder step. Do not send a raw random
  JSON blob and expect the model to infer its job.
- Every LLM output must pass schema, content, relevance, and citation/evidence gates before it can
  feed the next step.
- No silent generic fallback may be shown as a successful result. Failed LLM output must be visible
  as a failed quality gate with validation errors.
- Final report paragraphs must be drafted from a deterministic validated fact packet, not raw
  retrieval dumps or malformed prior model responses.
- Keep simulated SQL source format isolated behind input adapters.
- Every workflow step shown in the UI must include input, process, and output.
- Large step outputs should be persisted and available for full preview.
- The detailed gate design is in `docs/quality-gates.md`.

## Current Work Status

- Input adapter layer: implemented.
- Workflow-wide model routing: implemented.
- New vertical workflow UI: implemented.
- Async job polling and Cancel Run button: implemented.
- Preflight token/cost estimate before model calls: implemented.
- Token budget enforcement and estimate-vs-actual reporting: implemented.
- Workflow handoff steps and separate preview windows: implemented.
- Run persistence: implemented.
- Unit tests for complete workflow and parser normalization: implemented.
- Quality-gate design documentation: completed in `docs/quality-gates.md`.
- Runtime quality-gate implementation: implemented for risk-answer model calls and final
  paragraph model calls. Gates validate prompt contracts, schema/content/evidence quality, and
  final report consistency; failed gates retry and then fail visibly.
- Browser failure modal: implemented for workflow/preflight/job failures with operator and
  system-owner remediation guidance.
- Generation temperature: set to `0` for OpenAI and Ollama clients to reduce variation between
  identical inputs.
- Third-party service folders under `third_party/`: configured in Docker Compose.
- Standards ingestion into Qdrant/BM25/Neo4j: completed.
- Store counts after clean ingestion:
  - Qdrant: 22,800 chunks.
  - BM25 JSONL: 22,800 unique chunks.
  - Neo4j: 2,800 nodes and 14,021 relationships.
- Last successful local workflow run:
  - `data/workflow_runs/run-2026-06-14T215427209749+0000-ba59a442/run.json`
  - model `qwen3:14b`
  - 5 workflow steps
  - 2 risk evaluations
  - anti-malware gap produced CIS Safeguard 10.1-10.5 and SCF END-04 controls
  - DR/business-continuity gap produced SCF BCD recovery controls
- Browser verification in this goal run:
  - `http://127.0.0.1:8000/mock/foundation` showed `Preflight estimate` plus
    `Background workflow job`.
  - Run -> Cancel moved the job to `cancelled` and re-enabled Run.
  - `ollama ps` after cancellation showed no `qwen3:14b` resident; GPU utilization was 0%.
  - The Configuration panel must show the whole estimate block inside the top-left frame without
    clipping. On 2026-06-15 the live page was verified after pressing `Estimate cost`: estimate
    bottom `458px`, optional DB form top `477px`, panel scroll height equaled client height.
    Keep estimate content visible, readable, and on one line. It must show exactly `Model`,
    `Expected in/out` with input and output token estimates, `Hard cap`, and `Total cost` in USD.
    Do not show mini cards, total-token clutter, LLM call count, EUR, or a redundant `Status OK`
    chip in that top-frame estimate.
- Commit/push/PR update: record compact Configuration panel changes in git and PR comments.

## 2026-06-15 Step 17 Quality Finding

The newest inspected run was:

```text
data/workflow_runs/run-2026-06-15T090956203951+0000-fff55270/run.json
```

Step 17, `Ask model to draft report paragraphs`, failed semantically. The raw `qwen3:14b` output
started critiquing/repairing JSON instead of drafting the requested report paragraphs. The parser
then produced generic fallback paragraphs such as `Risk exposure is driven by weak questionnaire
responses and retrieved evidence.`

This run must not be treated as a trustworthy final report. It demonstrates the required next
implementation work:

- add prompt-builder steps before every LLM call,
- add output quality gates after every LLM call,
- add bounded repair retries,
- mark exhausted failures visibly instead of continuing,
- build a deterministic validated fact packet before final paragraph generation,
- remove or quarantine silent fallback behavior from final report generation.

These items are now implemented for the complete-assessment risk-answer and final-paragraph LLM
calls. Remaining expansion work is to apply the same gate pattern to any other future LLM-powered
workflow and to improve the semantic validators as real analyst feedback accumulates.

## 2026-06-15 Q2 Graph Noise Finding

A failed `Q2 / PR.PS-01 Endpoint protection` run showed that text retrieval worked correctly: the
top evidence included CIS malware defenses and SCF endpoint anti-malware controls. The workflow
stopped because the local model could not produce a complete risk/control matrix after repair
attempts.

Root cause:

- graph lookup matched loose query terms, including generic words such as `risk` and `control`;
- raw graph rows were not restricted to relationships anchored to the retrieved evidence chunks;
- heuristic graph extraction had created malformed entity labels from dense standards text, such as
  partial risk-code fragments;
- the prompt treated raw graph rows as trusted graph evidence, which distracted the local model from
  the cleaner standards excerpts.

Implemented correction:

- Qdrant/BM25 text evidence is now the anchor for model prompts.
- Graph rows are filtered before prompt construction and retained only when their `source_chunk_id`
  maps to one of the retrieved source chunks.
- Malformed graph entity labels are dropped before the prompt is built.
- The prompt now calls graph rows `filtered graph hints`; retrieved text evidence is explicitly
  authoritative.
- The browser failure modal now explains the failed risk matrix in human language and keeps raw
  technical error details collapsed.

## 2026-06-15 Prompt And Handoff Bloat Finding

A later failed run showed a separate workflow hygiene issue:

- the risk-answer prompt included a full repeated planner JSON object plus long retrieved chunks;
- the visible workflow step stored both the prompt and retrieved evidence package, duplicating large
  data in the UI;
- the `Store risk answer` step carried the full `GraphRagAnswer.debug` payload forward, including
  prompt messages and retrieved chunks;
- the next step therefore started with a very large input that was not useful to a human reviewer;
- quality-gate warnings were treated as blocking failures, causing a usable first answer to be
  retried until the model produced a worse answer.

Implemented correction:

- risk-answer prompts now use a compact search-focus summary instead of full planner JSON;
- retrieved evidence sent to the model is excerpted per source;
- visible prompt steps show compact prompt previews and source summaries, not full debug dumps;
- normal workflow handoff state stores compact risk answers and compact sources only;
- full debug material remains in evaluation/debug logs, not in every step input/output;
- risk-answer warnings remain visible but no longer fail the workflow when there are no blocking
  issues.
- Ollama calls now receive `num_predict` from the configured output-token estimate, matching the
  OpenAI `max_output_tokens` behavior.
- Risk-answer prompts and validators enforce surgical output: short labels, limited list sizes,
  limited matrix rows, and no background prose.
- Final report prompts and validators allow prose only where useful: controlled 2-4 sentence
  paragraphs with the most important finding first.

## Known Risks And Improvements

- LLM output validation now exists for the complete-assessment risk-answer and final-paragraph
  calls. Continue improving semantic checks as new failure modes are observed.
- The graph extractor is heuristic. Graph prompt rows are now filtered, but the extractor should
  later be replaced or supplemented by a stronger controlled extractor with confidence scoring.
- BM25 persistence appends chunks. Re-ingestion should clear or version the keyword index to avoid
  duplicate keyword hits. Tests must use temporary `keyword_index_path` values and never write to
  `third_party/keyword_index/chunks.jsonl`.
- Metadata control-ID extraction is imperfect for some standards chunks. The workflow now extracts
  exact displayed control labels from evidence text, but metadata enrichment should still improve.
- Local `qwen3:14b` sometimes reports that threats/likelihood/impact are missing from evidence.
  This is preferable to invention, but better source curation and graph extraction can improve it.
- Reranking is lightweight. Add a local reranker when an available model is confirmed.
- The normalized assessment packet is a prototype contract. Production PostgreSQL can map into it
  through a new adapter, or the contract can version forward if product data changes.
- Cost estimates are conservative rough estimates for one complete run. They should be biased high
  so normal runs usually finish below the estimate. They are sufficient for comparison and budget
  protection, not billing.
