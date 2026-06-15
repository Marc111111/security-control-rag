# Agent Continuity Notes

This file is required reading for future agents and context rollovers. It captures the durable
project objective, architecture decisions, commands, and current implementation state.

## Objective

Build a local Python RAG system that uses a private information-security documentation base
(PDF, Word, structured data, and similar sources) to generate security measures or controls
related to input criteria such as risks, vulnerabilities, tiers, maturity levels, and control
framework mappings.

The system should answer primarily from the user's corpus, not from generic LLM training data.
When evidence is weak, it should say so instead of inventing controls.

## Current Goal And Runbook

Active implementation goal: replace the mock/legacy path with a real end-to-end assessment
workflow that uses simulated PostgreSQL input only at the first boundary, then runs real
normalization, sanitization, GraphRAG retrieval, Qdrant, BM25, Neo4j, selected LLM calls,
structured result assembly, run persistence, tests, and a clean workflow UI.

Read `docs/runbooks/complete-assessment-graphrag-goal.md` before continuing any context rollover.
That runbook tracks what is real, what is simulated, what worked, what did not, and what still
needs improvement.

The source SQL/result shape must remain replaceable. The workflow API accepts an `input_source`
object with an adapter name and payload. The adapter normalizes the incoming source shape into the
canonical assessment packet; the rest of the chain must not depend on the simulated SQL format.

## Current Architecture

The current project is organized around five building blocks:

1. Knowledge ingestion pipeline for PDFs, Word files, text, CSV, JSON, YAML, and Excel.
2. Security knowledge schema for chunks, controls, risks, vulnerabilities, tiers, and provenance.
3. GraphRAG retrieval using Qdrant dense vectors, persistent BM25 keyword search, and Neo4j graph
   traversal.
4. LLM orchestration through Ollama or guarded OpenAI calls with source-grounded prompts.
5. Governance layer: this file, architecture docs, tests, GitHub repository, and small PRs.

Details live in `docs/architecture.md` and decisions live in `docs/decisions/`.

## Decisions

- Language: Python 3.11.
- Local generation model: `qwen3:14b` through Ollama. The machine initially had `gemma3:4b`,
  which was enough to prove the pipeline but produced average reasoning and inconsistent answer
  shape on security-control recommendation tasks. Fallback candidates are `gemma3:12b`,
  `qwen3.5:9b`, or a smaller Llama-family model if VRAM pressure is a problem.
- Embedding model: start with `mxbai-embed-large` through Ollama; fallback option is
  `nomic-embed-text`. Keep `mxbai-embed-large` for now unless retrieval evidence proves it is
  the bottleneck.
- Vector store: Qdrant is the default for the advanced workflow. Persistent BM25 lives at
  `third_party/keyword_index/chunks.jsonl`. Neo4j Community stores extracted graph entities and
  relationships.
- Legacy Chroma is decommissioned for new workflow work. `src/secure_rag` remains only for
  backwards compatibility and old tests; do not build new features on it.
- API and UI delivery are required. The user does not want to operate the system through a CLI.
- A simple local web UI and HTTP API are required because the user does not want to operate the
  system through a CLI.
- Private corpora, generated vector stores, and model blobs are not committed to Git.
- The answer contract must become structured rather than free text. The model should produce a
  predictable schema and the UI should render that schema cleanly. Do not rely on the LLM to
  format long prose correctly.

## Local Model Handling

Ollama normally stores models outside the repository. For a project-local model cache, start
Ollama with `OLLAMA_MODELS` pointing at `.ollama/models` before pulling models. The helper script
`scripts/setup_ollama.ps1` documents and performs the expected pulls against the active Ollama
instance. The `.ollama/` folder is ignored because model files are large binary artifacts.

## Important Commands

```powershell
python -m pip install -e ".[dev]"
pytest
python -m ruff check .
.\scripts\setup_ollama.ps1
docker compose up -d
python scripts\ingest_graphrag.py standards
grc-graphrag-api
```

## Implementation Notes

- Keep answers grounded in retrieved chunks and include source metadata.
- Retrieval and prompt code must remain testable without a live Ollama server.
- Use `HashEmbeddingClient` and `MemoryVectorStore` for deterministic tests.
- Runtime adapters may depend on optional heavy libraries such as ChromaDB, pypdf, python-docx,
  and openpyxl.
- `POST /api/workflows/complete-assessment/run` is the current SaaS workflow entrypoint. It accepts
  `input_source` plus model settings and returns persisted workflow steps plus a deterministic
  result contract.
- The browser must use the async job endpoints, not the synchronous `run` endpoint:
  `preflight -> jobs -> poll jobs/{job_id} -> optional cancel`. This prevents long local
  `qwen3:14b` calls from freezing the UI and gives the user a Cancel Run button.
- Cancelling an Ollama-backed job calls `ollama stop <model>` to release GPU/VRAM. Always verify
  with `ollama ps` or `nvidia-smi` if a user reports GPU still being used. Running jobs pass through
  `cancelling` until any in-flight model call exits, then the worker calls `ollama stop <model>`
  again before reporting `cancelled`. Workflow-scoped Ollama calls stream responses and use
  `keep_alive=0s` to reduce cancellation latency and avoid the default keepalive residency.
- `POST /api/query` remains the lower-level GraphRAG natural-language query endpoint.
- The complete workflow must show each step's input, process, and output in the UI.
- The workflow trace must be visually honest. Do not hide several actions inside one broad step.
  RAG planning, Qdrant/BM25/Neo4j retrieval, and each model call must appear as individual
  workflow steps with plain-language explanations. The input of each step should be the previous
  step output, or the process text must explain why it changed.
- `/mock/foundation` includes an optional simulated DB input form. It edits the same canonical
  packet JSON that the workflow sends, can save a browser-local scenario, and can reset to the
  initial sample. Field wording must stay aligned with `docs/business-context.md`.
- Add tests with every new behavior.
- GitHub Actions CI is expected to run Ruff and pytest on pull requests.

## PR Roadmap

1. Bootstrap repository, documentation, core package, CLI, and tests.
2. Improve ingestion for real-world PDFs and Word documents.
3. Add retrieval quality evaluation with representative security-control fixtures.
4. Add API service for the existing app.
5. Add monitoring, corpus refresh, and answer-quality reports.

## Current State

Initial implementation is being built in `D:\projects\mike-test`.

First draft PR: `https://github.com/Marc111111/security-control-rag/pull/1`.

On 2026-06-13, the `standards/` folder was populated locally with NIST SP 800-53 Rev. 5,
NIST CSF 2.0, Secure Controls Framework 2026.x, and CIS Controls v8.1 material. The large
downloaded standards files are ignored by Git; `standards/SOURCES.md` documents the source URLs
and ingestion result. The local `storage/chroma` collection contains 22,801 records after standards
ingestion.

Also on 2026-06-13, a bad answer exposed three defects:

- The web UI was stateless, so follow-up prompts such as "I want concrete controls" lost the
  original ransomware-playbook context.
- The toy test fixture was mixed into the production Chroma collection and polluted answers.
- Semantic-only retrieval missed security-domain intent such as "playbook" meaning incident
  response plan, roles, communication, exercises, and response procedures.

Fixes applied:

- The UI now sends conversation history to `POST /api/query`.
- Retrieval filters `tests/fixtures` sources by default.
- Retrieval expands common security terms and reranks by keyword overlap plus standards-source
  priority.
- The live Chroma collection was cleaned to remove the three sample fixture chunks, leaving 22,798
  standards chunks.

Later on 2026-06-13, a second bad-answer review exposed a deeper product issue:

- The formatting was over-corrected to be minimal, which made some answers drop requested
  threat, vulnerability, and risk sections.
- The LLM is still free-texting. This makes output inconsistent even when retrieval finds useful
  standards excerpts.
- Retrieval is still shallow. It retrieves relevant excerpts, but it does not yet assemble a
  control recommendation packet across frameworks before asking the model to write.
- The main fix is not just a bigger local model. The durable fix is: stronger local model,
  enforced structured output contract, and a UI renderer that expects those sections.

The next implementation slice should add an enforced answer schema shaped like:

```json
{
  "recommended_controls": [],
  "related_threats": [],
  "related_vulnerabilities": [],
  "related_risks": [],
  "implementation_notes": [],
  "source_mappings": []
}
```

The API should return this structured object alongside the human-readable answer so the user's
future app can query the RAG system reliably through natural language or structured JSON criteria.

On 2026-06-14, the requested direction changed from a small single-shot RAG prototype toward an
open-source advanced RAG / GraphRAG prototype for cybersecurity and GRC risk documentation. The
new implementation lives under `src/app` and is intentionally modular:

- `app/main.py` FastAPI entrypoint.
- `app/config.py` environment-driven settings for Ollama/OpenAI, Qdrant, and Neo4j.
- `app/ingestion/` document loading/chunk enrichment.
- `app/retrieval/` Qdrant dense retrieval, BM25 keyword retrieval, merge/dedup/rerank.
- `app/graph/` heuristic entity and relationship extraction plus memory/Neo4j stores.
- `app/planning/` decomposition into exposure/gap, threats, vulnerabilities, risks, controls,
  and framework references.
- `app/generation/` structured answer prompt and Ollama/OpenAI chat clients.
- `app/evaluation/` query logs and manual relevance feedback.

The new API entrypoint is `grc-graphrag-api`. The old `security-rag-api` remains for compatibility.
Qdrant and Neo4j are defined in `docker-compose.yml`; `.env.example` documents local settings.

Also on 2026-06-14, a Foundation Assessment Summary Agent was added as the first SaaS-facing AI
prototype before full Risk Evaluation. It lives under `app/assessment/` and exposes
`POST /api/assessments/foundation-summary`.

Purpose:

- Accept canonical PostgreSQL-shaped vendor assessment data.
- Include vendor profile, tier level/attributes, questionnaire results, linked controls,
  vendor comments, analyst comments, compliance status, maturity, and evidence descriptions.
- Sanitize human-generated comments before LLM prompting.
- Classify full compliance as strengths and partial/no compliance as weaknesses.
- Generate Management Summary, Introduction, Objective, Key Findings, Strengths, Weaknesses,
  Risk Exposure, Conclusion, and Missing Information.
- Return `postgres_payload` for draft insertion; immutable snapshots remain an application
  decision after analyst approval.

This is the prototype layer. The deeper Risk Evaluation Agent should later consume the weaknesses
and run the GraphRAG gap -> threat -> vulnerability -> risk -> control workflow.

After that, a visible mockup was added at `/mock/foundation`. It uses simulated PostgreSQL input
and output and can run entirely through mock endpoints:

- `GET /api/mock/foundation-packet`
- `POST /api/mock/foundation-summary`

It also exposes token estimation through
`POST /api/assessments/foundation-summary/token-estimate` and a guarded OpenAI smoke test through
`POST /api/assessments/foundation-summary/openai-smoke-test`. The smoke test must require
`confirm_external_call=true` and must reject prompts above the configured token guard. Do not
remove that safety guard.

The mock screen now includes a runtime model selector backed by
`POST /api/assessments/foundation-summary/model-run`. It supports `mock`, `ollama`, and `openai`
providers. Every run must return `model_run.token_estimate` so the UI can show estimated request
price. OpenAI runs must remain blocked unless `confirm_external_call=true`.

The mock screen also accepts a request-scoped OpenAI API key. Never persist that key, include it in
debug payloads, or return it in API responses. `.env` remains the option for longer-lived local
development keys.

The mock screen should start clean and only render output after a run. It displays a vertical
execution workflow with expandable input/output previews for simulated PostgreSQL read, packet
assembly, sanitization, finding classification, token estimate, model call, parsing, and simulated
PostgreSQL write payload.

For the complete GraphRAG workflow, keep the visible steps as a handoff chain: a step's output
should become the next step's input. When the workflow loops over multiple weak questionnaire
answers, add explicit selection/storage steps instead of hiding the branch. Each Input, Process,
and Output preview should support opening a separate inspection window that can be closed without
affecting the workflow page.

The complete workflow has a token budget guard. Preflight estimates the cumulative tokens for one
complete workflow run across all planned LLM calls, then the configured
`token_budget_tolerance_percent` produces `allowed_total_tokens`. Keep
`enforce_token_budget=true` as the default. The estimate should be conservative by design: prefer a
high estimate and a positive surprise after the run over an underestimate that blocks normal work or
risks external-model spend. The workflow must compare final actual usage against preflight and
include per-call token usage in `token_budget.calls`. Use provider-reported token counts when
available and the local rough estimate only as a fallback. Internal Ollama models should show token
counts with zero API cost. External OpenAI estimate data may include USD and EUR metadata using the
configured pricing table and USD-to-EUR rate, but the browser Configuration panel shows USD only.

During development, the Final Result Contract panel includes an Expand button and an "Open Codex
review packet" button. The app cannot directly call the Codex desktop agent; the review button
opens a separate closeable window containing the initial input, run metadata, workflow summaries,
final result, and review rubric so the user can paste it into Codex for an honest quality review.
Do not represent this as an automatic Codex API call unless a real authorized Codex review API is
added later.

The workflow UI must not force-open workflow step tabs while polling. New step tabs should be
created closed, and existing open/closed state should be preserved. The Configuration panel includes
a manual "Estimate cost" button at the bottom. It preflights the current edited input and selected
model without model calls or API keys, and it must show input tokens, output tokens, total tokens,
token cap, and cost. Actual OpenAI runs still require explicit confirmation/API credentials. Running
jobs show ETA based on median duration from previous completed saved runs. If no saved duration
history exists, show that honestly.

Keep the Configuration panel dense enough that all controls and the full run estimate are visible in
the top-left frame without hidden clipping, but do not make the estimate unreadably tiny. The run
estimate must be one readable line with exactly these business items: `Model`, `Expected in/out`
(input and output token estimates), `Hard cap`, and `Total cost` in USD. Do not show mini cards, total-token
clutter, LLM call count, EUR, or a redundant `Status OK` chip in that top-frame estimate.

Quality gates are now a core design requirement, documented in `docs/quality-gates.md`. Runtime
gates are implemented for complete-assessment risk-answer model calls and final-paragraph model
calls. Every gated LLM call has a professional prompt contract, schema/content/evidence validation,
bounded repair retries, and visible failure when output does not satisfy expectations. Never
silently replace failed LLM output with generic fallback text. Step 17 in the 2026-06-15 run
demonstrated the failure mode: the final paragraph model call returned JSON-repair commentary and
the parser hid it behind generic paragraphs. The final report writer now receives a clean validated
fact packet, not raw debug dumps or malformed risk answers. The browser shows failed workflow/job
quality gates in a modal with operator and system-owner remediation guidance.

GraphRAG prompt context must stay clean. A later Q2 / PR.PS-01 failure showed that text retrieval
found the right CIS/SCF anti-malware evidence, but raw graph rows included loose, malformed
relationship labels and distracted the local model. Qdrant/BM25 text evidence is now authoritative;
graph rows must be filtered to retrieved chunk IDs, malformed graph entity names must be dropped,
and prompts must describe graph output as secondary filtered hints.

LLM output must be bounded by purpose. Risk-analysis calls should produce compact JSON with short
labels, limited lists, and no background prose. Final report calls may use prose, but only controlled
2-4 sentence paragraphs with the most important finding first. OpenAI calls use
`max_output_tokens`; Ollama calls use `num_predict`. Token gates protect model-call size/cost,
payload hygiene gates protect step-to-step data cleanliness, and style gates protect usefulness.
