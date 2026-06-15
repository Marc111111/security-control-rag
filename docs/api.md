# API Contract

The API gives the existing app a simple way to query the local RAG system without knowing about
Ollama, embeddings, or Chroma internals.

The advanced GraphRAG API lives in `app.main` and can be started with:

```powershell
grc-graphrag-api
```

It uses Qdrant for dense vectors, an in-process BM25 index for keyword retrieval, and Neo4j
Community for graph relationships.

## Advanced GraphRAG Query

`POST /api/query`

```json
{
  "question": "A medium company has no anti-malware solution in place. What threats, vulnerabilities, risks and controls should I document?",
  "context": {
    "company_size": "medium",
    "known_gap": "no anti-malware solution"
  },
  "top_k": 12,
  "debug": true
}
```

Response shape:

```json
{
  "answer": {
    "executive_summary": "",
    "assumptions": [],
    "threats": [],
    "vulnerabilities": [],
    "risks": [],
    "recommended_controls": [],
    "risk_control_matrix": [],
    "missing_information": [],
    "source_citations": [],
    "from_retrieved_evidence": "",
    "general_model_reasoning": ""
  },
  "insufficient_evidence": false,
  "sources": [],
  "debug": {}
}
```

Debug mode includes the generated sub-question plan, retrieved chunks with scores and source
metadata, graph traversal rows, final prompt messages, and the raw model response. The service
logs query debug data under `storage/evaluation`.

The development UI at `/mock/foundation` also exposes a compact "Ask Standards Corpus" line that
calls this endpoint directly. It is intentionally separate from the full questionnaire assessment
workflow, so it can be used to test how the configured GraphRAG service model answers an ad hoc
standards question without running the simulated PostgreSQL-to-report chain.

## Local OpenAI Key Cache

`GET /api/local/openai-key`, `POST /api/local/openai-key`, and `DELETE /api/local/openai-key`
support the development UI's local key cache. The key is stored on the local machine under
`storage/local/openai_api_key.txt`, which is ignored by Git. A cached key only pre-fills the UI
field; external model calls must still be explicitly enabled per run.

## Advanced GraphRAG Ingestion

`POST /api/ingest`

```json
{
  "source": "standards",
  "chunk_size": 1500,
  "overlap": 200,
  "batch_size": 64
}
```

The ingestion path extracts text, chunks it, enriches metadata, stores vectors in Qdrant, indexes
keywords for hybrid retrieval, and extracts candidate graph entities and relationships into Neo4j.

## Advanced GraphRAG Evaluation

`POST /api/retrieve` returns the retrieval plan, chunks, and graph traversal rows without asking
the LLM to generate an answer.

`POST /api/evaluation/feedback` records manual `relevant` / `not relevant` feedback for retrieved
chunks.

## Foundation Assessment Summary

`POST /api/assessments/foundation-summary`

This endpoint supports the first SaaS prototype. It accepts one canonical vendor assessment packet
containing vendor profile data, tier information, questionnaire results, linked controls, vendor
comments, analyst comments, and evidence descriptions. It returns structured draft report fields
that can be inserted into PostgreSQL as transitory draft values.

```json
{
  "packet": {
    "assessment_id": "A-100",
    "vendor": {
      "vendor_id": "V-1",
      "name": "Acme SaaS",
      "vendor_type": "SaaS provider",
      "business_relationship": "customer data processing"
    },
    "tier": {
      "level": 2,
      "definition": "Important vendor with access to sensitive business data.",
      "attributes": []
    },
    "questionnaire_results": []
  },
  "debug": false
}
```

Response:

```json
{
  "assessment_id": "A-100",
  "vendor_id": "V-1",
  "draft": {
    "management_summary": "",
    "introduction": "",
    "objective": "",
    "key_findings": [],
    "strengths": [],
    "weaknesses": [],
    "risk_exposure": "",
    "conclusion": "",
    "missing_information": [],
    "source_question_ids": [],
    "from_assessment_data": "",
    "general_model_reasoning": ""
  },
  "findings": {
    "strengths": [],
    "weaknesses": [],
    "unknowns": []
  },
  "postgres_payload": {
    "assessment_id": "A-100",
    "vendor_id": "V-1",
    "draft_sections": {},
    "snapshot_ready": false,
    "source_question_ids": []
  }
}
```

Human-generated comments are sanitized before prompt construction. Full compliance is classified
as strength; partial and no compliance are classified as weakness. The endpoint does not create
immutable snapshots itself; that remains an application persistence decision after analyst review.

## Complete Assessment GraphRAG Workflow

The browser UI uses the async job API so long local Ollama calls do not freeze the page.

Primary async endpoints:

- `POST /api/workflows/complete-assessment/preflight`
- `POST /api/workflows/complete-assessment/jobs`
- `GET /api/workflows/complete-assessment/jobs/{job_id}`
- `POST /api/workflows/complete-assessment/jobs/{job_id}/cancel`

The synchronous endpoint `POST /api/workflows/complete-assessment/run` remains available for
debugging and compatibility, but the UI should not use it for local model runs.

The workflow accepts a replaceable source input adapter, normalizes the source data, runs the real
GraphRAG workflow, and returns persisted workflow steps plus a deterministic PostgreSQL-ready
result contract.

Request:

```json
{
  "input_source": {
    "adapter": "foundation_packet_v1",
    "payload": {
      "assessment_id": "A-100",
      "vendor": {},
      "tier": {},
      "questionnaire_results": []
    }
  },
  "model": {
    "provider": "ollama",
    "model": "qwen3:14b",
    "confirm_external_call": false,
    "estimated_output_tokens": 1200,
    "max_estimated_input_tokens": 60000,
    "enforce_token_budget": true,
    "token_budget_tolerance_percent": 10
  },
  "top_k": 8,
  "debug": true
}
```

Supported `input_source.adapter` values:

- `foundation_packet_v1`: payload is already the normalized assessment packet.
- `simulated_postgres_v1`: payload can be a simulated PostgreSQL row, `{ "row": ... }`,
  `{ "rows": [ ... ] }`, or `{ "packet": ... }`.

The source adapter is intentionally the only layer that knows the SQL/result shape. A production
PostgreSQL adapter can be added later without changing GraphRAG retrieval, LLM prompting, result
assembly, or the UI renderer.

Supported model selections:

- Local: `provider=ollama`, `model=qwen3:14b` or `gemma3:4b`.
- External: `provider=openai` with a GPT-4.1/GPT-5 text model from
  `POST /api/models/available`.

The model selector endpoint always returns curated fallbacks and, when an OpenAI key is available,
also merges in currently listed OpenAI GPT text models:

```http
POST /api/models/available
```

```json
{
  "openai_api_key": "optional request-scoped key"
}
```

OpenAI discovery is filtered to general text models used by this RAG workflow. It excludes Codex,
pro, realtime, audio, image, search, TTS, and transcription variants.

OpenAI cost/token estimates do not require `confirm_external_call` or an API key because preflight
does not call the external model. Actual OpenAI workflow runs require
`confirm_external_call=true` and either a request-scoped `openai_api_key` or `OPENAI_API_KEY` in
the environment.

Preflight response:

```json
{
  "adapter": "foundation_packet_v1",
  "assessment_id": "A-100",
  "vendor_id": "V-1",
  "provider": "ollama",
  "model": "qwen3:14b",
  "weakness_count": 2,
  "retrieval_query_count": 2,
  "top_k": 8,
  "llm_call_count": 3,
  "estimated_input_tokens": 38907,
  "estimated_output_tokens": 3600,
  "estimated_total_tokens": 42507,
  "estimate_policy": "conservative_workflow_reserve",
  "enforce_token_budget": true,
  "token_budget_tolerance_percent": 10,
  "allowed_total_tokens": 46758,
  "estimated_cost_usd": 0.0,
  "estimated_cost_eur": 0.0,
  "usd_to_eur_rate": 0.92,
  "price_per_million_tokens": null,
  "pricing_note": "Internal/local model run: token use is tracked, but API price is $0.",
  "will_exceed_guard": false
}
```

`estimated_total_tokens` is cumulative for one complete workflow run across all planned LLM calls.
The estimate is intentionally conservative: it reserves tokens for the assessment packet, findings,
each risk-answer prompt, top-k retrieved chunks, graph context, final report drafting, and a safety
margin. `allowed_total_tokens` is the workflow estimate plus the configured tolerance. When
`enforce_token_budget=true`, the workflow checks each model call before it is sent and again after
the response is received. Provider-reported token counts are used where available; otherwise the
same conservative estimate is used for actual/estimated comparison.

For local Ollama models, API cost fields remain `0`; token counts are still shown. For external
OpenAI models, `estimated_cost_usd` and `estimated_cost_eur` are calculated from the configured
per-million-token price table. EUR uses the configured `usd_to_eur_rate`. If OpenAI discovers a
new compatible model before exact pricing is configured, the estimate uses a conservative
placeholder of `$10/M input` and `$60/M output` tokens rather than showing `$0`.

Start-job response:

```json
{
  "job_id": "job-...",
  "status": "queued",
  "provider": "ollama",
  "model": "qwen3:14b",
  "preflight": {}
}
```

Status values are `queued`, `running`, `cancelling`, `completed`, `failed`, and `cancelled`.
A completed job returns `result` with the same shape as the synchronous workflow response.
Cancelling an Ollama job sets the cancellation flag and calls `ollama stop <model>` immediately.
If a worker is already inside a model call, the job remains `cancelling` until that call unwinds;
job-scoped Ollama calls stream responses so the worker can observe cancellation between chunks.
The worker then calls `ollama stop <model>` again before marking the job `cancelled`. Workflow
Ollama calls use `keep_alive=0s` so a completed local run does not keep the generator resident
for the default Ollama keepalive window.

While a job is `running` or `cancelling`, the status response also includes:

- `current_step`: the last completed step or the current high-level status.
- `partial_steps`: completed workflow steps so far. The UI renders these while the job is still
  running so an analyst can see whether the chain is progressing properly.

The complete-assessment workflow is intentionally transparent. Search planning, standards
retrieval, graph lookup, model risk-answer calls, report paragraph drafting, and final contract
assembly are separate visible steps with their own input, plain-language process explanation, and
output.

Response:

```json
{
  "run_id": "run-...",
  "created_at": "2026-06-14T...",
  "completed_at": "2026-06-14T...",
  "duration_seconds": 42.3,
  "assessment_id": "A-100",
  "vendor_id": "V-1",
  "provider": "ollama",
  "model": "qwen3:14b",
  "cost_estimate": {
    "llm_call_count": 5,
    "estimated_input_tokens": 44577,
    "estimated_output_tokens": 6000,
    "estimated_total_tokens": 50577,
    "estimated_cost_usd": 0.0,
    "estimated_cost_eur": 0.0
  },
  "preflight": {},
  "token_budget": {
    "preflight_estimated_total_tokens": 50577,
    "tolerance_percent": 10,
    "allowed_total_tokens": 55635,
    "actual_total_tokens": 12000,
    "difference_percent": -71.77,
    "within_budget": true,
    "actual_cost_estimate": {},
    "calls": []
  },
  "steps": [
    {
      "name": "Input source adapter",
      "explanation": "",
      "tool": "",
      "input": {},
      "process": "",
      "output": {}
    }
  ],
  "final_result": {
    "assessment_id": "A-100",
    "vendor_id": "V-1",
    "draft_sections": {
      "management_summary": "",
      "risk_exposure": "",
      "conclusion": "",
      "storyline_report": {}
    },
    "risk_evaluations": [],
    "risk_assessment_chains": [],
    "business_storylines": [],
    "storyline_report": {
      "title": "Gap-to-risk storyline for Acme SaaS",
      "purpose": "Human-readable view of the same validated JSON result.",
      "per_gap": [
        {
          "question_id": "Q2",
          "linked_control": {
            "framework": "NIST CSF",
            "control_id": "PR.PS-01",
            "title": "Endpoint protection"
          },
          "gap_to_risk_story": {
            "question_id": "Q2",
            "gap_story": "",
            "business_meaning": "",
            "risk_logic": "",
            "control_logic": "",
            "resilience_logic": "",
            "residual_conclusion": ""
          }
        }
      ],
      "overall_conclusion": ""
    },
    "snapshot_ready": false,
    "source_question_ids": []
  },
  "run_path": "data/workflow_runs/run-....json"
}
```

The UI renders `steps` vertically. Each step includes input, process, and output so analysts can
inspect what was called, what was sent to the selected model, and what came back. The intended
display contract is that a step output becomes the next step input. Where the workflow branches
over multiple weak questionnaire answers, explicit selection/storage handoff steps keep that chain
readable. `storyline_report` is the human-readable report view and is assembled from the same
accepted `business_storylines` stored in the JSON contract, so browser text and API data stay
aligned.
visible. Every preview block can be expanded into a separate inspection window without changing the
running workflow.

### Mock UI and Mock Endpoints

Open the local mockup at:

```text
http://127.0.0.1:8000/mock/foundation
```

It simulates PostgreSQL input and output without requiring PostgreSQL, Qdrant, Neo4j, Ollama, or
OpenAI.

The mock screen also includes an optional simulated database input form. It edits the same packet
JSON sent to the workflow, can save one browser-local scenario, and can reset to the original sample.
The business meaning of those fields is documented at:

```text
http://127.0.0.1:8000/mock/foundation/business-context
```

Mock endpoints:

- `GET /mock/foundation/business-context`
- `GET /api/mock/foundation-packet`
- `POST /api/mock/foundation-summary`

### Token Estimate and OpenAI Smoke Test

`POST /api/assessments/foundation-summary/token-estimate` estimates prompt size and cost before
any external API call.

`POST /api/assessments/foundation-summary/openai-smoke-test` makes a real OpenAI call only when:

- `confirm_external_call` is `true`,
- `OPENAI_API_KEY` is configured,
- the estimated input tokens are below `max_estimated_input_tokens`.

The smoke test sends only the compact foundation assessment packet and generated prompt. It does
not send full documents, vector stores, evidence files, or the GraphRAG corpus.

For runtime comparison from the mock UI, use:

```http
POST /api/assessments/foundation-summary/model-run
```

Request fields:

- `provider`: `mock`, `ollama`, or `openai`
- `model`: for example `qwen3:14b`, `gpt-5.4-mini`, or `gpt-4.1-mini`
- `confirm_external_call`: required for OpenAI calls
- `openai_api_key`: optional request-scoped key for one OpenAI call

The response includes `model_run.token_estimate` so each request can display estimated price next
to the generated result.

The mock UI renders the response as an execution workflow. Each step has expandable input and
output previews so the analyst can inspect what was sent to the model and what came back.

## Start the Server

```powershell
$env:SECURE_RAG_DB_PATH = "storage/chroma"
security-rag-api
```

Open the chat UI at `http://127.0.0.1:8000/`.

## Corpus-First Query Flow

Every `/api/query` call follows this order:

1. Convert the natural-language message plus optional JSON criteria into one retrieval query.
2. Embed that query with `mxbai-embed-large` through Ollama.
3. Search the local Chroma vector database.
4. If no source chunks meet the relevance threshold, return `insufficient_evidence: true`
   without asking Gemma to answer.
5. If source chunks are found, send only those excerpts to `qwen3:14b`.
6. Return the generated answer plus source metadata.

The API is therefore strict corpus-first. General model knowledge is not the intended source of
answers. The default relevance threshold is `0.6`; it can be changed with
`SECURE_RAG_MIN_SCORE` if a corpus needs looser or stricter matching.

## Planned Structured Answer Contract

The current API returns a human-readable `answer` string plus source metadata. The next API slice
should also return an enforced structured object so the UI and the user's downstream app can depend
on stable sections instead of parsing free text.

Target shape:

```json
{
  "recommended_controls": [
    {
      "framework": "CIS Controls",
      "control_id": "17.4",
      "title": "Establish and Maintain an Incident Response Process",
      "why_it_applies": "The query describes missing ransomware response preparation.",
      "sources": ["S1"]
    }
  ],
  "related_threats": [],
  "related_vulnerabilities": [],
  "related_risks": [],
  "implementation_notes": [],
  "source_mappings": []
}
```

This structured contract is important because the user expects recommendations to consistently
include controls, threats, vulnerabilities, risks, implementation notes, and source mappings when
the corpus supports them. A stronger local model alone is not enough; the API must enforce the
shape and the UI must render that shape cleanly.

## Natural-Language Query

`POST /api/query`

```json
{
  "message": "tier 2 ransomware controls for backup and recovery",
  "top_k": 8
}
```

Response:

```json
{
  "answer": "Recommended controls...",
  "insufficient_evidence": false,
  "sources": [
    {
      "source_path": "data/raw/security-controls.csv",
      "record_index": 12,
      "chunk_id": "abc123",
      "score": 0.82
    }
  ],
  "raw": {
    "hit_count": 3
  }
}
```

## Structured JSON Criteria

Use `context` when your app already has structured risk data.

```json
{
  "message": "recommend security controls",
  "context": {
    "risk": "ransomware",
    "tier": 2,
    "asset_type": "business critical file server",
    "frameworks": ["ISO27001", "NIST CSF"],
    "vulnerabilities": ["weak backup isolation", "missing restore test"]
  },
  "top_k": 10
}
```

The API folds the JSON into the retrieval query so the vector search and Gemma both see the same
criteria.

## Retrieve Without Generation

`POST /api/retrieve`

Use this endpoint when the app wants to inspect which corpus chunks would be used before asking
Gemma to answer.

```json
{
  "message": "ransomware backup controls",
  "top_k": 5
}
```

## Ingest Documents

`POST /api/ingest`

```json
{
  "source": "data/raw",
  "chunk_size": 1500,
  "overlap": 200
}
```

Response:

```json
{
  "indexed_chunks": 42
}
```

Supported source types currently include text, Markdown, PDF, Word, JSON, YAML, CSV, and Excel.

## Health

`GET /api/health`

Returns the configured vector database path and local model names.
