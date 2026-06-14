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
- Exact recommended controls are post-processed from retrieved evidence text, because local models
  can misread framework version numbers as control IDs.
- Run output is saved under `data/workflow_runs`.
- Request-scoped OpenAI API key is accepted but never persisted or returned.
- Browser runs are async. The UI preflights token/cost estimates before model calls, starts a
  background job, polls status, and can cancel running jobs.
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
- Keep simulated SQL source format isolated behind input adapters.
- Every workflow step shown in the UI must include input, process, and output.
- Large step outputs should be persisted and available for full preview.

## Current Work Status

- Input adapter layer: implemented.
- Workflow-wide model routing: implemented.
- New vertical workflow UI: implemented.
- Async job polling and Cancel Run button: implemented.
- Preflight token/cost estimate before model calls: implemented.
- Run persistence: implemented.
- Unit tests for complete workflow and parser normalization: implemented.
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
- Commit/push/PR update: pending in this goal run.

## Known Risks And Improvements

- The graph extractor is heuristic. It is useful for a prototype but should later be replaced or
  supplemented by a stronger controlled extractor with confidence scoring.
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
- Cost estimates are conservative rough estimates. They are sufficient for comparison, not billing.
