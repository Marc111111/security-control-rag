# Security Control RAG

Local Python RAG system for producing information-security control recommendations from a
private documentation corpus.

## Advanced GraphRAG Prototype

The repository now includes a second, more professional prototype under `src/app` for
cybersecurity and GRC risk documentation. It is designed for questions like:

```text
A medium company has no anti-malware solution in place. What threats, vulnerabilities, risks and controls should I document?
```

The new pipeline is multi-step rather than single-shot RAG:

- ingest PDFs, DOCX, TXT, Markdown, CSV, JSON, YAML, and Excel through the existing loaders,
- enrich chunks with filename, page/section, document type, framework, control ID, and source path,
- store dense vectors in Qdrant,
- keep an in-process BM25/keyword index for hybrid retrieval,
- extract candidate graph entities and relationships into Neo4j Community,
- decompose risk questions into gap, threat, vulnerability, risk, control, and framework
  sub-questions,
- retrieve from vector search, keyword search, and graph traversal,
- merge/rerank evidence,
- generate a structured risk answer with citations.

Start Qdrant and Neo4j:

```powershell
docker compose up -d
```

Copy configuration:

```powershell
Copy-Item .env.example .env
```

Pull local Ollama models:

```powershell
ollama pull qwen3:14b
ollama pull mxbai-embed-large
```

Ingest documents:

```powershell
python scripts/ingest_graphrag.py standards
```

Start the advanced API:

```powershell
grc-graphrag-api
```

Open the Foundation Summary mock screen:

```text
http://127.0.0.1:8000/mock/foundation
```

Query it:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/query -ContentType application/json -Body '{
  "question": "A medium company has no anti-malware solution in place. What threats, vulnerabilities, risks and controls should I document?",
  "debug": true
}'
```

The response contains:

- `answer.executive_summary`
- `answer.assumptions`
- `answer.threats`
- `answer.vulnerabilities`
- `answer.risks`
- `answer.recommended_controls`
- `answer.risk_control_matrix`
- `answer.missing_information`
- `answer.source_citations`
- `debug.retrieved_chunks` when debug mode is enabled

The API supports Ollama by default and OpenAI through environment variables:

```powershell
$env:GRAPHRAG_LLM_PROVIDER = "openai"
$env:OPENAI_API_KEY = "<your key>"
```

The older `secure_rag` Chroma-based API is still present for compatibility only. New workflow
work should use Qdrant, BM25, and Neo4j through `src/app`.

## Foundation Assessment Summary Prototype

The first SaaS-facing AI workflow is the Foundation Summary Agent. It accepts a canonical vendor
assessment packet from PostgreSQL-shaped data and returns draft report sections ready to store back
as transitory fields or later-approved snapshots.

Endpoint:

```http
POST /api/assessments/foundation-summary
```

Example request:

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
    "questionnaire_results": [
      {
        "question_id": "Q2",
        "question_text": "Do you run anti-malware on endpoints?",
        "control": {
          "framework": "NIST CSF",
          "control_id": "PR.PS-01",
          "title": "Endpoint protection",
          "control_type": "preventative"
        },
        "response": "No",
        "vendor_comment": "No endpoint tool is currently deployed.",
        "analyst_comment": "No anti-malware solution is in place.",
        "compliance": "no",
        "maturity": "basic",
        "evidence": []
      }
    ]
  },
  "debug": true
}
```

Response fields include:

- `draft.management_summary`
- `draft.introduction`
- `draft.objective`
- `draft.key_findings`
- `draft.strengths`
- `draft.weaknesses`
- `draft.risk_exposure`
- `draft.conclusion`
- `draft.missing_information`
- `postgres_payload`

The workflow sanitizes human-generated vendor and analyst comments before placing them in an LLM
prompt. Full compliance becomes strengths; partial and no compliance become weaknesses. The LLM
may draft the business wording, but deterministic code prepares the findings and fallback summary.

For the current complete workflow demo, the screen starts from simulated PostgreSQL-shaped input
but then runs the real workflow stack. The first source shape is replaceable through
`input_source.adapter`, so production PostgreSQL can later supply a different row/view model
without rewriting the chain.

Browser-safe async endpoints:

```http
POST /api/workflows/complete-assessment/preflight
POST /api/workflows/complete-assessment/jobs
GET /api/workflows/complete-assessment/jobs/{job_id}
POST /api/workflows/complete-assessment/jobs/{job_id}/cancel
```

Example request shape:

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
    "max_estimated_input_tokens": 60000,
    "enforce_token_budget": true,
    "token_budget_tolerance_percent": 10
  },
  "top_k": 8,
  "debug": true
}
```

The UI uses preflight, starts a background job, polls job status, and can cancel a local Ollama
run. Cancelling moves a running job through `cancelling` while any in-flight model call unwinds,
then calls `ollama stop <model>` again before the job becomes `cancelled`. Job-scoped Ollama calls
stream responses and use `keep_alive=0s` to avoid holding the generator after a workflow call. The
completed job response includes `steps` for the visual workflow, `cost_estimate`, `final_result`,
and `run_path`. Saved runs can be compared through
`GET /api/workflows/complete-assessment/runs`.
For the complete workflow, token estimates are cumulative for one full run across all planned LLM
calls and are deliberately conservative. Local Ollama runs show token counts with zero API price;
external OpenAI runs include estimated USD and EUR cost from the configured model price table.

For a no-database packet sample, the screen calls:

- `GET /api/mock/foundation-packet`

For a cost check before OpenAI testing, call:

```http
POST /api/assessments/foundation-summary/token-estimate
```

The guarded OpenAI smoke-test endpoint is:

```http
POST /api/assessments/foundation-summary/openai-smoke-test
```

It refuses to run unless `confirm_external_call` is `true` and the estimated input size is below
the request guard. This is intentional: the prototype sends only the compact assessment packet,
not full documents or vector-store contents.

The mock screen also has a model selector for runtime comparison. It can run local Ollama models
such as `qwen3:14b`, estimate OpenAI model costs, and run OpenAI models only when the external-call
checkbox is enabled. Each comparison run returns `model_run.token_estimate` with estimated input
tokens, output tokens, total tokens, and price.

The mock screen can accept an OpenAI API key for a single request. The key is not saved by the
application and is not returned in debug output; use `.env` for a longer-lived local development
configuration.

The mock screen starts clean. After a run, it renders a vertical execution workflow with expandable
steps for the simulated PostgreSQL read, canonical packet, sanitization, finding classification,
token estimate, model call, parsed draft, and PostgreSQL draft-write payload.

## Original Local RAG Prototype

The first target stack is:

- Python 3.11
- Ollama with a local Qwen model for generation
- Ollama embedding model for local embeddings
- ChromaDB for persistent vector search
- pytest for unit and integration tests

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pytest
```

Pull local models:

```powershell
.\scripts\setup_ollama.ps1
```

Index a corpus:

```powershell
security-rag ingest --source data/raw --db storage/chroma
```

Query the corpus:

```powershell
security-rag query --db storage/chroma --criteria "controls for ransomware risk in tier 2"
```

Start the local API and chat UI:

```powershell
$env:SECURE_RAG_DB_PATH = "storage/chroma"
security-rag-api
```

Then open `http://127.0.0.1:8000/`.

Your app can call `POST /api/query` with natural language:

```json
{
  "message": "tier 2 ransomware controls for backup and recovery"
}
```

or natural language plus structured criteria:

```json
{
  "message": "recommend security controls",
  "context": {
    "risk": "ransomware",
    "tier": 2,
    "frameworks": ["ISO27001", "NIST CSF"]
  }
}
```

See `docs/api.md` for the full API contract.

The API defaults to strict corpus-first behavior with a relevance threshold of `0.6`. If retrieval
finds no good source chunks, it returns `insufficient_evidence: true` instead of asking the model
to improvise.

Large private documents and model blobs are intentionally excluded from Git.
