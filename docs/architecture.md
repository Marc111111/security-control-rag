# Architecture

## Goal

Create a local RAG system that recommends information-security controls from a private corpus.
The system must prefer source-grounded answers over general model knowledge and must provide
provenance for generated recommendations.

The second implementation slice adds an advanced cybersecurity/GRC GraphRAG prototype under
`src/app`. It keeps the original local-first requirement but replaces single-shot retrieval with
question planning, hybrid retrieval, graph extraction, and structured risk answers.

## Building Block 1: Knowledge Ingestion

The ingestion pipeline loads documents from the private corpus, extracts text and metadata, and
normalizes them into source documents. Supported initial formats are text, Markdown, PDF, Word,
JSON, YAML, CSV, and Excel.

The output of this layer is a sequence of source documents with metadata such as source path,
file type, page, worksheet, record index, and any structured fields discovered during parsing.

## Building Block 2: Security Knowledge Schema

The schema layer defines stable internal records:

- `SourceDocument`: extracted text plus source metadata.
- `Chunk`: retrieval-sized text segment with provenance metadata.
- `RetrievalHit`: scored search result used for prompt construction.
- `ControlAnswer`: structured answer returned to the application.

The schema is intentionally generic in the first slice. Domain-specific fields such as framework,
control ID, tier, vulnerability, asset class, and risk type live in metadata so ingestion can
evolve without rewriting the entire retrieval layer.

## Building Block 3: Vector Database and Retrieval

Chunks are embedded with a local Ollama embedding model and stored in a vector database. ChromaDB
is the default persistent store. Tests use an in-memory vector store and deterministic hash
embeddings.

Retrieval started with semantic ranking and now also expands common security terms and reranks by
keyword overlap plus standards-source priority. The next improvement is to build a control
recommendation packet before generation: grouped candidate controls, related threats, related
vulnerabilities, related risks, implementation notes, and source mappings. This is needed because
the user expects answers that preserve those sections consistently, not just plausible prose.

## Building Block 4: Local LLM Orchestration

The query layer retrieves relevant chunks, builds a grounded prompt, and sends it to Gemma through
Ollama. The prompt instructs the model to:

- derive controls from retrieved sources,
- cite source identifiers,
- mark weak evidence clearly,
- return structured sections suitable for app integration,
- avoid controls or implementation details that are not present in retrieved source excerpts.

When no relevant sources are found, the engine returns an insufficient-evidence response without
asking the LLM to invent an answer.

The local generation model is `qwen3:14b`, selected as the strongest practical first target for
the user's RTX 5080 class machine. The original `gemma3:4b` proved the pipeline but showed weak
consistency for security-control reasoning. Fallback candidates are `gemma3:12b`, `qwen3.5:9b`,
or a smaller Llama-family model if VRAM pressure is a problem. The embedding model should remain
`mxbai-embed-large` until evidence shows retrieval quality is the bottleneck.

The answer should move from free-text prompting to an enforced schema:

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

The UI can then render clean sections without depending on the model to format paragraphs and
bullets perfectly.

## Foundation Assessment Flow

The first SaaS-facing AI use case is a Foundation Summary Agent. It is deliberately narrower than
the full risk evaluation workflow.

```text
PostgreSQL assessment rows
  -> canonical assessment packet
  -> sanitize human-generated comments
  -> classify questionnaire results
  -> full compliance -> strengths
  -> partial/no compliance -> weaknesses
  -> LLM drafts business-friendly sections
  -> deterministic fallback if model output is invalid
  -> PostgreSQL-ready draft payload
```

The endpoint is `POST /api/assessments/foundation-summary`. It returns Management Summary,
Introduction, Objective, Key Findings, Strengths, Weaknesses, Risk Exposure, Conclusion, Missing
Information, source question IDs, and a `postgres_payload` object. Immutable snapshots should be
created by the application only after analyst approval.

The mock screen at `/mock/foundation` demonstrates this flow without real PostgreSQL. It shows:

- simulated source assessment JSON,
- draft report sections,
- classified findings,
- PostgreSQL-ready draft payload,
- token estimate/debug output.

OpenAI testing is guarded by token estimation and an explicit `confirm_external_call` flag.

## Building Block 5: Governance and Delivery

The project uses:

- `AGENT.md` for continuity across agents and context rollovers,
- unit tests for each core module,
- small Git commits and draft PRs,
- ignored local folders for private corpus files, vector stores, and model blobs.

The first user-facing surface is a local web UI at `/`, backed by `POST /api/query`. The same API
also exposes `/api/ingest`, `/api/retrieve`, and `/api/health` for app integration.

## Data Flow

```text
private corpus -> loaders -> source documents -> chunker -> embeddings -> vector store
                                                                 |
criteria/query -> retriever -------------------------------------+
                                                                 |
retrieved evidence -> grounded prompt -> Ollama/Gemma -> control answer
```

## Advanced GraphRAG Data Flow

```text
documents
  -> loaders and enriched chunks
  -> dense embeddings -> Qdrant
  -> BM25 keyword index
  -> heuristic graph extraction -> Neo4j/Memgraph-compatible graph

question
  -> risk planner
  -> sub-questions for gap, threats, vulnerabilities, risks, controls, frameworks
  -> dense retrieval + keyword retrieval + graph traversal
  -> merge, deduplicate, rerank
  -> structured answer prompt
  -> Ollama or OpenAI model
  -> JSON risk answer with citations and debug evidence
```

Graph entity types:

- Asset
- Gap
- Threat
- Vulnerability
- Risk
- Control
- Compliance requirement
- Evidence source

Core relationship types:

- `THREAT_EXPLOITS_VULNERABILITY`
- `VULNERABILITY_CREATES_RISK`
- `CONTROL_MITIGATES_RISK`
- `CONTROL_ADDRESSES_VULNERABILITY`
- `GAP_INCREASES_LIKELIHOOD_OF_THREAT`
- `REQUIREMENT_REQUIRES_CONTROL`
