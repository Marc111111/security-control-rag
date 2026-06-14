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

## Current Architecture

The project is organized around five building blocks:

1. Knowledge ingestion pipeline for PDFs, Word files, text, CSV, JSON, YAML, and Excel.
2. Security knowledge schema for chunks, controls, risks, vulnerabilities, tiers, and provenance.
3. Vector database and retrieval layer using local embeddings and ChromaDB.
4. Local LLM orchestration through Ollama and Gemma with source-grounded prompts.
5. Governance layer: this file, architecture docs, tests, GitHub repository, and small PRs.

Details live in `docs/architecture.md` and decisions live in `docs/decisions/`.

## Decisions

- Language: Python 3.11.
- Local generation model: Gemma through Ollama. The current machine currently has `gemma3:4b`.
  This was enough to prove the pipeline but produces average reasoning and inconsistent answer
  shape on security-control recommendation tasks. The next recommended model upgrade for the
  user's RTX 5080 class machine is `qwen3:14b` if it runs comfortably; fallback candidates are
  `gemma3:12b`, `qwen3.5:9b`, or a smaller Llama-family model if VRAM pressure is a problem.
- Embedding model: start with `mxbai-embed-large` through Ollama; fallback option is
  `nomic-embed-text`. Keep `mxbai-embed-large` for now unless retrieval evidence proves it is
  the bottleneck.
- Vector store: ChromaDB persistent local store, with an in-memory store for tests.
- CLI-first delivery. API integration can follow once retrieval quality is proven.
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
security-rag ingest --source data/raw --db storage/chroma
security-rag query --db storage/chroma --criteria "controls for ransomware risk"
security-rag-api
```

## Implementation Notes

- Keep answers grounded in retrieved chunks and include source metadata.
- Retrieval and prompt code must remain testable without a live Ollama server.
- Use `HashEmbeddingClient` and `MemoryVectorStore` for deterministic tests.
- Runtime adapters may depend on optional heavy libraries such as ChromaDB, pypdf, python-docx,
  and openpyxl.
- `POST /api/query` is the preferred app integration point. It accepts natural language in
  `message` and optional structured criteria in `context`.
- The default minimum retrieval score is `0.6` to reduce general-knowledge drift. Tune with
  `SECURE_RAG_MIN_SCORE` only when evidence shows the corpus needs a different threshold.
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
