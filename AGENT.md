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
- Local generation model: Gemma through Ollama. The current machine already has `gemma3:4b`.
- Embedding model: start with `mxbai-embed-large` through Ollama; fallback option is
  `nomic-embed-text`.
- Vector store: ChromaDB persistent local store, with an in-memory store for tests.
- CLI-first delivery. API integration can follow once retrieval quality is proven.
- Private corpora, generated vector stores, and model blobs are not committed to Git.

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
```

## Implementation Notes

- Keep answers grounded in retrieved chunks and include source metadata.
- Retrieval and prompt code must remain testable without a live Ollama server.
- Use `HashEmbeddingClient` and `MemoryVectorStore` for deterministic tests.
- Runtime adapters may depend on optional heavy libraries such as ChromaDB, pypdf, python-docx,
  and openpyxl.
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
