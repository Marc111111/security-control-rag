# Architecture

## Goal

Create a local RAG system that recommends information-security controls from a private corpus.
The system must prefer source-grounded answers over general model knowledge and must provide
provenance for generated recommendations.

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

Retrieval supports semantic ranking first. Metadata filters and hybrid keyword search are planned
once the first real corpus examples are available.

## Building Block 4: Local LLM Orchestration

The query layer retrieves relevant chunks, builds a grounded prompt, and sends it to Gemma through
Ollama. The prompt instructs the model to:

- derive controls from retrieved sources,
- cite source identifiers,
- mark weak evidence clearly,
- return structured sections suitable for app integration.

When no relevant sources are found, the engine returns an insufficient-evidence response without
asking the LLM to invent an answer.

## Building Block 5: Governance and Delivery

The project uses:

- `AGENT.md` for continuity across agents and context rollovers,
- unit tests for each core module,
- small Git commits and draft PRs,
- ignored local folders for private corpus files, vector stores, and model blobs.

## Data Flow

```text
private corpus -> loaders -> source documents -> chunker -> embeddings -> vector store
                                                                 |
criteria/query -> retriever -------------------------------------+
                                                                 |
retrieved evidence -> grounded prompt -> Ollama/Gemma -> control answer
```

