# 0001 Initial Stack

## Status

Accepted.

## Context

The system must run locally on the user's workstation and use the private security documentation
base as its primary source of truth.

## Decision

Use Python 3.11, Ollama, Gemma, Ollama embeddings, and ChromaDB.

Use a CLI as the first interface because it allows ingestion and retrieval to be verified before
an app/API contract is finalized.

## Consequences

- The project can run without cloud LLM dependencies.
- The local Ollama server becomes an operational dependency.
- Large private data, vector stores, and model files stay outside Git.
- Retrieval quality can be improved independently from the app integration.

