# Security Control RAG

Local Python RAG system for producing information-security control recommendations from a
private documentation corpus.

The first target stack is:

- Python 3.11
- Ollama with a local Gemma model for generation
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

Large private documents and model blobs are intentionally excluded from Git.

