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
