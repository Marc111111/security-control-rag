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

The older `secure_rag` Chroma-based API is still present for compatibility while the GraphRAG
prototype matures.

## Original Local RAG Prototype

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
