# API Contract

The API gives the existing app a simple way to query the local RAG system without knowing about
Ollama, embeddings, or Chroma internals.

The advanced GraphRAG API lives in `app.main` and can be started with:

```powershell
grc-graphrag-api
```

It uses Qdrant for dense vectors, an in-process BM25 index for keyword retrieval, and Neo4j
Community for graph relationships.

## Advanced GraphRAG Query

`POST /api/query`

```json
{
  "question": "A medium company has no anti-malware solution in place. What threats, vulnerabilities, risks and controls should I document?",
  "context": {
    "company_size": "medium",
    "known_gap": "no anti-malware solution"
  },
  "top_k": 12,
  "debug": true
}
```

Response shape:

```json
{
  "answer": {
    "executive_summary": "",
    "assumptions": [],
    "threats": [],
    "vulnerabilities": [],
    "risks": [],
    "recommended_controls": [],
    "risk_control_matrix": [],
    "missing_information": [],
    "source_citations": [],
    "from_retrieved_evidence": "",
    "general_model_reasoning": ""
  },
  "insufficient_evidence": false,
  "sources": [],
  "debug": {}
}
```

Debug mode includes the generated sub-question plan, retrieved chunks with scores and source
metadata, graph traversal rows, final prompt messages, and the raw model response. The service
logs query debug data under `storage/evaluation`.

## Advanced GraphRAG Ingestion

`POST /api/ingest`

```json
{
  "source": "standards",
  "chunk_size": 1500,
  "overlap": 200,
  "batch_size": 64
}
```

The ingestion path extracts text, chunks it, enriches metadata, stores vectors in Qdrant, indexes
keywords for hybrid retrieval, and extracts candidate graph entities and relationships into Neo4j.

## Advanced GraphRAG Evaluation

`POST /api/retrieve` returns the retrieval plan, chunks, and graph traversal rows without asking
the LLM to generate an answer.

`POST /api/evaluation/feedback` records manual `relevant` / `not relevant` feedback for retrieved
chunks.

## Start the Server

```powershell
$env:SECURE_RAG_DB_PATH = "storage/chroma"
security-rag-api
```

Open the chat UI at `http://127.0.0.1:8000/`.

## Corpus-First Query Flow

Every `/api/query` call follows this order:

1. Convert the natural-language message plus optional JSON criteria into one retrieval query.
2. Embed that query with `mxbai-embed-large` through Ollama.
3. Search the local Chroma vector database.
4. If no source chunks meet the relevance threshold, return `insufficient_evidence: true`
   without asking Gemma to answer.
5. If source chunks are found, send only those excerpts to `gemma3:4b`.
6. Return the generated answer plus source metadata.

The API is therefore strict corpus-first. General model knowledge is not the intended source of
answers. The default relevance threshold is `0.6`; it can be changed with
`SECURE_RAG_MIN_SCORE` if a corpus needs looser or stricter matching.

## Planned Structured Answer Contract

The current API returns a human-readable `answer` string plus source metadata. The next API slice
should also return an enforced structured object so the UI and the user's downstream app can depend
on stable sections instead of parsing free text.

Target shape:

```json
{
  "recommended_controls": [
    {
      "framework": "CIS Controls",
      "control_id": "17.4",
      "title": "Establish and Maintain an Incident Response Process",
      "why_it_applies": "The query describes missing ransomware response preparation.",
      "sources": ["S1"]
    }
  ],
  "related_threats": [],
  "related_vulnerabilities": [],
  "related_risks": [],
  "implementation_notes": [],
  "source_mappings": []
}
```

This structured contract is important because the user expects recommendations to consistently
include controls, threats, vulnerabilities, risks, implementation notes, and source mappings when
the corpus supports them. A stronger local model alone is not enough; the API must enforce the
shape and the UI must render that shape cleanly.

## Natural-Language Query

`POST /api/query`

```json
{
  "message": "tier 2 ransomware controls for backup and recovery",
  "top_k": 8
}
```

Response:

```json
{
  "answer": "Recommended controls...",
  "insufficient_evidence": false,
  "sources": [
    {
      "source_path": "data/raw/security-controls.csv",
      "record_index": 12,
      "chunk_id": "abc123",
      "score": 0.82
    }
  ],
  "raw": {
    "hit_count": 3
  }
}
```

## Structured JSON Criteria

Use `context` when your app already has structured risk data.

```json
{
  "message": "recommend security controls",
  "context": {
    "risk": "ransomware",
    "tier": 2,
    "asset_type": "business critical file server",
    "frameworks": ["ISO27001", "NIST CSF"],
    "vulnerabilities": ["weak backup isolation", "missing restore test"]
  },
  "top_k": 10
}
```

The API folds the JSON into the retrieval query so the vector search and Gemma both see the same
criteria.

## Retrieve Without Generation

`POST /api/retrieve`

Use this endpoint when the app wants to inspect which corpus chunks would be used before asking
Gemma to answer.

```json
{
  "message": "ransomware backup controls",
  "top_k": 5
}
```

## Ingest Documents

`POST /api/ingest`

```json
{
  "source": "data/raw",
  "chunk_size": 1500,
  "overlap": 200
}
```

Response:

```json
{
  "indexed_chunks": 42
}
```

Supported source types currently include text, Markdown, PDF, Word, JSON, YAML, CSV, and Excel.

## Health

`GET /api/health`

Returns the configured vector database path and local model names.
