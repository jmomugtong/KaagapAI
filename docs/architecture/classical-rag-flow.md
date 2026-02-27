# Classical RAG Pipeline Flow

## Overview

The Classical RAG pipeline implements an advanced retrieval-augmented generation system with hybrid search, cross-encoder reranking, and comprehensive quality checks.

## Flow Diagram

```mermaid
flowchart TD
    Start([User Query]) --> PII1[PII Redaction<br/>Remove patient identifiers]
    PII1 --> Validate[Input Validation<br/>SQL/XSS prevention]
    Validate --> Cache{Cache Check<br/>Redis 1h TTL}

    Cache -->|Hit| CacheReturn[Return Cached Result<br/>&lt;200ms]
    Cache -->|Miss| Preprocess[Query Preprocessing<br/>Lowercase + medical abbrev expansion]

    Preprocess --> Embed[Generate Query Embedding<br/>sentence-transformers 768-dim<br/>~160ms]

    Embed --> LoadChunks[Load Document Chunks<br/>From PostgreSQL]

    LoadChunks --> BM25[BM25 Keyword Search<br/>Top-10 results]
    LoadChunks --> Vector[pgvector Cosine Search<br/>Top-10 results<br/>CAST to vector type]

    BM25 --> Fusion[Score Fusion<br/>0.4 × BM25 + 0.6 × cosine]
    Vector --> Fusion

    Fusion --> Dedup[Deduplicate by chunk_id<br/>Keep highest scores]

    Dedup --> Rerank[FlashRank Cross-Encoder<br/>Batch reranking &lt;100ms]

    Rerank --> TopK[Select Top-3 Chunks]

    TopK --> Synthesize[LLM Synthesis<br/>Qwen 2.5 1.5B via Ollama<br/>With citations]

    Synthesize --> Confidence[Confidence Scoring<br/>0.0-1.0 scale]

    Confidence --> Threshold{Confidence<br/>&gt;= 0.70?}

    Threshold -->|Yes| Hallucination[Hallucination Detection<br/>Validate citations exist]
    Threshold -->|No| Snippets[Extractive Fallback<br/>Key sentences from docs]

    Hallucination --> PII2[Output PII Redaction<br/>Re-scan response]
    Snippets --> PII2

    PII2 --> CacheWrite[Cache Result<br/>Redis 1h TTL]

    CacheWrite --> Audit[Audit Logging<br/>queries_log table]

    Audit --> Response([Return Response<br/>Answer + citations + chunks])
    CacheReturn --> Response

    style Start fill:#e1f5ff
    style Response fill:#e1f5ff
    style Cache fill:#fff3cd
    style Threshold fill:#fff3cd
    style BM25 fill:#d4edda
    style Vector fill:#d4edda
    style Fusion fill:#d4edda
    style Rerank fill:#cce5ff
    style Synthesize fill:#f8d7da
    style CacheReturn fill:#d1ecf1
```

## Pipeline Stages

### 1. Security & Preprocessing
- **PII Redaction**: Removes patient names, MRNs, DOBs, phone numbers, SSNs, emails
- **Input Validation**: Blocks SQL injection, XSS, malicious patterns
- **Query Preprocessing**: Lowercase normalization, medical abbreviation expansion (MI → myocardial infarction)

### 2. Caching
- **Redis Cache**: 1-hour TTL for query results
- **Cache Key**: SHA256 hash of normalized query text
- **Hit Rate Target**: >70%

### 3. Embedding Generation
- **Model**: sentence-transformers/nomic-embed-text-v1.5
- **Dimensions**: 768
- **Performance**: ~160ms per query embedding (60× faster than Ollama HTTP)
- **Prefix**: `search_query: ` for semantic differentiation

### 4. Hybrid Retrieval
- **BM25 Search**: Keyword-based retrieval with stop word removal
  - Top-10 results
  - Scores normalized to 0.0-1.0
- **Vector Search**: pgvector cosine similarity
  - SQL: `CAST(:query_vector AS vector)` for proper type handling
  - Top-10 results with `WHERE embedding IS NOT NULL`
- **Fusion**: `final_score = 0.4 × bm25_score + 0.6 × cosine_similarity`
- **Deduplication**: By chunk_id, keeping highest-scoring version

### 5. Reranking
- **Model**: FlashRank (ms-marco-TinyBERT-L-2-v2, auto-downloaded)
- **Method**: Cross-encoder batch reranking with sentence-level extraction
- **Performance**: <100ms for batch of 20 chunks
- **Output**: Top-3 re-scored chunks (configurable via `max_results`)

### 6. LLM Synthesis
- **Model**: Qwen 2.5 1.5B via Ollama (configurable via `OLLAMA_MODEL`)
- **Context**: Top-3 chunks + metadata (max 400 chars each, configurable)
- **Output**: Answer with inline citations `[Document Name, Section X]`
- **Timeout**: 120 seconds (configurable via `OLLAMA_TIMEOUT_SECONDS`)
- **Params**: temperature=0.0, max_tokens=200, top_p=0.9

### 7. Quality Assurance
- **Confidence Scoring**: 0.0-1.0 based on retrieval scores and LLM certainty
- **Threshold**: ≥0.70 required for synthesis, else extractive fallback (key sentences from docs)
- **Hallucination Detection**: Token-overlap validation of citations against retrieval set
- **Minimum Chunk Relevance**: Chunks below `MIN_CHUNK_RELEVANCE` (default 0.35) are excluded from LLM context

### 8. Output & Logging
- **Output PII Redaction**: Re-scan synthesized response
- **Result Caching**: Store in Redis for repeat queries
- **Audit Logging**: Log to `queries_log` table with execution metrics

## Performance Metrics

| Metric | Target | Typical |
|--------|--------|---------|
| Query Latency (p95) | <2s | 1.5-1.8s |
| Cached Response | <200ms | 50-150ms |
| Embedding Generation | <500ms | ~160ms |
| Retrieval + Fusion | <500ms | 200-400ms |
| Reranking | <200ms | 50-100ms |
| LLM Synthesis | <3s | 1-2s |

## Enhancements

- **Multi-Query Retrieval**: Generates query reformulations for broader recall (opt-in via `MULTI_QUERY_ENABLED=1`)
- **Entity-Aware Boosting**: Boosts chunks containing medical entities from the query
- **Context Window Expansion**: Fetches adjacent chunks for richer context (opt-in, off by default via `SKIP_CONTEXT_EXPANSION=1`)
- **Extractive Fallback**: Returns key sentences from documents when LLM confidence is below threshold
- **Web Search Fallback**: Searches DuckDuckGo when no local documents match (discounted confidence)
- **Document Name Resolution**: `doc_name_map` resolves `document_id` to actual filenames for citation accuracy
- **Table/Noise Filtering**: GRADE evidence tables and pipe-delimited rows are stripped from chunk text before prompting

## Error Handling

- **Vector Search Failure**: Falls back to BM25-only retrieval
- **Ollama Unavailable**: Returns snippets without synthesis
- **No Results Found**: Web search fallback, then "No relevant results" message
- **Timeout**: 120s timeout on LLM, returns partial results
- **Embedding Failure**: Returns error message with 0% confidence

## Source Annotations

Retrieved chunks carry both retrieval method and document name:
- `source` field: `"bm25"`, `"vector"`, `"hybrid"`, or `"fallback"`
- `document_name` field: Actual filename from `clinical_docs` table (resolved via `doc_name_map`)
