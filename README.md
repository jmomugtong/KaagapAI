# MedQuery

<div align="center">

![MedQuery Logo](docs/assets/logo.png)

**Production RAG System for Clinical Documentation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

</div>

---

## Overview

MedQuery is a production-grade Retrieval-Augmented Generation (RAG) system designed to surface evidence-based clinical answers from medical documentation. Built 100% with open-source models, achieving sub-2 second response times and 95%+ relevance while maintaining zero external API costs.

### Key Features

- **Fast Responses**: Query latency p95 < 2 seconds, cached queries < 200ms
- **High Accuracy**: ROUGE-L score >= 0.60, hallucination rate < 5%
- **HIPAA Compliant**: PII redaction, audit logging, row-level security
- **Zero API Cost**: 100% open-source stack (Ollama, FlashRank, pgvector)
- **Medical-Domain LLM**: MedGemma 4B fine-tuned on clinical QA and FHIR EHR data
- **Full Observability**: Prometheus metrics, Grafana dashboards, OpenTelemetry tracing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                                │
│  Next.js Frontend: Query Interface + Results Display           │
└────────────────────┬────────────────────────────────────────────┘
                     │ REST API + JWT Authentication
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     API GATEWAY LAYER                           │
│  FastAPI Application Server                                    │
│  - POST /api/v1/query (rate limited: 10/min per user)          │
│  - POST /api/v1/upload (async document ingestion)              │
│  - GET /api/v1/evals (evaluation suite execution)              │
│  - GET /metrics (Prometheus endpoint)                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌───────────────┐       ┌──────────────────┐
│  Celery Queue │       │  RAG Pipeline    │
│  - Embedding  │       │  - Chunking      │
│  - Indexing   │       │  - Retrieval     │
│  - Evaluation │       │  - Reranking     │
└───────┬───────┘       └────────┬─────────┘
        │                        │
        └──────────┬─────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                   DATA LAYER                                    │
│  PostgreSQL + pgvector  │  Redis Cache  │  Ollama LLM          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- **Docker** & **Docker Compose** v2.0+
- **Git**
- 8GB+ RAM recommended
- (Optional) NVIDIA GPU for faster LLM inference

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/medquery.git
cd medquery
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Generate a secure secret key
python -c "import secrets; print(secrets.token_hex(32))"

# Update .env with your values (especially SECRET_KEY, DB_PASSWORD)
```

### 3. Start All Services

```bash
# Start all containers
docker-compose up -d

# Wait for services to initialize (about 30 seconds)
sleep 30

# Check service status
docker-compose ps
```

### 4. Download AI Models

```bash
# Download MedGemma model for Ollama (one-time, ~2.5GB)
docker-compose exec ollama ollama pull alibayram/medgemma

# Download nomic-embed-text embedding model (one-time, ~274MB)
docker-compose exec ollama ollama pull nomic-embed-text

# Or pre-download all models for offline deployment:
python scripts/setup_offline.py
```

### 5. Initialize Database

```bash
# Run database migrations
docker-compose exec api alembic upgrade head

# (Optional) Seed sample clinical documents
docker-compose exec api python scripts/seed_clinical_docs.py
```

### 6. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "version": "0.1.0"}
```

### 7. Access Services

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | FastAPI application |
| API Docs | http://localhost:8000/docs | Swagger UI |
| Grafana | http://localhost:3000 | Monitoring dashboards |
| Prometheus | http://localhost:9090 | Metrics |
| Ollama | http://localhost:11434 | LLM API |

---

## Usage

### Submit a Query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the post-operative pain protocol for knee replacement?",
    "max_results": 5,
    "confidence_threshold": 0.70
  }'
```

### Upload a Document

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@clinical_protocol.pdf" \
  -F "document_type=protocol" \
  -F 'metadata={"department": "orthopedics", "version": "3.2"}'
```

### Run Evaluation Suite

```bash
# Run full evaluation
docker-compose exec api python scripts/run_evals.py

# Check if thresholds are met
docker-compose exec api python scripts/check_thresholds.py
```

---

## Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e ".[dev,test]"

# Install pre-commit hooks
pre-commit install

# Run development server
make dev
```

### Running Tests

```bash
# All tests
make test

# With coverage report
make test-cov

# Unit tests only
make test-unit

# Integration tests
make test-int
```

### Code Quality

```bash
# Run linters
make lint

# Format code
make format

# Type checking
make type-check

# All checks
make check
```

---

## Project Structure

```
medquery/
├── src/
│   ├── api/              # FastAPI routes and middleware
│   │   ├── routes.py     # API endpoints
│   │   ├── auth.py       # Authentication
│   │   └── middleware.py # Request/response middleware
│   ├── rag/              # RAG pipeline components
│   │   ├── chunker.py    # Document chunking
│   │   ├── embedding.py  # Embedding generation
│   │   ├── retriever.py  # Hybrid retrieval (BM25 + vector)
│   │   ├── reranker.py   # FlashRank cross-encoder reranking
│   │   └── cache.py      # Caching logic
│   ├── llm/              # LLM integration
│   │   ├── ollama_client.py    # Ollama API client
│   │   └── prompt_templates.py # Prompt engineering
│   ├── security/         # Security modules
│   │   ├── pii_redaction.py   # PII detection & redaction
│   │   ├── input_validation.py # Input sanitization
│   │   └── rate_limiter.py    # Rate limiting
│   ├── db/               # Database layer
│   │   ├── postgres.py   # PostgreSQL + pgvector
│   │   └── models.py     # SQLAlchemy models
│   ├── observability/    # Monitoring & logging
│   │   ├── telemetry.py  # OpenTelemetry setup
│   │   └── metrics.py    # Prometheus metrics
│   └── main.py           # Application entry point
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── datasets/             # Evaluation datasets
├── k6/                   # Load test scripts
├── docker-compose.yml    # Docker services
├── Dockerfile            # Container build
├── Makefile              # Development commands
├── pyproject.toml        # Project configuration
├── requirements.txt      # Pinned dependencies
└── README.md             # This file
```

---

## Metrics & Monitoring

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Query Latency (p95) | < 2s | 95th percentile response time |
| Cache Hit Rate | > 70% | Embedding cache efficiency |
| Hallucination Rate | < 5% | False citation detection |
| ROUGE-L Score | ≥ 0.60 | Answer relevance |

### Grafana Dashboards

Access Grafana at http://localhost:3000 (default: admin/admin)

- **Query Performance**: Latency percentiles, throughput
- **Cache Analytics**: Hit rates, memory usage
- **Quality Metrics**: Hallucination rate, confidence scores
- **System Health**: Error rates, resource utilization

---

## Configuration

### Environment Variables

See [.env.example](.env.example) for all available configuration options.

Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | JWT signing key | (generate) |
| `DATABASE_URL` | PostgreSQL connection | postgresql+asyncpg://... |
| `REDIS_URL` | Redis connection | redis://redis:6379/0 |
| `OLLAMA_MODEL` | LLM model name | alibayram/medgemma |
| `RATE_LIMIT_PER_MINUTE` | API rate limit | 10 |

---

## Evaluation

The evaluation framework validates system quality:

```bash
# Run evaluation suite
python scripts/run_evals.py

# Output:
# [PASS] ROUGE-L: 0.68 (threshold: 0.60)
# [PASS] Hallucination Rate: 2% (threshold: 5%)
# [PASS] Retrieval Recall: 94% (threshold: 90%)
# 
# PASSED: All metrics within thresholds
```

### Evaluation Dataset

Located at `datasets/clinical_qa_50.json`:
- 50 clinical Q&A pairs
- Ground truth answers from clinical experts
- Multi-hop and single-document queries

---

## Security

### HIPAA Compliance

- **PII Redaction**: Automatic detection of patient names, MRNs, DOBs
- **Audit Logging**: All queries logged with user context
- **Row-Level Security**: Hospital data isolation in PostgreSQL
- **Encryption**: TLS 1.3 in transit, AES-256 at rest

### Authentication

- OAuth2 + JWT tokens
- Rate limiting per user (10 req/min)
- Input validation and sanitization

---

## API Documentation

Full API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Submit clinical query |
| POST | `/api/v1/upload` | Upload document |
| GET | `/api/v1/jobs/{id}` | Check job status |
| GET | `/api/v1/evals` | Run evaluation |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Technology Stack

The open-source model choices were informed by research from 10+ AI practitioners and engineers. Each component was selected for optimal clinical RAG performance on CPU-only, 16 GB RAM deployments with zero API cost.

| Component | Technology | Why |
|-----------|-----------|-----|
| **LLM** | [MedGemma 4B](https://ollama.com/alibayram/medgemma) | Google's medical-domain model (clinical QA + FHIR EHR data), 40% less RAM than 7B |
| **Embedding** | [nomic-embed-text](https://ollama.com/library/nomic-embed-text) | 768-dim, 8192 token context, Ollama-native, superior MTEB retrieval |
| **Reranker** | [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) | <100ms batch reranking on CPU, 4MB model, no torch dependency |
| **Chunking** | [LangChain SemanticChunker](https://python.langchain.com/) + medical separators | Section-aware boundaries (1500 chars, 200 overlap) |
| **Retrieval** | BM25 + pgvector hybrid | 0.4 BM25 + 0.6 cosine similarity fusion |
| **Vector DB** | PostgreSQL + pgvector | 768-dim vectors with IVFFlat indexing |
| **LLM Runtime** | [Ollama](https://ollama.ai/) | Local inference for LLM + embeddings, zero API cost |
| **Cache** | Redis | Two-tier: embedding (7d TTL) + query (1h TTL) |
| **Framework** | [FastAPI](https://fastapi.tiangolo.com/) | Async Python web framework |

### Research References

Model and architecture decisions were guided by recommendations from these AI practitioners:

**Thought Leaders:**
- **Sebastian Raschka** -- [State of LLMs 2025](https://sebastianraschka.com/blog/2025/state-of-llms-2025.html): "Qwen overtook Llama. Smaller well-trained models match larger predecessors."
- **Andrej Karpathy** -- [2025 LLM Year in Review](https://karpathy.bearblog.dev/year-in-review-2025/): Context engineering -- precision over volume in RAG context windows
- **Andrew Ng** -- [RAG Course (Coursera)](https://www.coursera.org/learn/retrieval-augmented-generation-rag), [Advanced Retrieval with Chroma](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/): Cross-encoder reranking + larger chunk context + evaluate retrieval independently
- **Chorouk Malmoum** -- SLMs for Agentic AI (citing NVIDIA research): SLMs outperform LLMs for specialized tasks

**RAG Practitioners:**
- **Paul Iusztin** -- [Build RAG Pipelines That Actually Work](https://decodingml.substack.com/p/build-rag-pipelines-that-actually): "RAG projects fail because no one measures retrieval quality"
- **Shantanu Ladhwe** -- [Production RAG with RAGAS Evaluation](https://jamwithai.substack.com/): Ollama + section-aware chunking, Langfuse observability
- **Paolo Perrone** -- [The Open-Source Stack for AI Agents](https://www.decodingai.com/p/the-open-source-stack-for-ai-agents): NVIDIA Nemotron, Docling for parsing
- **Alex Razvant** -- [A Practical Roadmap on LLM Systems](https://multimodalai.substack.com/p/a-practical-roadmap-on-llm-systems)
- **Daniel Lee** -- RRA Pattern (Retrieval, Rerank, Augment) for enterprise-grade LLMs
- **Jeremy Park** -- Production RAG architecture patterns

---

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Ollama](https://ollama.ai/) - Local LLM and embedding inference
- [MedGemma](https://ollama.com/alibayram/medgemma) - Medical-domain LLM
- [nomic-embed-text](https://ollama.com/library/nomic-embed-text) - Embedding model
- [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) - Lightweight reranker
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search
- [LangChain](https://python.langchain.com/) - Document processing & semantic chunking

---

<div align="center">

**Built for healthcare professionals**

</div>
