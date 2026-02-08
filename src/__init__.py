"""
MedQuery - Production RAG System for Clinical Documentation

A production-grade Retrieval-Augmented Generation (RAG) system designed to
surface evidence-based clinical answers from medical documentation.

Features:
- Hybrid retrieval (BM25 + vector search)
- LLM-powered reranking and synthesis
- PII redaction and HIPAA compliance
- < 2s query response time
- 95%+ relevance with citation grounding
"""

__version__ = "0.1.0"
__author__ = "MedQuery Team"
