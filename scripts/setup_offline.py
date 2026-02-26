#!/usr/bin/env python3
"""
One-time setup script for offline hospital deployment.

Run this ONCE while internet is available to download all models
and verify the system can operate fully offline afterwards.

Usage:
    python scripts/setup_offline.py

What it does:
    1. Pulls the LLM model (alibayram/medgemma) into Ollama
    2. Pulls the embedding model (nomic-embed-text) into Ollama
    3. Pre-downloads the FlashRank reranker model
    4. Verifies all models are available
    5. Prints confirmation that the system can operate offline
"""

import os
import sys


def pull_ollama_model() -> bool:
    """Pull the configured LLM model into Ollama."""
    model = os.environ.get("OLLAMA_MODEL", "alibayram/medgemma")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    print(f"[1/5] Pulling Ollama LLM model: {model}")
    print(f"      Ollama URL: {ollama_url}")

    try:
        import httpx

        resp = httpx.post(
            f"{ollama_url}/api/pull",
            json={"name": model, "stream": False},
            timeout=600,
        )
        if resp.status_code == 200:
            print(f"      Model '{model}' pulled successfully.")
            return True
        else:
            print(f"      ERROR: Ollama returned status {resp.status_code}")
            print(f"      Response: {resp.text[:500]}")
            return False
    except Exception as e:
        print(f"      ERROR: Could not connect to Ollama: {e}")
        print("      Make sure Ollama is running (docker-compose up ollama)")
        return False


def pull_embedding_model() -> bool:
    """Pull the embedding model into Ollama."""
    model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    print(f"\n[2/5] Pulling Ollama embedding model: {model}")

    try:
        import httpx

        resp = httpx.post(
            f"{ollama_url}/api/pull",
            json={"name": model, "stream": False},
            timeout=600,
        )
        if resp.status_code == 200:
            print(f"      Model '{model}' pulled successfully.")
            return True
        else:
            print(f"      ERROR: Ollama returned status {resp.status_code}")
            print(f"      Response: {resp.text[:500]}")
            return False
    except Exception as e:
        print(f"      ERROR: Could not connect to Ollama: {e}")
        return False


def download_reranker_model() -> bool:
    """Pre-download the FlashRank reranker model."""
    print("\n[3/5] Downloading FlashRank reranker model...")

    try:
        from flashrank import Ranker

        Ranker()  # Downloads model on first instantiation
        print("      FlashRank reranker model: CACHED")
        return True
    except Exception as e:
        print(f"      FlashRank reranker model: FAILED ({e})")
        return False


def verify_embedding() -> bool:
    """Verify the embedding model works via Ollama."""
    model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    print("\n[4/5] Verifying embedding model...")

    try:
        import httpx

        resp = httpx.post(
            f"{ollama_url}/api/embed",
            json={"model": model, "input": ["test sentence"]},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            dim = len(data["embeddings"][0])
            print(f"      Embedding model '{model}': OK ({dim}-dimensional vectors)")
            return True
        else:
            print(f"      ERROR: Ollama embed returned status {resp.status_code}")
            return False
    except Exception as e:
        print(f"      ERROR: Embedding verification failed: {e}")
        return False


def verify_system() -> bool:
    """Verify all models are available for offline operation."""
    print("\n[5/5] Verifying offline readiness...")

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    llm_model = os.environ.get("OLLAMA_MODEL", "alibayram/medgemma")
    embed_model = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")

    checks_passed = True

    # Check Ollama models are available
    try:
        import httpx

        resp = httpx.get(f"{ollama_url}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            for needed in [llm_model, embed_model]:
                if any(needed in m for m in models):
                    print(f"      Ollama model '{needed}': AVAILABLE")
                else:
                    print(
                        f"      Ollama model '{needed}': NOT FOUND (available: {models})"
                    )
                    checks_passed = False
        else:
            print(f"      Ollama API: ERROR (status {resp.status_code})")
            checks_passed = False
    except Exception as e:
        print(f"      Ollama API: UNREACHABLE ({e})")
        checks_passed = False

    return checks_passed


def main():
    print("=" * 60)
    print("KaagapAI Offline Setup")
    print("=" * 60)
    print()

    ollama_ok = pull_ollama_model()
    embedding_ok = pull_embedding_model()
    reranker_ok = download_reranker_model()
    embed_verify_ok = verify_embedding()
    system_ok = verify_system()

    print()
    print("=" * 60)
    if ollama_ok and embedding_ok and reranker_ok and embed_verify_ok and system_ok:
        print("SUCCESS: KaagapAI is ready for offline operation.")
        print("You can now disconnect from the internet.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("WARNING: Some checks failed. Review errors above.")
        print("The system may not operate fully offline.")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
