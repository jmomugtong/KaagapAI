#!/usr/bin/env python3
"""
One-time setup script for offline hospital deployment.

Run this ONCE while internet is available to download all models
and verify the system can operate fully offline afterwards.

Usage:
    python scripts/setup_offline.py

What it does:
    1. Pulls the LLM model (phi3:mini) into Ollama
    2. Pre-downloads the sentence-transformers embedding model
    3. Verifies both models are available
    4. Prints confirmation that the system can operate offline
"""

import os
import subprocess
import sys


def pull_ollama_model() -> bool:
    """Pull the configured LLM model into Ollama."""
    model = os.environ.get("OLLAMA_MODEL", "phi3:mini")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    print(f"[1/3] Pulling Ollama model: {model}")
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


def download_embedding_model() -> bool:
    """Pre-download the sentence-transformers embedding model."""
    model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    print(f"\n[2/3] Downloading embedding model: {model_name}")

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        # Quick test to verify it works
        test_embedding = model.encode(["test sentence"])
        dim = len(test_embedding[0])
        print(f"      Model downloaded and verified ({dim}-dimensional vectors).")
        return True
    except Exception as e:
        print(f"      ERROR: Failed to download embedding model: {e}")
        return False


def verify_system() -> bool:
    """Verify both models are available for offline operation."""
    print("\n[3/3] Verifying offline readiness...")

    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "phi3:mini")

    checks_passed = True

    # Check Ollama model is available
    try:
        import httpx

        resp = httpx.get(f"{ollama_url}/api/tags", timeout=10)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(model in m for m in models):
                print(f"      Ollama model '{model}': AVAILABLE")
            else:
                print(f"      Ollama model '{model}': NOT FOUND (available: {models})")
                checks_passed = False
        else:
            print(f"      Ollama API: ERROR (status {resp.status_code})")
            checks_passed = False
    except Exception as e:
        print(f"      Ollama API: UNREACHABLE ({e})")
        checks_passed = False

    # Check embedding model is cached
    try:
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
        print("      Embedding model: CACHED")
    except Exception as e:
        print(f"      Embedding model: NOT CACHED ({e})")
        checks_passed = False

    return checks_passed


def main():
    print("=" * 60)
    print("MedQuery Offline Setup")
    print("=" * 60)
    print()

    ollama_ok = pull_ollama_model()
    embedding_ok = download_embedding_model()
    system_ok = verify_system()

    print()
    print("=" * 60)
    if ollama_ok and embedding_ok and system_ok:
        print("SUCCESS: MedQuery is ready for offline operation.")
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
