# ============================================
# MedQuery Makefile
# ============================================
# Common commands for development and deployment
# ============================================

.PHONY: help install install-dev test test-cov lint format type-check \
        run dev docker-up docker-down docker-build docker-logs \
        db-migrate db-upgrade db-downgrade db-revision \
        ollama-pull eval clean

# Default target
help:
	@echo "MedQuery - Production RAG System for Clinical Documentation"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install        Install production dependencies"
	@echo "  install-dev    Install development dependencies"
	@echo "  dev            Run development server with auto-reload"
	@echo "  run            Run production server"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-cov       Run tests with coverage report"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-int       Run integration tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run linters (ruff)"
	@echo "  format         Format code (black, isort)"
	@echo "  type-check     Run type checker (mypy)"
	@echo "  check          Run all checks (lint, format, type)"
	@echo ""
	@echo "Docker:"
	@echo "  docker-up      Start all services"
	@echo "  docker-down    Stop all services"
	@echo "  docker-build   Build Docker images"
	@echo "  docker-logs    View logs from all services"
	@echo "  docker-shell   Open shell in API container"
	@echo ""
	@echo "Database:"
	@echo "  db-migrate     Generate new migration"
	@echo "  db-upgrade     Apply all migrations"
	@echo "  db-downgrade   Rollback last migration"
	@echo ""
	@echo "AI/ML:"
	@echo "  ollama-pull    Download Ollama model (phi3:mini)"
	@echo "  eval           Run evaluation suite"
	@echo ""
	@echo "Utilities:"
	@echo "  clean          Remove cache and build files"
	@echo "  seed           Seed sample clinical documents"

# ============================================
# Development
# ============================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,test]"
	pre-commit install

dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

run:
	uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4

# ============================================
# Testing
# ============================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/ -v -m "unit"

test-int:
	pytest tests/ -v -m "integration"

test-fast:
	pytest tests/ -v -m "not slow"

# ============================================
# Code Quality
# ============================================

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

format:
	black src/ tests/
	isort src/ tests/

format-check:
	black --check src/ tests/
	isort --check-only src/ tests/

type-check:
	mypy src/

check: lint format-check type-check

# ============================================
# Docker
# ============================================

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-rebuild:
	docker-compose build --no-cache

docker-logs:
	docker-compose logs -f

docker-logs-api:
	docker-compose logs -f api

docker-shell:
	docker-compose exec api /bin/bash

docker-ps:
	docker-compose ps

docker-clean:
	docker-compose down -v --remove-orphans
	docker system prune -f

# ============================================
# Database
# ============================================

db-migrate:
ifndef msg
	$(error msg is required. Usage: make db-migrate msg="migration message")
endif
	alembic revision --autogenerate -m "$(msg)"

db-upgrade:
	alembic upgrade head

db-downgrade:
	alembic downgrade -1

db-history:
	alembic history

db-current:
	alembic current

# ============================================
# AI/ML
# ============================================

ollama-pull:
	docker-compose exec ollama ollama pull phi3:mini

ollama-list:
	docker-compose exec ollama ollama list

embedding-warmup:
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# ============================================
# Evaluation
# ============================================

eval:
	python scripts/run_evals.py

eval-check:
	python scripts/check_thresholds.py

# ============================================
# Seeding & Setup
# ============================================

seed:
	python scripts/seed_clinical_docs.py

init: docker-up
	@echo "Waiting for services to start..."
	@sleep 30
	make ollama-pull
	make embedding-warmup
	make db-upgrade
	@echo "MedQuery initialized successfully!"

health:
	curl -s http://localhost:8000/health | python -m json.tool

# ============================================
# Load Testing
# ============================================

load-test:
	k6 run k6/load_test.js

load-test-smoke:
	k6 run --vus 1 --duration 30s k6/load_test.js

# ============================================
# Utilities
# ============================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

# Create .env from example
env:
	cp .env.example .env
	@echo "Created .env file. Please update with your values."

# Generate secure secret key
secret-key:
	python -c "import secrets; print(secrets.token_hex(32))"

# ============================================
# CI/CD Helpers
# ============================================

ci-test:
	pytest tests/ -v --cov=src --cov-report=xml --junitxml=test-results.xml

ci-check: lint type-check test

# ============================================
# Documentation
# ============================================

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build
