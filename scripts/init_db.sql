-- ============================================
-- MedQuery Database Initialization Script
-- ============================================
-- Run automatically by PostgreSQL Docker container on first start.
-- Only creates extensions and grants permissions.
-- Table creation is handled by the ORM via Base.metadata.create_all().
-- ============================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation (may be needed later)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- Grants
-- ============================================

-- Grant permissions to the application user
-- Note: Replace 'medquery_user' with your actual DB user if different
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO medquery_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO medquery_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO medquery_user;
