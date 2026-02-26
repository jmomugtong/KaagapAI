-- ============================================
-- KaagapAI Database Initialization Script
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
-- Test Database (isolated from production data)
-- ============================================

-- Create a separate database for tests so pytest never touches production data
SELECT 'CREATE DATABASE kaagapai_test OWNER kaagapai_user'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'kaagapai_test')\gexec

-- Enable pgvector in the test database
\c kaagapai_test
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
\c kaagapai

-- ============================================
-- Grants
-- ============================================

-- Grant permissions to the application user
-- Note: Replace 'kaagapai_user' with your actual DB user if different
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO kaagapai_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO kaagapai_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO kaagapai_user;
