-- ============================================
-- MedQuery Database Initialization Script
-- ============================================
-- Run automatically by PostgreSQL Docker container
-- ============================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- Tables
-- ============================================

-- Hospitals table (for multi-tenancy)
CREATE TABLE IF NOT EXISTS hospitals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    code VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    hospital_id UUID REFERENCES hospitals(id),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Clinical documents table
CREATE TABLE IF NOT EXISTS clinical_docs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    hospital_id UUID NOT NULL REFERENCES hospitals(id),
    filename VARCHAR(255) NOT NULL,
    document_type VARCHAR(50) NOT NULL,
    upload_timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    uploader_user_id UUID REFERENCES users(id),
    total_chunks INTEGER,
    metadata JSONB,
    file_path TEXT NOT NULL,
    file_size_bytes BIGINT,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Embeddings cache table
CREATE TABLE IF NOT EXISTS embeddings_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_hash VARCHAR(64) UNIQUE NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(384) NOT NULL,
    document_id UUID REFERENCES clinical_docs(id) ON DELETE CASCADE,
    chunk_index INTEGER,
    section_title TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Queries log table
CREATE TABLE IF NOT EXISTS queries_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id),
    hospital_id UUID REFERENCES hospitals(id),
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64),
    response_text TEXT,
    confidence_score FLOAT,
    processing_time_ms INTEGER,
    cache_hit BOOLEAN,
    hallucination_detected BOOLEAN,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    citations JSONB,
    retrieved_doc_ids UUID[]
);

-- User feedback table
CREATE TABLE IF NOT EXISTS user_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_log_id UUID REFERENCES queries_log(id),
    user_id UUID REFERENCES users(id),
    rating INTEGER CHECK (rating BETWEEN 1 AND 5),
    feedback_text TEXT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ============================================
-- Indexes
-- ============================================

-- Clinical docs indexes
CREATE INDEX IF NOT EXISTS idx_clinical_docs_hospital_id ON clinical_docs(hospital_id);
CREATE INDEX IF NOT EXISTS idx_clinical_docs_document_type ON clinical_docs(document_type);
CREATE INDEX IF NOT EXISTS idx_clinical_docs_upload_timestamp ON clinical_docs(upload_timestamp);

-- Embeddings cache indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_hash ON embeddings_cache(chunk_hash);
CREATE INDEX IF NOT EXISTS idx_embeddings_document_id ON embeddings_cache(document_id);

-- Vector similarity index (IVFFlat for approximate nearest neighbor search)
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings_cache 
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Queries log indexes
CREATE INDEX IF NOT EXISTS idx_queries_user_id ON queries_log(user_id);
CREATE INDEX IF NOT EXISTS idx_queries_hospital_id ON queries_log(hospital_id);
CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_queries_hallucination ON queries_log(hallucination_detected);

-- User feedback indexes
CREATE INDEX IF NOT EXISTS idx_feedback_query_id ON user_feedback(query_log_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON user_feedback(user_id);

-- ============================================
-- Row Level Security (for multi-tenancy)
-- ============================================

-- Enable RLS on tables
ALTER TABLE clinical_docs ENABLE ROW LEVEL SECURITY;
ALTER TABLE queries_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_feedback ENABLE ROW LEVEL SECURITY;

-- Note: Actual RLS policies should be created based on your auth implementation
-- Example policy (commented out - implement based on your needs):
-- CREATE POLICY hospital_isolation ON clinical_docs
--     USING (hospital_id = current_setting('app.current_hospital_id')::UUID);

-- ============================================
-- Functions
-- ============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_hospitals_updated_at
    BEFORE UPDATE ON hospitals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_clinical_docs_updated_at
    BEFORE UPDATE ON clinical_docs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- Seed Data (for development)
-- ============================================

-- Insert a default hospital for development
INSERT INTO hospitals (id, name, code)
VALUES ('00000000-0000-0000-0000-000000000001', 'Development Hospital', 'DEV')
ON CONFLICT (code) DO NOTHING;

-- ============================================
-- Grants
-- ============================================

-- Grant permissions to the application user
-- Note: Replace 'medquery_user' with your actual DB user if different
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO medquery_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO medquery_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO medquery_user;
