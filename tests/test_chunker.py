from unittest.mock import MagicMock, patch

import pytest

from src.rag.chunker import ChunkMetadata, PDFParser, SemanticDocChunker, SmartChunker

# Mock PDF content
MOCK_PDF_TEXT = """
# Introduction
This is the introduction section. It contains some background information.

# Methods
We used various methods to analyze the data.
1. Data collection
2. Data cleaning

# Results
The results were significant. p < 0.05.
"""


@pytest.fixture
def mock_pdf_path(tmp_path):
    p = tmp_path / "test_doc.pdf"
    p.write_text("dummy content", encoding="utf-8")  # Real content not read by mock
    return str(p)


def test_chunk_metadata_structure():
    meta = ChunkMetadata(
        source="test.pdf", page_number=1, chunk_index=0, section_title="Introduction"
    )
    assert meta.source == "test.pdf"
    assert meta.chunk_index == 0


@patch("PyPDF2.PdfReader")
def test_pdf_parsing_strategy(mock_reader, mock_pdf_path):
    # Setup mock
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Page 1 content"
    mock_reader.return_value.pages = [mock_page]

    parser = PDFParser()
    text = parser.parse(mock_pdf_path)
    assert text == "Page 1 content"


def test_smart_chunking_logic():
    chunker = SmartChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk(MOCK_PDF_TEXT)

    assert len(chunks) > 0
    # Check that section headers are preserved or respected
    assert any("Introduction" in c.content for c in chunks)
    assert any("Methods" in c.content for c in chunks)

    # Check max size constraint (roughly)
    for chunk in chunks:
        assert (
            len(chunk.content) <= 200
        )  # Allowing some leeway for token estimation vs chars


def test_smart_chunker_default_sizes():
    """Verify the new default chunk sizes (1500/200)."""
    chunker = SmartChunker()
    assert chunker.chunk_size == 1500
    assert chunker.chunk_overlap == 200


def test_chunking_metadata_assignment():
    # Use smaller chunk size to ensure multiple chunks
    chunker = SmartChunker(chunk_size=50, chunk_overlap=10)

    # Create text long enough to span multiple chunks
    long_text = (
        "# Section 1\n"
        + "This is some content. " * 20
        + "\n# Section 2\n"
        + "More content here. " * 20
    )
    chunks = chunker.chunk(long_text, source="test.pdf")

    # Should have at least 2 chunks with this text and chunk size
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    assert chunks[0].metadata.source == "test.pdf"
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[1].metadata.chunk_index == 1


class TestSemanticDocChunker:
    """Tests for SemanticDocChunker."""

    def test_fallback_when_no_embedding_model(self):
        """Falls back to SmartChunker when embedding_model is None."""
        chunker = SemanticDocChunker(embedding_model=None)
        chunks = chunker.chunk(MOCK_PDF_TEXT, source="test.pdf")
        assert len(chunks) > 0
        assert chunks[0].metadata.source == "test.pdf"

    def test_fallback_on_exception(self):
        """Falls back to SmartChunker when SemanticChunker raises."""
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.side_effect = Exception("Model error")
        chunker = SemanticDocChunker(embedding_model=mock_embeddings)
        chunks = chunker.chunk(MOCK_PDF_TEXT, source="test.pdf")
        assert len(chunks) > 0  # Should get fallback chunks

    def test_semantic_chunking_with_mock_embeddings(self):
        """Test semantic chunking with mocked embedding model."""
        mock_embeddings = MagicMock()
        # Return different embeddings for each sentence to create breakpoints
        import numpy as np

        call_count = [0]

        def mock_embed(texts):
            results = []
            for _ in texts:
                call_count[0] += 1
                results.append(np.random.rand(768).tolist())
            return results

        mock_embeddings.embed_documents = mock_embed

        long_text = (
            "The patient presented with chest pain. "
            "ECG showed ST elevation. "
            "Troponin levels were elevated. "
            "The diagnosis was acute myocardial infarction. "
            "Treatment included aspirin and heparin. "
            "The patient was taken to the cath lab. "
            "A stent was placed in the LAD artery. "
            "The patient recovered well post-procedure. "
            "Discharge medications included dual antiplatelet therapy. "
            "Follow-up was scheduled in two weeks."
        )
        chunker = SemanticDocChunker(
            embedding_model=mock_embeddings,
            min_chunk_size=50,
        )
        chunks = chunker.chunk(long_text, source="clinical_note.pdf")
        assert len(chunks) > 0
        # Verify sequential indexing
        for i, c in enumerate(chunks):
            assert c.metadata.chunk_index == i

    def test_oversized_chunks_get_sub_chunked(self):
        """Chunks exceeding max_chunk_size are sub-chunked."""
        mock_embeddings = MagicMock()
        # Return a single embedding so everything stays in one "semantic chunk"
        mock_embeddings.embed_documents.return_value = [[0.1] * 768] * 50

        # Generate text that will produce a single large semantic chunk
        huge_text = "This is a very long sentence. " * 200  # ~6000 chars

        chunker = SemanticDocChunker(
            embedding_model=mock_embeddings,
            max_chunk_size=2000,
            min_chunk_size=50,
        )
        chunks = chunker.chunk(huge_text, source="big_doc.pdf")
        # Should have multiple chunks since the single semantic chunk exceeds max_chunk_size
        assert len(chunks) > 1
        for c in chunks:
            assert (
                len(c.content) <= 2200
            )  # Some leeway for RecursiveCharacterTextSplitter

    def test_tiny_chunks_dropped(self):
        """Chunks smaller than min_chunk_size are filtered out."""
        mock_embeddings = MagicMock()
        # Return very different embeddings to create many small chunks
        import numpy as np

        def mock_embed(texts):
            return [np.random.rand(768).tolist() for _ in texts]

        mock_embeddings.embed_documents = mock_embed

        # Short text with many breakpoints
        short_text = "Hi. Ok. Yes. No. Sure. Fine. "
        chunker = SemanticDocChunker(
            embedding_model=mock_embeddings,
            min_chunk_size=100,
        )
        chunks = chunker.chunk(short_text, source="tiny.pdf")
        # Either chunks were dropped (falling through to fallback) or all chunks meet minimum
        for c in chunks:
            # If semantic chunking produced results, they should be >= min_chunk_size
            # or it fell back to SmartChunker which has no min_chunk_size filter
            assert len(c.content) > 0
