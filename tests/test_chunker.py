import pytest
from unittest.mock import MagicMock, patch
from src.rag.chunker import PDFParser, SmartChunker, ChunkMetadata

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
    p.write_text("dummy content", encoding="utf-8") # Real content not read by mock
    return str(p)

def test_chunk_metadata_structure():
    meta = ChunkMetadata(
        source="test.pdf",
        page_number=1,
        chunk_index=0,
        section_title="Introduction"
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
        assert len(chunk.content) <= 200 # Allowing some leeway for token estimation vs chars

def test_chunking_metadata_assignment():
    # Use smaller chunk size to ensure multiple chunks
    chunker = SmartChunker(chunk_size=50, chunk_overlap=10)

    # Create text long enough to span multiple chunks
    long_text = "# Section 1\n" + "This is some content. " * 20 + "\n# Section 2\n" + "More content here. " * 20
    chunks = chunker.chunk(long_text, source="test.pdf")

    # Should have at least 2 chunks with this text and chunk size
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    assert chunks[0].metadata.source == "test.pdf"
    assert chunks[0].metadata.chunk_index == 0
    assert chunks[1].metadata.chunk_index == 1
