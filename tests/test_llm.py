"""
Tests for KaagapAI LLM Integration (Phase 5)

Tests for:
- OllamaClient: async HTTP client with retries, timeouts, health checks
- PromptTemplate: clinical QA prompt formatting
- ResponseParser: answer extraction, confidence scoring, citation parsing
"""

import httpx
import pytest

from src.llm.ollama_client import OllamaClient
from src.llm.prompt_templates import PromptTemplate
from src.llm.response_parser import ParsedResponse, ResponseParser

# ============================================
# OllamaClient Tests
# ============================================


class TestOllamaClientInit:
    """Tests for OllamaClient initialization and configuration."""

    @pytest.mark.unit
    def test_default_configuration(self):
        """Client initializes with sensible defaults including LLM params."""
        client = OllamaClient()
        assert client.base_url == "http://localhost:11434"
        assert client.model == "alibayram/medgemma"
        assert client.timeout > 0
        assert client.temperature == 0.1
        assert client.max_tokens == 256
        assert client.top_p == 0.9
        assert client.num_ctx == 2048
        assert client.num_thread == 0
        assert client.keep_alive == "60m"

    @pytest.mark.unit
    def test_custom_configuration(self):
        """Client accepts custom base_url, model, timeout, and LLM params."""
        client = OllamaClient(
            base_url="http://localhost:11434",
            model="llama2",
            timeout=60,
            temperature=0.5,
            max_tokens=256,
            top_p=0.8,
            num_ctx=4096,
            num_thread=4,
        )
        assert client.base_url == "http://localhost:11434"
        assert client.model == "llama2"
        assert client.timeout == 60
        assert client.temperature == 0.5
        assert client.max_tokens == 256
        assert client.top_p == 0.8
        assert client.num_ctx == 4096
        assert client.num_thread == 4

    @pytest.mark.unit
    def test_trailing_slash_stripped(self):
        """Trailing slash on base_url is stripped."""
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"


class TestOllamaClientGenerate:
    """Tests for the generate method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_returns_string(self, mocker):
        """generate() returns the response text from Ollama."""
        client = OllamaClient(base_url="http://fake:11434")

        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test answer from LLM."}
        mock_response.raise_for_status = mocker.Mock()

        mock_client = mocker.AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await client.generate("What is hypertension?")
        assert result == "Test answer from LLM."

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_sends_correct_payload(self, mocker):
        """generate() sends model, prompt, stream=False, and options to Ollama API."""
        client = OllamaClient(base_url="http://fake:11434", model="mistral")

        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = mocker.Mock()

        mock_client = mocker.AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        await client.generate("test prompt")

        call_args = mock_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["model"] == "mistral"
        assert payload["prompt"] == "test prompt"
        assert payload["stream"] is False
        # Verify options are passed
        options = payload["options"]
        assert options["temperature"] == client.temperature
        assert options["top_p"] == client.top_p
        assert options["num_predict"] == client.max_tokens
        assert options["num_ctx"] == client.num_ctx

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_num_thread_omitted_when_zero(self, mocker):
        """num_thread is NOT included in options when set to 0 (auto)."""
        client = OllamaClient(base_url="http://fake:11434", num_thread=0)

        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = mocker.Mock()

        mock_client = mocker.AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        await client.generate("test")

        payload = mock_client.post.call_args[1]["json"]
        assert "num_thread" not in payload["options"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_num_thread_included_when_set(self, mocker):
        """num_thread IS included in options when set to a positive value."""
        client = OllamaClient(base_url="http://fake:11434", num_thread=4)

        mock_response = mocker.Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "answer"}
        mock_response.raise_for_status = mocker.Mock()

        mock_client = mocker.AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        await client.generate("test")

        payload = mock_client.post.call_args[1]["json"]
        assert payload["options"]["num_thread"] == 4

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_returns_empty_on_http_error(self, mocker):
        """generate() returns empty string when Ollama returns HTTP error."""
        client = OllamaClient(base_url="http://fake:11434")

        import httpx

        mock_client = mocker.AsyncMock()
        mock_client.post.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=mocker.Mock(),
            response=mocker.Mock(status_code=500),
        )
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await client.generate("test")
        assert result == ""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_returns_empty_on_timeout(self, mocker):
        """generate() returns empty string on connection timeout."""
        client = OllamaClient(base_url="http://fake:11434")

        import httpx

        mock_client = mocker.AsyncMock()
        mock_client.post.side_effect = httpx.ConnectTimeout("Connection timed out")
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await client.generate("test")
        assert result == ""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_returns_empty_on_connection_error(self, mocker):
        """generate() returns empty string when Ollama is unreachable."""
        client = OllamaClient(base_url="http://fake:11434")

        import httpx

        mock_client = mocker.AsyncMock()
        mock_client.post.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        result = await client.generate("test")
        assert result == ""


class TestOllamaClientHealthCheck:
    """Tests for the health_check method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_returns_true_when_healthy(self, mocker):
        """health_check() returns True when Ollama responds with 200."""
        client = OllamaClient(base_url="http://fake:11434")

        mock_response = mocker.AsyncMock()
        mock_response.status_code = 200

        mock_client = mocker.AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        assert await client.health_check() is True

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_down(self, mocker):
        """health_check() returns False when Ollama is unreachable."""
        client = OllamaClient(base_url="http://fake:11434")

        import httpx

        mock_client = mocker.AsyncMock()
        mock_client.get.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        assert await client.health_check() is False


# ============================================
# PromptTemplate Tests
# ============================================


class TestPromptTemplate:
    """Tests for clinical QA prompt template."""

    @pytest.mark.unit
    def test_build_prompt_contains_question(self):
        """Built prompt contains the user's question."""
        template = PromptTemplate()
        chunks = [{"text": "Some context.", "metadata": {}}]
        prompt = template.build(question="What is hypertension?", chunks=chunks)
        assert "What is hypertension?" in prompt

    @pytest.mark.unit
    def test_build_prompt_contains_context(self):
        """Built prompt contains chunk text as context."""
        template = PromptTemplate()
        chunks = [
            {
                "text": "Hypertension is high blood pressure exceeding 140/90 mmHg.",
                "metadata": {"source": "Cardiology Protocol v2.1"},
            }
        ]
        prompt = template.build(question="What is hypertension?", chunks=chunks)
        assert "Hypertension is high blood pressure exceeding 140/90 mmHg." in prompt

    @pytest.mark.unit
    def test_build_prompt_contains_citation_instructions(self):
        """Built prompt instructs the LLM to cite sources."""
        template = PromptTemplate()
        chunks = [{"text": "Context text.", "metadata": {}}]
        prompt = template.build(question="test?", chunks=chunks)
        assert "cite" in prompt.lower() or "citation" in prompt.lower()

    @pytest.mark.unit
    def test_build_prompt_contains_confidence_instructions(self):
        """Built prompt instructs the LLM to assign a confidence score."""
        template = PromptTemplate()
        chunks = [{"text": "Context text.", "metadata": {}}]
        prompt = template.build(question="test?", chunks=chunks)
        assert "confidence" in prompt.lower()

    @pytest.mark.unit
    def test_build_prompt_multiple_chunks(self):
        """Built prompt includes chunks up to the max_chunks limit."""
        template = PromptTemplate(max_chunks=3)
        chunks = [
            {"text": "Chunk one content.", "metadata": {"source": "Doc A"}},
            {"text": "Chunk two content.", "metadata": {"source": "Doc B"}},
            {"text": "Chunk three content.", "metadata": {"source": "Doc C"}},
        ]
        prompt = template.build(question="test?", chunks=chunks)
        assert "Chunk one content." in prompt
        assert "Chunk two content." in prompt
        assert "Chunk three content." in prompt

    @pytest.mark.unit
    def test_build_prompt_empty_chunks(self):
        """Built prompt handles empty chunks list gracefully."""
        template = PromptTemplate()
        prompt = template.build(question="test?", chunks=[])
        assert "test?" in prompt
        # Should still produce a valid prompt, just with no context
        assert len(prompt) > 0

    @pytest.mark.unit
    def test_build_prompt_includes_source_metadata(self):
        """Built prompt includes source names from metadata."""
        template = PromptTemplate()
        chunks = [
            {
                "text": "Protocol content here.",
                "metadata": {
                    "source": "Pain Management Protocol v3.2",
                    "page": 12,
                    "chunk_index": 5,
                },
            }
        ]
        prompt = template.build(question="test?", chunks=chunks)
        assert "Pain Management Protocol v3.2" in prompt

    @pytest.mark.unit
    def test_build_prompt_limits_chunk_count(self):
        """Passing 5 chunks with max_chunks=3 only includes the first 3."""
        template = PromptTemplate(max_chunks=3)
        chunks = [
            {"text": f"Chunk {i} content.", "metadata": {"source": f"Doc {i}"}}
            for i in range(5)
        ]
        prompt = template.build(question="test?", chunks=chunks)
        assert "Chunk 0 content." in prompt
        assert "Chunk 1 content." in prompt
        assert "Chunk 2 content." in prompt
        assert "Chunk 3 content." not in prompt
        assert "Chunk 4 content." not in prompt

    @pytest.mark.unit
    def test_build_prompt_truncates_long_chunks(self):
        """A 500-char chunk is truncated to max_chunk_chars with '...' suffix."""
        template = PromptTemplate(max_chunk_chars=300)
        long_text = "A" * 500
        chunks = [{"text": long_text, "metadata": {"source": "Doc"}}]
        prompt = template.build(question="test?", chunks=chunks)
        # The full 500-char text should NOT appear
        assert long_text not in prompt
        # The truncated version (300 chars + "...") should appear
        assert "A" * 300 + "..." in prompt

    @pytest.mark.unit
    def test_build_prompt_does_not_truncate_short_chunks(self):
        """A chunk within max_chunk_chars is not truncated."""
        template = PromptTemplate(max_chunk_chars=300)
        short_text = "Short clinical text."
        chunks = [{"text": short_text, "metadata": {"source": "Doc"}}]
        prompt = template.build(question="test?", chunks=chunks)
        assert short_text in prompt
        assert "..." not in prompt


# ============================================
# ResponseParser Tests
# ============================================


class TestResponseParserConfidence:
    """Tests for confidence score extraction."""

    @pytest.mark.unit
    def test_extracts_confidence_from_response(self):
        """Parser extracts confidence score from LLM output."""
        parser = ResponseParser()
        text = (
            "Based on the evidence, acetaminophen is recommended.\n" "Confidence: 0.92"
        )
        result = parser.parse(text, retrieved_chunks=[])
        assert result.confidence == 0.92

    @pytest.mark.unit
    def test_extracts_confidence_with_score_label(self):
        """Parser extracts confidence from 'Confidence Score: X.XX' format."""
        parser = ResponseParser()
        text = "Answer here.\nConfidence Score: 0.85"
        result = parser.parse(text, retrieved_chunks=[])
        assert result.confidence == 0.85

    @pytest.mark.unit
    def test_default_confidence_when_missing(self):
        """Parser returns default confidence when not found in response."""
        parser = ResponseParser()
        text = "Just an answer with no confidence mentioned."
        result = parser.parse(text, retrieved_chunks=[])
        assert result.confidence == 0.5

    @pytest.mark.unit
    def test_confidence_clamped_to_valid_range(self):
        """Confidence is clamped to 0.0-1.0 range."""
        parser = ResponseParser()
        text = "Answer here.\nConfidence: 1.5"
        result = parser.parse(text, retrieved_chunks=[])
        assert result.confidence <= 1.0

    @pytest.mark.unit
    def test_confidence_zero_floor(self):
        """Confidence below 0.0 is clamped to 0.0."""
        parser = ResponseParser()
        text = "Answer here.\nConfidence: -0.3"
        result = parser.parse(text, retrieved_chunks=[])
        assert result.confidence >= 0.0


class TestResponseParserCitations:
    """Tests for citation extraction and validation."""

    @pytest.mark.unit
    def test_extracts_bracket_citations(self):
        """Parser extracts citations in [Document, Section] format."""
        parser = ResponseParser()
        text = (
            "Acetaminophen 1000mg is recommended "
            "[Pain Protocol v3.2, Section 4, p. 12].\n"
            "Confidence: 0.90"
        )
        result = parser.parse(text, retrieved_chunks=[])
        assert len(result.citations) >= 1
        assert result.citations[0].document == "Pain Protocol v3.2"

    @pytest.mark.unit
    def test_extracts_multiple_citations(self):
        """Parser extracts multiple citations from response."""
        parser = ResponseParser()
        text = (
            "Use acetaminophen [Pain Protocol v3.2, Section 4, p. 12] "
            "and consider ibuprofen [Anti-Inflammatory Guide, Section 2, p. 5].\n"
            "Confidence: 0.88"
        )
        result = parser.parse(text, retrieved_chunks=[])
        assert len(result.citations) >= 2

    @pytest.mark.unit
    def test_citation_without_page(self):
        """Parser handles citations without page numbers."""
        parser = ResponseParser()
        text = "See guidelines [Cardiology Protocol, Section 3].\n" "Confidence: 0.80"
        result = parser.parse(text, retrieved_chunks=[])
        assert len(result.citations) >= 1
        assert result.citations[0].document == "Cardiology Protocol"

    @pytest.mark.unit
    def test_no_citations_in_response(self):
        """Parser returns empty citations when none found."""
        parser = ResponseParser()
        text = "Just a plain answer without any citations.\nConfidence: 0.60"
        result = parser.parse(text, retrieved_chunks=[])
        assert result.citations == []

    @pytest.mark.unit
    def test_validates_citations_against_retrieval_set(self):
        """Parser flags citations that don't match any retrieved chunk source."""
        parser = ResponseParser()
        text = "Answer based on [Fake Document, Section 1].\n" "Confidence: 0.85"
        retrieved = [
            {"text": "chunk text", "source": "Real Protocol v1.0"},
        ]
        result = parser.parse(text, retrieved_chunks=retrieved)
        assert result.has_hallucinated_citations is True

    @pytest.mark.unit
    def test_valid_citations_not_flagged(self):
        """Parser does not flag hallucination when citations match retrieval set."""
        parser = ResponseParser()
        text = "Answer based on [Pain Protocol, Section 4].\n" "Confidence: 0.90"
        retrieved = [
            {"text": "chunk text", "source": "Pain Protocol"},
        ]
        result = parser.parse(text, retrieved_chunks=retrieved)
        assert result.has_hallucinated_citations is False

    @pytest.mark.unit
    def test_no_citations_means_no_hallucination(self):
        """No citations in response means no hallucination flagged."""
        parser = ResponseParser()
        text = "Plain answer.\nConfidence: 0.70"
        result = parser.parse(text, retrieved_chunks=[])
        assert result.has_hallucinated_citations is False


class TestResponseParserAnswer:
    """Tests for answer text extraction."""

    @pytest.mark.unit
    def test_extracts_answer_text(self):
        """Parser extracts the answer portion without confidence line."""
        parser = ResponseParser()
        text = "Acetaminophen is recommended for pain.\nConfidence: 0.90"
        result = parser.parse(text, retrieved_chunks=[])
        assert "Acetaminophen is recommended for pain." in result.answer
        assert "Confidence: 0.90" not in result.answer

    @pytest.mark.unit
    def test_empty_llm_response(self):
        """Parser handles empty LLM response gracefully."""
        parser = ResponseParser()
        result = parser.parse("", retrieved_chunks=[])
        assert result.answer == ""
        assert result.confidence == 0.5
        assert result.citations == []

    @pytest.mark.unit
    def test_parsed_response_has_all_fields(self):
        """ParsedResponse contains answer, confidence, citations, hallucination flag."""
        parser = ResponseParser()
        text = "Take acetaminophen [Pain Protocol, Section 1].\n" "Confidence: 0.88"
        result = parser.parse(text, retrieved_chunks=[])
        assert isinstance(result, ParsedResponse)
        assert isinstance(result.answer, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.citations, list)
        assert isinstance(result.has_hallucinated_citations, bool)


class TestResponseParserLowConfidence:
    """Tests for low-confidence behavior."""

    @pytest.mark.unit
    def test_low_confidence_flagged(self):
        """Responses below 0.70 confidence are identifiable."""
        parser = ResponseParser()
        text = "Uncertain answer.\nConfidence: 0.45"
        result = parser.parse(text, retrieved_chunks=[])
        assert result.confidence < 0.70

    @pytest.mark.unit
    def test_high_confidence_not_flagged(self):
        """Responses at or above 0.70 confidence are normal."""
        parser = ResponseParser()
        text = "Confident answer.\nConfidence: 0.92"
        result = parser.parse(text, retrieved_chunks=[])
        assert result.confidence >= 0.70


# ============================================
# Streaming Tests
# ============================================


class TestOllamaClientStream:
    """Tests for the generate_stream method."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_stream_yields_tokens(self, mocker):
        """generate_stream() yields individual tokens from NDJSON lines."""
        import json
        from contextlib import asynccontextmanager

        client = OllamaClient(base_url="http://fake:11434")

        lines = [
            json.dumps({"response": "Hello", "done": False}),
            json.dumps({"response": " world", "done": False}),
            json.dumps({"response": "", "done": True}),
        ]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_response = mocker.Mock()
        mock_response.raise_for_status = mocker.Mock()
        mock_response.aiter_lines = fake_aiter_lines

        @asynccontextmanager
        async def fake_stream(*args, **kwargs):
            yield mock_response

        mock_client = mocker.AsyncMock()
        mock_client.stream = fake_stream
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        tokens = []
        async for token in client.generate_stream("test prompt"):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_generate_stream_fallback_on_error(self, mocker):
        """generate_stream() falls back to non-streaming on connection error."""
        client = OllamaClient(base_url="http://fake:11434")

        # Make the entire AsyncClient context manager raise on stream
        mock_client = mocker.AsyncMock()
        mock_client.stream.side_effect = httpx.ConnectError("Connection refused")
        mock_client.__aenter__ = mocker.AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = mocker.AsyncMock(return_value=False)

        mocker.patch("httpx.AsyncClient", return_value=mock_client)

        # Mock the non-streaming generate to return a fallback
        mocker.patch.object(client, "generate", return_value="Fallback response")

        tokens = []
        async for token in client.generate_stream("test"):
            tokens.append(token)

        assert tokens == ["Fallback response"]
