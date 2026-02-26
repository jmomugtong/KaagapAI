import logging
from dataclasses import dataclass

import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    source: str
    page_number: int
    chunk_index: int
    section_title: str = "Unknown"


@dataclass
class Chunk:
    content: str
    metadata: ChunkMetadata


class PDFParser:
    def parse(self, file_path: str) -> str:
        with open(file_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()


class SmartChunker:
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "ASSESSMENT:",
                "PLAN:",
                "HISTORY:",
                "DIAGNOSIS:",
                "MEDICATIONS:",
                "ALLERGIES:",
                "PROCEDURES:",
                "\n\n",
                "\n",
                ".",
                " ",
                "",
            ],
        )

    def chunk(self, text: str, source: str = "unknown") -> list[Chunk]:
        # Split text
        text_chunks = self.splitter.split_text(text)

        chunks = []
        for i, content in enumerate(text_chunks):
            # TODO: Improve section extraction
            section = "Unknown"
            if "# " in content:
                # Naive section detection
                lines = content.split("\n")
                for line in lines:
                    if line.startswith("# "):
                        section = line.strip("# ").strip()
                        break

            metadata = ChunkMetadata(
                source=source,
                page_number=1,  # Todo: map back to pages if possible, but hard with simple text split
                chunk_index=i,
                section_title=section,
            )
            chunks.append(Chunk(content=content, metadata=metadata))

        return chunks


class SemanticDocChunker:
    """Semantic chunking that groups sentences by embedding similarity.

    Uses LangChain's SemanticChunker to split text at natural semantic
    boundaries instead of fixed character counts. Falls back to
    SmartChunker if embedding model is unavailable or on error.
    """

    def __init__(
        self,
        embedding_model=None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
        min_chunk_size: int = 150,
        max_chunk_size: int = 2000,
    ):
        self._embedding_model = embedding_model
        self._breakpoint_type = breakpoint_threshold_type
        self._breakpoint_amount = breakpoint_threshold_amount
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._fallback = SmartChunker()

    def chunk(self, text: str, source: str = "unknown") -> list[Chunk]:
        if self._embedding_model is None:
            return self._fallback.chunk(text, source)

        try:
            from langchain_experimental.text_splitter import SemanticChunker

            splitter = SemanticChunker(
                self._embedding_model,
                breakpoint_threshold_type=self._breakpoint_type,
                breakpoint_threshold_amount=self._breakpoint_amount,
            )
            documents = splitter.create_documents([text])

            chunks = []
            for doc in documents:
                content = doc.page_content

                # Enforce max chunk size by sub-chunking oversized results
                if len(content) > self._max_chunk_size:
                    sub_chunker = SmartChunker(
                        chunk_size=self._max_chunk_size,
                        chunk_overlap=100,
                    )
                    sub_chunks = sub_chunker.chunk(content, source)
                    chunks.extend(sub_chunks)
                    continue

                # Skip tiny chunks
                if len(content) < self._min_chunk_size:
                    continue

                section = "Unknown"
                if "# " in content:
                    lines = content.split("\n")
                    for line in lines:
                        if line.startswith("# "):
                            section = line.strip("# ").strip()
                            break

                metadata = ChunkMetadata(
                    source=source,
                    page_number=1,
                    chunk_index=0,  # Will be re-indexed below
                    section_title=section,
                )
                chunks.append(Chunk(content=content, metadata=metadata))

            # Re-index chunks sequentially
            for i, c in enumerate(chunks):
                c.metadata.chunk_index = i

            return chunks if chunks else self._fallback.chunk(text, source)

        except Exception as e:
            logger.warning("Semantic chunking failed, falling back to recursive: %s", e)
            return self._fallback.chunk(text, source)
