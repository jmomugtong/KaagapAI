from dataclasses import dataclass

import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter


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
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()


class SmartChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def chunk(self, text: str, source: str = "unknown") -> list[Chunk]:
        # Simple splitting for now. Section headers detection requires more complex logic.
        # For MVP, we use recursive splitter and dummy section titles or try to find headers regex.

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
