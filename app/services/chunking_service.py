"""Page-wise chunking using LangChain RecursiveCharacterTextSplitter.

Each page is chunked independently so page-level citations stay accurate:
no chunk ever mixes content from different PDF pages.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.services.language_service import LanguageService

MULTILINGUAL_SEPARATORS = [
    "\n\n",
    "\n",
    "।\n",
    "। ",
    "।",
    "॥\n",
    "॥ ",
    "॥",
    ".\n",
    ". ",
    "? ",
    "! ",
    " ",
    "",
]


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    chunk_index: int
    text: str
    language: str = "unknown"


class ChunkingService:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            separators=MULTILINGUAL_SEPARATORS,
            length_function=len,
        )

    def chunk_pages(self, pages: list[tuple[int, str]]) -> list[TextChunk]:
        """Chunk cleaned page texts with multilingual boundary preservation.

        Args:
            pages: list of (page_number, cleaned_text) tuples.

        Returns:
            Chunks with a document-wide sequential chunk_index, language metadata,
            and the originating page_number preserved on every chunk.
        """
        chunks: list[TextChunk] = []
        chunk_index = 0
        for page_number, text in pages:
            if not text or not text.strip():
                continue
            for piece in self._splitter.split_text(text):
                piece = piece.strip()
                if not piece:
                    continue
                chunk_lang = LanguageService.detect_language(piece)
                chunks.append(
                    TextChunk(
                        page_number=page_number,
                        chunk_index=chunk_index,
                        text=piece,
                        language=chunk_lang,
                    )
                )
                chunk_index += 1
        return chunks
