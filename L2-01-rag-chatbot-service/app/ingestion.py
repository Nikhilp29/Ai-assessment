"""
Document ingestion: extract raw text from PDF/DOCX, then split it into
overlapping chunks that fit safely inside an LLM context window.

The splitter below re-implements the same idea as LangChain's
RecursiveCharacterTextSplitter: try to split on the largest "natural"
boundary first (paragraphs), and only fall back to a harder split
(sentences, then raw characters) if a piece is still too big. Overlap
between consecutive chunks keeps context from being lost at the boundary.

Implemented by hand (instead of importing langchain-text-splitters) so the
logic is fully visible and explainable in an interview -- swapping in the
LangChain splitter later is a one-line change if preferred.
"""
import io
from pypdf import PdfReader
from docx import Document as DocxDocument

from app.config import get_settings

settings = get_settings()

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class UnsupportedFileTypeError(Exception):
    pass


class EmptyDocumentError(Exception):
    pass


def extract_text(filename: str, file_bytes: bytes) -> str:
    """Extract raw text from an uploaded PDF or DOCX file."""
    ext = _get_extension(filename)

    if ext == ".pdf":
        text = _extract_pdf_text(file_bytes)
    elif ext == ".docx":
        text = _extract_docx_text(file_bytes)
    else:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    text = text.strip()
    if not text:
        raise EmptyDocumentError("No extractable text found in the document.")
    return text


def _get_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)
    return "\n\n".join(pages)


def _extract_docx_text(file_bytes: bytes) -> str:
    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def recursive_character_split(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into chunks of roughly `chunk_size` characters, preserving
    natural boundaries where possible, with `chunk_overlap` characters of
    context repeated between consecutive chunks.

    Split priority (largest/most natural boundary first):
      1. Double newline (paragraph break)
      2. Single newline
      3. Sentence-ending punctuation + space
      4. Raw character split (last resort, only if a "paragraph" is itself
         larger than chunk_size)
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if len(text) <= chunk_size:
        return [text]

    separators = ["\n\n", "\n", ". ", ""]
    pieces = _split_by_separators(text, separators, chunk_size)

    # Merge small pieces together up to chunk_size, adding overlap between chunks
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current:
                chunks.append(current.strip())
            # start new chunk with overlap tail from previous chunk
            overlap_text = current[-chunk_overlap:] if current else ""
            current = overlap_text + piece
    if current.strip():
        chunks.append(current.strip())

    return chunks


def _split_by_separators(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split `text` on the first separator that yields pieces
    small enough to fit in chunk_size; falls back to the next separator."""
    if not separators:
        return [text]

    sep = separators[0]
    if sep == "":
        # last resort: hard split by raw characters
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    result: list[str] = []
    for part in parts:
        piece = part + sep
        if len(piece) > chunk_size:
            # this piece is still too big, recurse with the next separator
            result.extend(_split_by_separators(piece, separators[1:], chunk_size))
        else:
            result.append(piece)
    return result
