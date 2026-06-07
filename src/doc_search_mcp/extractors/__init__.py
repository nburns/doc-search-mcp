"""Document extractors for PDF, EPUB, and plain text/Markdown."""

from doc_search_mcp.extractors.epub import extract_epub
from doc_search_mcp.extractors.pdf import extract_pdf
from doc_search_mcp.extractors.text import extract_text

SUPPORTED_EXTENSIONS = {".pdf", ".epub", ".txt", ".md"}

__all__ = ["extract_pdf", "extract_epub", "extract_text", "SUPPORTED_EXTENSIONS"]
