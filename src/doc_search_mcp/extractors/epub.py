from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub


@dataclass
class RawChunk:
    text: str
    page_or_section: str
    position: int


def extract_epub(path: Path) -> list[RawChunk]:
    """Extract text chunks from an EPUB, one chunk per chapter/document item."""
    book = epub.read_epub(str(path), options={"ignore_ncx": False})
    chunks = []
    position = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        title = _item_title(book, item) or f"Chapter {position + 1}"
        text = _html_to_text(item.get_content())
        if text:
            chunks.append(RawChunk(text=text, page_or_section=title, position=position))
            position += 1

    if not chunks:
        raise ValueError(f"No text extracted from {path.name} - EPUB may have no readable content")
    return chunks


def _html_to_text(html: bytes) -> str:
    soup = BeautifulSoup(html, "html.parser")
    # Remove script/style elements
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    lines = [line.strip() for line in text.splitlines()]
    paragraphs = []
    blank_count = 0
    for line in lines:
        if line:
            blank_count = 0
            paragraphs.append(line)
        else:
            blank_count += 1
            if blank_count == 1:
                paragraphs.append("")
    return "\n".join(paragraphs).strip()


def _item_title(book: epub.EpubBook, item: epub.EpubItem) -> str | None:
    """Try to find a human-readable title for a spine item via the NCX/TOC."""
    item_name = item.get_name()
    for link in book.toc:
        title = _find_title_in_toc(link, item_name)
        if title:
            return title
    return None


def _find_title_in_toc(node, item_name: str) -> str | None:
    if isinstance(node, epub.Link):
        href = node.href.split("#")[0]
        if href == item_name:
            return node.title
    elif isinstance(node, tuple):
        section, children = node
        href = section.href.split("#")[0] if hasattr(section, "href") else ""
        if href == item_name:
            return section.title
        for child in children:
            result = _find_title_in_toc(child, item_name)
            if result:
                return result
    return None
