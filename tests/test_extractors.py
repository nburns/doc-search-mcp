import textwrap
from pathlib import Path

import pytest

from doc_search_mcp.extractors.text import extract_text


def test_plain_text_splits_on_blank_lines(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph.\n")
    chunks = extract_text(f)
    assert len(chunks) == 3
    assert chunks[0].text == "First paragraph here."
    assert chunks[1].text == "Second paragraph here."
    assert chunks[2].text == "Third paragraph."


def test_plain_text_line_labels(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("Line one.\nLine two.\n\nLine three.\n")
    chunks = extract_text(f)
    # First chunk spans lines 1-2, second is line 4
    assert "Lines" in chunks[0].page_or_section or "Line" in chunks[0].page_or_section


def test_plain_text_positions_are_sequential(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("A\n\nB\n\nC\n")
    chunks = extract_text(f)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_plain_text_empty_file_raises(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("")
    with pytest.raises(ValueError, match="empty"):
        extract_text(f)


def test_plain_text_whitespace_only_raises(tmp_path):
    f = tmp_path / "blank.txt"
    f.write_text("   \n\n   \n")
    with pytest.raises(ValueError, match="empty"):
        extract_text(f)


def test_markdown_splits_on_headings(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text(textwrap.dedent("""\
        # Title

        Intro text here.

        ## Section One

        Content of section one.

        ## Section Two

        Content of section two.
    """))
    chunks = extract_text(f)
    labels = [c.page_or_section for c in chunks]
    # Should have sections labeled with headings
    assert any("Section One" in label for label in labels)
    assert any("Section Two" in label for label in labels)


def test_markdown_positions_sequential(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("## A\n\ntext a\n\n## B\n\ntext b\n")
    chunks = extract_text(f)
    assert [c.position for c in chunks] == list(range(len(chunks)))


def test_markdown_no_headings_treated_as_plain(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("Just some text.\n\nAnother paragraph.\n")
    chunks = extract_text(f)
    # Should still produce chunks
    assert len(chunks) >= 1
