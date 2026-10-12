"""
Tests for KnowledgeBase._chunk_text (pure function, no database needed).
Run: pytest -v
"""

from knowledge_base import KnowledgeBase

chunk = KnowledgeBase._chunk_text


def test_empty_text():
    assert chunk("") == []
    assert chunk("   \n\t ") == []


def test_short_text_is_one_chunk():
    assert chunk("One sentence. Another sentence.") == ["One sentence. Another sentence."]


def test_long_text_splits_and_respects_size():
    text = " ".join(f"This is sentence number {i}." for i in range(200))
    chunks = chunk(text, chunk_size=600, overlap=100)
    assert len(chunks) > 1
    # A chunk can exceed chunk_size only by the overlap it carries over
    assert all(len(c) <= 600 + 100 + 1 for c in chunks)


def test_overlap_carries_text_forward():
    text = " ".join(f"Sentence {i} is here." for i in range(100))
    chunks = chunk(text, chunk_size=200, overlap=50)
    assert chunks[0][-50:].strip() in chunks[1]
