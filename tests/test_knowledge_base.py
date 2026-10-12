"""Chunking and vector-store tests. Uses a tiny deterministic embedding function
so the tests run offline and don't download the ONNX model."""

import hashlib

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from knowledge_base import KnowledgeBase, chunk_text


class BagOfWordsEmbedding(EmbeddingFunction):
    """Hashes words into a 64-dim vector — enough to make similar text score higher."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            v = [0.0] * 64
            for word in text.lower().split():
                word = word.strip(".,!?")
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1.0
            vectors.append(v if any(v) else [1e-6] * 64)
        return vectors

    @staticmethod
    def name() -> str:
        return "bag-of-words-test"

    def get_config(self):
        return {}


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(persist_dir=str(tmp_path), embedding_fn=BagOfWordsEmbedding())


class TestChunking:
    def test_empty(self):
        assert chunk_text("   \n ") == []

    def test_short_text_single_chunk(self):
        assert chunk_text("One sentence. Two sentences.") == ["One sentence. Two sentences."]

    def test_chunks_respect_size(self):
        text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(200))
        chunks = chunk_text(text, chunk_size=300, overlap=60)
        assert len(chunks) > 5
        assert all(len(c) <= 300 for c in chunks)

    def test_overlap_starts_on_word_boundary(self):
        text = " ".join(f"Alpha{i} beta{i} gamma{i} delta{i}." for i in range(100))
        words = set(text.replace(".", "").split())
        for chunk in chunk_text(text, chunk_size=200, overlap=50)[1:]:
            first = chunk.split()[0].rstrip(".")
            assert first in words, f"chunk starts mid-word: {first!r}"

    def test_long_sentence_is_split(self):
        text = "word " * 500  # no punctuation at all
        chunks = chunk_text(text, chunk_size=100, overlap=0)
        assert len(chunks) >= 20
        assert all(len(c) <= 100 for c in chunks)


class TestKnowledgeBase:
    def test_add_and_search(self, kb):
        kb.add_document("a.md", "Kubernetes schedules pods onto nodes. " * 5, {"filename": "a.md"})
        kb.add_document("b.md", "Bananas are a yellow fruit rich in potassium. " * 5, {"filename": "b.md"})
        hits = kb.search("which fruit is yellow", n_results=1)
        assert hits[0]["filename"] == "b.md"
        assert {"text", "filename", "doc_id", "chunk_index", "score"} <= hits[0].keys()

    def test_reupload_replaces_chunks(self, kb):
        kb.add_document("a.md", "First version. " * 100, {"filename": "a.md"})
        kb.add_document("a.md", "Second version.", {"filename": "a.md"})
        assert kb.get_stats() == {"chunk_count": 1, "doc_count": 1}

    def test_filename_filter(self, kb):
        kb.add_document("a.md", "Shared topic text about pods.", {"filename": "a.md"})
        kb.add_document("b.md", "Shared topic text about pods.", {"filename": "b.md"})
        hits = kb.search("pods", filename="a.md")
        assert hits and all(h["filename"] == "a.md" for h in hits)

    def test_search_empty_kb(self, kb):
        assert kb.search("anything") == []

    def test_n_results_capped_at_collection_size(self, kb):
        kb.add_document("a.md", "Only one short chunk here.", {"filename": "a.md"})
        assert len(kb.search("chunk", n_results=10)) == 1

    def test_list_documents_and_clear(self, kb):
        kb.add_document("a.md", "Some text.", {"filename": "a.md"})
        kb.add_document("b.md", "Other text.", {"filename": "b.md"})
        assert kb.list_documents() == {"a.md": 1, "b.md": 1}
        kb.clear()
        assert kb.get_stats() == {"chunk_count": 0, "doc_count": 0}
