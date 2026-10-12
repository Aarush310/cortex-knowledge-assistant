"""
Knowledge Base — manages the vector store using ChromaDB.

Embeddings are computed locally with ChromaDB's ONNX build of all-MiniLM-L6-v2
(runs on CPU via onnxruntime, no PyTorch and no API key). Only the passages the
agent retrieves are ever sent to the LLM.
"""

import os
import re
from typing import Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_PERSIST_DIR = os.getenv("CORTEX_DATA_DIR", "./chroma_db")
DEFAULT_COLLECTION = "personal_kb"


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks of at most ``chunk_size`` characters.

    Packs whole sentences into each chunk. The overlap carried into the next
    chunk is trimmed to a word boundary so chunks never start mid-word, and a
    single sentence longer than ``chunk_size`` is hard-split on word boundaries.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences: List[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence) <= chunk_size:
            sentences.append(sentence)
            continue
        # Very long "sentence" (tables, code, missing punctuation): split by words.
        piece = ""
        for word in sentence.split(" "):
            if piece and len(piece) + 1 + len(word) > chunk_size:
                sentences.append(piece)
                piece = word
            else:
                piece = f"{piece} {word}" if piece else word
        if piece:
            sentences.append(piece)

    chunks: List[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= chunk_size:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            tail = _overlap_tail(current, overlap)
            current = f"{tail} {sentence}" if tail and len(tail) + 1 + len(sentence) <= chunk_size else sentence
    if current:
        chunks.append(current)
    return chunks


def _overlap_tail(chunk: str, overlap: int) -> str:
    """Last ``overlap`` characters of a chunk, starting at a word boundary."""
    if overlap <= 0 or len(chunk) <= overlap:
        return ""
    tail = chunk[-overlap:]
    space = tail.find(" ")
    return tail[space + 1:] if space != -1 else ""


class KnowledgeBase:
    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_fn=None,
    ):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name
        # Local embedding model — ~80MB, runs on CPU. Injectable for tests.
        self.embedding_fn = embedding_fn or embedding_functions.ONNXMiniLM_L6_V2()
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_document(self, doc_id: str, text: str, metadata: Optional[Dict] = None) -> int:
        """Chunk a document and add all chunks to the vector store. Returns the chunk count."""
        chunks = chunk_text(text)
        if not chunks:
            return 0

        # Re-uploading a file replaces its chunks instead of duplicating them.
        self.collection.delete(where={"doc_id": doc_id})

        ids = [f"{doc_id}::chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {**(metadata or {}), "doc_id": doc_id, "chunk_index": i}
            for i in range(len(chunks))
        ]
        self.collection.add(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def search(self, query: str, n_results: int = 5, filename: Optional[str] = None) -> List[Dict]:
        """Retrieve the most relevant chunks for a query, optionally within one document."""
        total = self.collection.count()
        if total == 0 or not query.strip():
            return []

        kwargs = {"query_texts": [query], "n_results": max(1, min(n_results, total))}
        if filename:
            kwargs["where"] = {"filename": filename}
        results = self.collection.query(**kwargs)

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({
                "text": doc,
                "filename": meta.get("filename", "unknown"),
                "doc_id": meta.get("doc_id", ""),
                "chunk_index": meta.get("chunk_index", -1),
                "score": round(1 - dist, 4),  # cosine distance -> similarity
            })
        return hits

    def list_documents(self) -> Dict[str, int]:
        """Map of filename -> number of chunks stored."""
        docs: Dict[str, int] = {}
        records = self.collection.get(include=["metadatas"])
        for meta in records.get("metadatas") or []:
            if meta:
                name = meta.get("filename") or meta.get("doc_id", "unknown")
                docs[name] = docs.get(name, 0) + 1
        return dict(sorted(docs.items()))

    def get_stats(self) -> Dict:
        docs = self.list_documents()
        return {"chunk_count": sum(docs.values()), "doc_count": len(docs)}

    def clear(self):
        """Wipe the entire collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_collection()
