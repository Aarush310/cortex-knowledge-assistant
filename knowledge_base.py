"""
Knowledge Base — manages the vector store using ChromaDB.
Embeddings are computed locally with sentence-transformers (no API key needed).
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import re


class KnowledgeBase:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "personal_kb"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        # Local embedding model — runs on CPU, no API key required.
        # all-MiniLM-L6-v2 is small (~80MB), fast, and good enough for most use cases.
        self.embedding_fn = embedding_functions.ONNXMiniLM_L6_V2()

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks. Tries to split on sentence boundaries.
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []

        # Split into sentences (rough but works for most prose)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= chunk_size:
                current += (" " + sentence) if current else sentence
            else:
                if current:
                    chunks.append(current.strip())
                # Start new chunk with overlap from previous
                if overlap > 0 and current:
                    overlap_text = current[-overlap:]
                    current = overlap_text + " " + sentence
                else:
                    current = sentence
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def add_document(self, doc_id: str, text: str, metadata: Dict = None):
        """Chunk a document and add all chunks to the vector store."""
        chunks = self._chunk_text(text)
        if not chunks:
            return

        # Remove any pre-existing chunks for this doc (so re-uploads don't duplicate)
        try:
            existing = self.collection.get(where={"doc_id": doc_id})
            if existing and existing.get("ids"):
                self.collection.delete(ids=existing["ids"])
        except Exception:
            pass

        ids = [f"{doc_id}::chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {**(metadata or {}), "doc_id": doc_id, "chunk_index": i}
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Retrieve the most relevant chunks for a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        hits = []
        if results and results.get("documents"):
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append({
                    "text": doc,
                    "filename": meta.get("filename", "unknown"),
                    "doc_id": meta.get("doc_id", ""),
                    "score": 1 - dist,  # convert cosine distance to similarity
                })
        return hits

    def list_documents(self) -> List[Dict]:
        """Return each document's file name and chunk count."""
        counts: Dict[str, int] = {}
        try:
            all_meta = self.collection.get(include=["metadatas"])
            for m in all_meta.get("metadatas") or []:
                if m:
                    name = m.get("filename", m.get("doc_id", "unknown"))
                    counts[name] = counts.get(name, 0) + 1
        except Exception:
            pass
        return [{"filename": name, "chunks": n} for name, n in sorted(counts.items())]

    def get_stats(self) -> Dict:
        """Return basic stats about the knowledge base."""
        try:
            count = self.collection.count()
            all_meta = self.collection.get()
            unique_docs = set()
            if all_meta and all_meta.get("metadatas"):
                for m in all_meta["metadatas"]:
                    if m and "doc_id" in m:
                        unique_docs.add(m["doc_id"])
            return {"chunk_count": count, "doc_count": len(unique_docs)}
        except Exception:
            return {"chunk_count": 0, "doc_count": 0}

    def clear(self):
        """Wipe the entire collection."""
        try:
            self.client.delete_collection(self.collection.name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name="personal_kb",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
