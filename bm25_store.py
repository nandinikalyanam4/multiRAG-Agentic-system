

# ======================== FILE: bm25_store.py ========================
"""
Sparse retrieval (BM25) for Hybrid RAG.
LEARNING: Dense embeddings miss exact keyword matches. BM25 catches them.
"""
from rank_bm25 import BM25Okapi
from langchain.schema import Document
import pickle, os
from config import settings

class BM25Store:
    def __init__(self, name: str):
        self.name = name
        self.path = settings.base_dir / f"bm25_{name}.pkl"
        self.documents: list[Document] = []
        self.bm25: BM25Okapi | None = None
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["docs"]
                self._rebuild_index()

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump({"docs": self.documents}, f)

    def _rebuild_index(self):
        if self.documents:
            tokenized = [d.page_content.lower().split() for d in self.documents]
            self.bm25 = BM25Okapi(tokenized)

    def add_documents(self, docs: list[Document]):
        self.documents.extend(docs)
        self._rebuild_index()
        self._save()

    def search(self, query: str, k: int = 5) -> list[Document]:
        if not self.bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.documents[i] for i in top_indices if scores[i] > 0]

