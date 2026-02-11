# ======================== FILE: vectorstore.py ========================
"""
Unified vector store manager. Each agent gets its own ChromaDB collection.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain.schema import Document
from embeddings import get_embeddings
from config import settings
from typing import Optional
import hashlib

class VectorStoreManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.embeddings = get_embeddings()
        self._stores: dict[str, Chroma] = {}

    def get_store(self, collection_name: str) -> Chroma:
        if collection_name not in self._stores:
            self._stores[collection_name] = Chroma(
                client=self.client,
                collection_name=collection_name,
                embedding_function=self.embeddings,
            )
        return self._stores[collection_name]

    def add_documents(self, collection: str, docs: list[Document]):
        store = self.get_store(collection)
        # Deduplicate by content hash
        ids = [hashlib.md5(d.page_content.encode()).hexdigest() for d in docs]
        store.add_documents(docs, ids=ids)
        return len(docs)

    def search(self, collection: str, query: str, k: int = 5,
               filter_dict: Optional[dict] = None) -> list[Document]:
        store = self.get_store(collection)
        kwargs = {"k": k}
        if filter_dict:
            kwargs["filter"] = filter_dict
        return store.similarity_search(query, **kwargs)

    def search_with_scores(self, collection: str, query: str, k: int = 5) -> list:
        store = self.get_store(collection)
        return store.similarity_search_with_relevance_scores(query, k=k)

    def list_collections(self) -> list[dict]:
        collections = self.client.list_collections()
        return [{"name": c.name, "count": c.count()} for c in collections]

# Global singleton
store_manager = VectorStoreManager()