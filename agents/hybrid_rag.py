

# ======================== FILE: agents/hybrid_rag.py ========================
"""
AGENT 8: HYBRID RAG (Dense + Sparse)
---------------------------------------
WHAT IT DOES: Combines vector similarity (dense) with BM25 keyword matching (sparse).
WHY LEARN IT: Dense misses exact keywords. Sparse misses semantics. Together = best of both.
KEY INSIGHT: Reciprocal Rank Fusion (RRF) merges two ranked lists into one.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from bm25_store import BM25Store
from langchain.schema import Document
from llm import llm_call

class HybridRAGAgent(BaseRAGAgent):
    name = "hybrid_rag"
    description = "Combines dense (embedding) + sparse (BM25) retrieval with rank fusion."

    COLLECTION = "naive_rag"

    def __init__(self):
        self.bm25 = BM25Store("hybrid")

    def reciprocal_rank_fusion(self, dense: list, sparse: list, k: int = 60) -> list:
        """RRF: score = sum(1 / (k + rank)) across both lists."""
        scores = {}
        doc_map = {}

        for rank, doc in enumerate(dense):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(sparse):
            key = doc.page_content[:100]
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[k] for k in sorted_keys]

    async def retrieve(self, query, top_k=5):
        dense = store_manager.search(self.COLLECTION, query, k=top_k * 2)
        sparse = self.bm25.search(query, k=top_k * 2)
        fused = self.reciprocal_rank_fusion(dense, sparse)
        return fused[:top_k]

    async def generate(self, query, context):
        ctx = "\n\n---\n\n".join([d.page_content for d in context])
        return llm_call(
            "Answer based on context retrieved via hybrid search (semantic + keyword matching).",
            f"Context:\n{ctx}\n\nQuestion: {query}"
        )
