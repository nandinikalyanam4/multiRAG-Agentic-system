

# ======================== FILE: agents/sentence_window_rag.py ========================
"""
AGENT 2: SENTENCE WINDOW RAG
-------------------------------
WHAT IT DOES: Embeds individual sentences, but returns surrounding window on retrieval.
WHY LEARN IT: Better precision (match exact sentence) + better context (return window).
KEY INSIGHT: Embedding small = precise matching. Returning big = better LLM context.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from langchain.schema import Document
from llm import llm_call

class SentenceWindowRAGAgent(BaseRAGAgent):
    name = "sentence_window_rag"
    description = "Embeds sentences, retrieves surrounding window for better context."

    COLLECTION = "sentence_window_rag"

    async def retrieve(self, query, top_k=5):
        # Search matches individual sentences
        results = store_manager.search(self.COLLECTION, query, k=top_k)

        # But we RETURN the surrounding window text
        expanded = []
        seen_windows = set()
        for doc in results:
            window = doc.metadata.get("window_text", doc.page_content)
            if window not in seen_windows:
                seen_windows.add(window)
                expanded.append(Document(
                    page_content=window,
                    metadata=doc.metadata,
                ))
        return expanded

    async def generate(self, query, context):
        ctx = "\n\n---\n\n".join([d.page_content for d in context])
        return llm_call(
            "Answer using the context. Each context block is a window around a relevant sentence.",
            f"Context:\n{ctx}\n\nQuestion: {query}"
        )
