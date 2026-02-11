"""
AGENT 1: NAIVE RAG (Baseline)
-------------------------------
WHAT IT DOES: Chunk → Embed → Retrieve top-k → Generate
WHY LEARN IT: This is the foundation. Every other RAG is an improvement over this.
WEAKNESS: Fixed chunk boundaries lose context. No quality check on retrieval.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from llm import llm_call

class NaiveRAGAgent(BaseRAGAgent):
    name = "naive_rag"
    description = "Basic chunk-retrieve-generate. The baseline RAG approach."

    COLLECTION = "naive_rag"

    async def retrieve(self, query, top_k=5):
        return store_manager.search(self.COLLECTION, query, k=top_k)

    async def generate(self, query, context):
        ctx = "\n\n---\n\n".join([d.page_content for d in context])
        return llm_call(
            "Answer based on the context provided. If unsure, say so.",
            f"Context:\n{ctx}\n\nQuestion: {query}"
        )

