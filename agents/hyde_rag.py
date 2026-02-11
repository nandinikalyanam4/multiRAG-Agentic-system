

# ======================== FILE: agents/hyde_rag.py ========================
"""
AGENT 9: HyDE RAG (Hypothetical Document Embeddings)
------------------------------------------------------
WHAT IT DOES: First generates a HYPOTHETICAL answer, then uses it to search.
WHY LEARN IT: Queries are short. Hypothetical docs are long and semantically rich.
KEY INSIGHT: Embed the answer you EXPECT to find, not the question itself.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from llm import llm_call

class HyDERAGAgent(BaseRAGAgent):
    name = "hyde_rag"
    description = "Generates hypothetical answer first, uses it for retrieval. Bridges query-document gap."

    COLLECTION = "naive_rag"

    async def generate_hypothesis(self, query: str) -> str:
        return llm_call(
            "Write a detailed, factual paragraph that would perfectly answer this question. "
            "This will be used for document retrieval, so make it specific and information-dense. "
            "Don't hedge — write as if you know the answer.",
            query
        )

    async def retrieve(self, query, top_k=5):
        # Step 1: Generate hypothetical document
        hypothesis = await self.generate_hypothesis(query)

        # Step 2: Search using the hypothesis (not the original query!)
        hyde_results = store_manager.search(self.COLLECTION, hypothesis, k=top_k)

        # Step 3: Also search with original query and merge
        original_results = store_manager.search(self.COLLECTION, query, k=top_k)

        # Deduplicate
        seen = set()
        merged = []
        for doc in hyde_results + original_results:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged[:top_k]

    async def generate(self, query, context):
        ctx = "\n\n---\n\n".join([d.page_content for d in context])
        return llm_call(
            "Answer using the provided context. Ignore any prior assumptions.",
            f"Context:\n{ctx}\n\nQuestion: {query}"
        )

