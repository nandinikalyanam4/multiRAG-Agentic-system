# ======================== FILE: agents/parent_child_rag.py ========================
"""
AGENT 3: PARENT-CHILD RAG (aka Small-to-Big)
----------------------------------------------
WHAT IT DOES: Embeds small child chunks, retrieves parent (larger) chunks.
WHY LEARN IT: Small chunks = better embedding match. Large parents = complete context.
KEY INSIGHT: Decouple what you SEARCH from what you SEND to the LLM.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from langchain.schema import Document
from llm import llm_call

class ParentChildRAGAgent(BaseRAGAgent):
    name = "parent_child_rag"
    description = "Searches small child chunks but returns their larger parent documents."

    CHILD_COLLECTION = "parent_child_rag_children"
    PARENT_COLLECTION = "parent_child_rag_parents"

    async def retrieve(self, query, top_k=5):
        # Step 1: Search children (small, precise)
        children = store_manager.search(self.CHILD_COLLECTION, query, k=top_k * 2)

        # Step 2: Map back to unique parents (large, complete)
        seen_parents = {}
        for child in children:
            pid = child.metadata.get("parent_id")
            if pid and pid not in seen_parents:
                parent_content = child.metadata.get("parent_content", child.page_content)
                seen_parents[pid] = Document(
                    page_content=parent_content,
                    metadata={**child.metadata, "retrieval_method": "parent_from_child"},
                )

        return list(seen_parents.values())[:top_k]

    async def generate(self, query, context):
        ctx = "\n\n---\n\n".join([d.page_content for d in context])
        return llm_call(
            "Answer using the full document sections provided as context.",
            f"Context:\n{ctx}\n\nQuestion: {query}"
        )
