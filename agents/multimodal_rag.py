

# ======================== FILE: agents/multimodal_rag.py ========================
"""
AGENT 5: MULTIMODAL RAG
--------------------------
WHAT IT DOES: Handles images + text. Uses vision LLM to describe images, retrieves by text.
WHY LEARN IT: Real-world data is multimodal. Interviews ask about this.
KEY INSIGHT: Convert all modalities to text embeddings, but preserve original modality for generation.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from llm import llm_call

class MultimodalRAGAgent(BaseRAGAgent):
    name = "multimodal_rag"
    description = "Handles images and text together. Images are described by vision LLM for retrieval."

    COLLECTION = "multimodal_rag"

    async def retrieve(self, query, top_k=5):
        return store_manager.search(self.COLLECTION, query, k=top_k)

    async def generate(self, query, context):
        # Separate text and image sources
        text_parts, image_parts = [], []
        for doc in context:
            if doc.metadata.get("modality") == "image":
                image_parts.append(f"[IMAGE: {doc.page_content}]")  # description
            else:
                text_parts.append(doc.page_content)

        ctx = "\n\n---\n\n".join(text_parts + image_parts)
        return llm_call(
            "Answer using the context which may include image descriptions marked with [IMAGE]. "
            "Reference visual elements when relevant.",
            f"Context:\n{ctx}\n\nQuestion: {query}"
        )
