"""
LoRA RAG: same retrieval as naive RAG, generation via local LoRA adapters when available.
Falls back to OpenAI if training/adapters not present or training deps not installed.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from llm import llm_call


class LoraRAGAgent(BaseRAGAgent):
    name = "lora_rag"
    description = "RAG with optional local LoRA-generated answers; falls back to API if no adapters."

    COLLECTION = "naive_rag"

    async def retrieve(self, query, top_k=5):
        return store_manager.search(self.COLLECTION, query, k=top_k)

    async def generate(self, query, context):
        ctx = "\n\n---\n\n".join([d.page_content for d in context])
        prompt = f"Answer based on the context provided. If unsure, say so.\nContext:\n{ctx}\n\nQuestion: {query}"
        try:
            from training.inference import generate as local_generate
            out = local_generate(prompt)
            if out:
                return out
        except Exception:
            pass
        return llm_call("Answer based on the context provided. If unsure, say so.", prompt)
