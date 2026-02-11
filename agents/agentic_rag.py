

# ======================== FILE: agents/agentic_rag.py ========================
"""
AGENT 4: AGENTIC RAG (Self-Correcting)
-----------------------------------------
WHAT IT DOES: Retrieve → Grade relevance → Reformulate if bad → Generate → Check hallucination
WHY LEARN IT: THIS IS THE INTERVIEW STAR. Shows you understand agents, reflection, self-correction.
KEY INSIGHT: Agents don't just retrieve — they evaluate, decide, and retry.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from llm import llm_call, llm_json
from config import settings

class AgenticRAGAgent(BaseRAGAgent):
    name = "agentic_rag"
    description = "Self-correcting RAG with relevance grading, query reformulation, and hallucination checks."

    COLLECTION = "naive_rag"  # reuses the same store, different retrieval logic

    async def retrieve(self, query, top_k=5):
        return store_manager.search(self.COLLECTION, query, k=top_k)

    async def grade_relevance(self, query: str, docs: list) -> tuple[list, list]:
        """LLM grades each doc. Returns (relevant, irrelevant)."""
        relevant, irrelevant = [], []
        for doc in docs:
            result = llm_json(
                "Grade if this document is relevant to the question. "
                "Return: {\"relevant\": true/false, \"reason\": \"brief reason\"}",
                f"Question: {query}\nDocument: {doc.page_content[:600]}"
            )
            if result.get("relevant", False):
                doc.metadata["grade_reason"] = result.get("reason", "")
                relevant.append(doc)
            else:
                irrelevant.append(doc)
        return relevant, irrelevant

    async def reformulate(self, query: str, attempt: int) -> str:
        return llm_call(
            "Rewrite this search query to find better, more relevant results. "
            "Try a different angle or phrasing. Return ONLY the new query.",
            f"Original query (attempt {attempt}): {query}"
        ).strip()

    async def check_hallucination(self, answer: str, docs: list) -> dict:
        ctx = "\n".join([d.page_content[:400] for d in docs])
        return llm_json(
            "Check if the answer is fully supported by the context. "
            "Return: {\"grounded\": true/false, \"unsupported_claims\": [\"...\"]}",
            f"Context:\n{ctx}\n\nAnswer: {answer}"
        )

    async def generate(self, query, context):
        ctx = "\n\n---\n\n".join([d.page_content for d in context])
        return llm_call(
            "Answer ONLY based on the context. Cite specific parts that support your answer. "
            "If context is insufficient, clearly state what's missing.",
            f"Context:\n{ctx}\n\nQuestion: {query}"
        )

    async def run(self, query: str, top_k: int = 5) -> dict:
        trace = []  # Full reasoning trace for learning/debugging
        current_query = query

        for attempt in range(settings.max_agentic_retries):
            # Retrieve
            docs = await self.retrieve(current_query, top_k)
            trace.append({"step": "retrieve", "attempt": attempt + 1,
                          "query": current_query, "docs_found": len(docs)})

            # Grade
            relevant, irrelevant = await self.grade_relevance(query, docs)
            trace.append({"step": "grade", "relevant": len(relevant),
                          "irrelevant": len(irrelevant)})

            if len(relevant) >= 2 or attempt == settings.max_agentic_retries - 1:
                break

            # Reformulate
            current_query = await self.reformulate(current_query, attempt + 1)
            trace.append({"step": "reformulate", "new_query": current_query})

        # Generate
        final_docs = relevant if relevant else docs
        answer = await self.generate(query, final_docs)
        trace.append({"step": "generate"})

        # Hallucination check
        hall_check = await self.check_hallucination(answer, final_docs)
        trace.append({"step": "hallucination_check", **hall_check})

        if not hall_check.get("grounded", True):
            answer = await self.generate(query, final_docs)
            answer += "\n\n[Note: Answer was regenerated after hallucination detection]"
            trace.append({"step": "regenerate"})

        return {
            "agent": self.name,
            "agent_description": self.description,
            "answer": answer,
            "sources": [{"content": d.page_content[:300], "metadata": d.metadata} for d in final_docs],
            "num_sources": len(final_docs),
            "grounded": hall_check.get("grounded", True),
            "retrieval_attempts": attempt + 1,
            "trace": trace,
        }
