

# ======================== FILE: agents/corrective_rag.py ========================
"""
AGENT 10: CORRECTIVE RAG (CRAG)
---------------------------------
WHAT IT DOES: If local retrieval is poor, falls back to web search.
WHY LEARN IT: Production RAG must handle knowledge gaps gracefully.
KEY INSIGHT: Grade retrieval quality. If bad → don't hallucinate, go to the web.
"""
from agents.base import BaseRAGAgent
from vectorstore import store_manager
from llm import llm_call, llm_json
import httpx

class CorrectiveRAGAgent(BaseRAGAgent):
    name = "corrective_rag"
    description = "Falls back to web search when local retrieval is insufficient."

    COLLECTION = "naive_rag"

    async def retrieve(self, query, top_k=5):
        return store_manager.search(self.COLLECTION, query, k=top_k)

    async def assess_retrieval(self, query: str, docs: list) -> str:
        """Returns: 'correct', 'ambiguous', or 'incorrect'."""
        if not docs:
            return "incorrect"
        ctx = "\n".join([d.page_content[:300] for d in docs[:3]])
        result = llm_json(
            "Assess if these documents can answer the question. "
            "Return: {\"assessment\": \"correct|ambiguous|incorrect\", \"reason\": \"...\"}",
            f"Question: {query}\n\nDocuments:\n{ctx}"
        )
        return result.get("assessment", "ambiguous")

    async def web_search_fallback(self, query: str) -> list:
        """Simple web search fallback using a free API."""
        # In production, use Tavily, Serper, or Brave Search API
        # This is a placeholder — replace with your preferred search API
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.duckduckgo.com/",
                    params={"q": query, "format": "json", "no_redirect": 1}
                )
                data = response.json()
                from langchain.schema import Document
                results = []
                for item in data.get("RelatedTopics", [])[:3]:
                    if "Text" in item:
                        results.append(Document(
                            page_content=item["Text"],
                            metadata={"source": "web_search", "url": item.get("FirstURL", "")}
                        ))
                return results
        except Exception:
            return []

    async def generate(self, query, context):
        ctx = "\n\n---\n\n".join([d.page_content for d in context])
        sources = set(d.metadata.get("source", "local") for d in context)
        source_note = " (includes web results)" if "web_search" in sources else ""
        return llm_call(
            f"Answer the question using context{source_note}. Clearly indicate if info comes from web search.",
            f"Context:\n{ctx}\n\nQuestion: {query}"
        )

    async def run(self, query: str, top_k: int = 5) -> dict:
        docs = await self.retrieve(query, top_k)
        assessment = await self.assess_retrieval(query, docs)

        web_docs = []
        if assessment in ("incorrect", "ambiguous"):
            web_docs = await self.web_search_fallback(query)

        if assessment == "incorrect":
            final_docs = web_docs if web_docs else docs
        elif assessment == "ambiguous":
            final_docs = docs + web_docs
        else:
            final_docs = docs

        answer = await self.generate(query, final_docs)
        return {
            "agent": self.name,
            "agent_description": self.description,
            "answer": answer,
            "sources": [{"content": d.page_content[:300], "metadata": d.metadata} for d in final_docs],
            "num_sources": len(final_docs),
            "retrieval_assessment": assessment,
            "web_fallback_used": len(web_docs) > 0,
        }
