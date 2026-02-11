
# ======================== FILE: orchestrator.py ========================
"""
MULTI-AGENT ORCHESTRATOR
Routes queries to the best agent OR runs multiple agents in parallel.
"""
from llm import llm_json
from agents.naive_rag import NaiveRAGAgent
from agents.sentence_window_rag import SentenceWindowRAGAgent
from agents.parent_child_rag import ParentChildRAGAgent
from agents.agentic_rag import AgenticRAGAgent
from agents.multimodal_rag import MultimodalRAGAgent
from agents.table_rag import TableRAGAgent
from agents.graph_rag import GraphRAGAgent
from agents.hybrid_rag import HybridRAGAgent
from agents.hyde_rag import HyDERAGAgent
from agents.corrective_rag import CorrectiveRAGAgent
import asyncio

class Orchestrator:
    def __init__(self):
        self.agents = {
            "naive_rag": NaiveRAGAgent(),
            "sentence_window_rag": SentenceWindowRAGAgent(),
            "parent_child_rag": ParentChildRAGAgent(),
            "agentic_rag": AgenticRAGAgent(),
            "multimodal_rag": MultimodalRAGAgent(),
            "table_rag": TableRAGAgent(),
            "graph_rag": GraphRAGAgent(),
            "hybrid_rag": HybridRAGAgent(),
            "hyde_rag": HyDERAGAgent(),
            "corrective_rag": CorrectiveRAGAgent(),
        }

    def list_agents(self) -> list[dict]:
        return [{"name": a.name, "description": a.description} for a in self.agents.values()]

    async def route(self, query: str) -> str:
        """LLM-based routing to the best agent."""
        agent_descriptions = "\n".join(
            [f"- {a.name}: {a.description}" for a in self.agents.values()]
        )
        result = llm_json(
            f"""Route this query to the best RAG agent. Available agents:
{agent_descriptions}

Guidelines:
- Use table_rag for data/statistics/CSV questions
- Use multimodal_rag for image/visual questions
- Use graph_rag for relationship/connection questions
- Use agentic_rag for complex multi-part questions needing high accuracy
- Use hybrid_rag for keyword-specific technical searches
- Use hyde_rag for vague or abstract questions
- Use corrective_rag when the question might not be answerable from uploaded docs
- Use sentence_window_rag or parent_child_rag for detailed document questions
- Use naive_rag for straightforward factual lookups

Return: {{"agent": "<name>", "reason": "..."}}""",
            query
        )
        return result.get("agent", "naive_rag")

    async def query(self, question: str, agent_name: str = None, top_k: int = 5) -> dict:
        if agent_name is None:
            agent_name = await self.route(question)

        agent = self.agents.get(agent_name)
        if not agent:
            return {"error": f"Unknown agent: {agent_name}", "available": list(self.agents.keys())}

        result = await agent.run(question, top_k)
        result["routed_to"] = agent_name
        return result

    async def compare(self, question: str, agent_names: list[str], top_k: int = 5) -> dict:
        """Run the same query on multiple agents in parallel."""
        tasks = [self.agents[name].run(question, top_k) for name in agent_names if name in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        comparison = {}
        for name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                comparison[name] = {"error": str(result)}
            else:
                comparison[name] = result

        return {"question": question, "comparisons": comparison}

