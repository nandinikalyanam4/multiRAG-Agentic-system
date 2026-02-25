import asyncio
from bandit_router import BanditRouter
from question_classifier import classify_question
from reward_store import log_interaction
from self_improver import SelfImprover

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
from agents.lora_rag import LoraRAGAgent


class RLOrchestrator:
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
            "lora_rag": LoraRAGAgent(),
        }

        self.bandit = BanditRouter()
        self.improver = SelfImprover(self.bandit)

    def list_agents(self) -> list[dict]:
        return [{"name": a.name, "description": a.description} for a in self.agents.values()]

    async def query(self, question: str, agent_name: str = None, top_k: int = 5) -> dict:
        # Classify question (free, no LLM call)
        category = classify_question(question)

        if agent_name:
            # User forced a specific agent
            selected = agent_name
            bandit_info = {"mode": "manual_override", "category": category}
        else:
            # Bandit selects agent
            selected, bandit_info = self.bandit.select_agent(category)

        agent = self.agents.get(selected)
        if not agent:
            return {"error": f"Unknown agent: {selected}", "available": list(self.agents.keys())}

        result = await agent.run(question, top_k)

        # Generate interaction ID for feedback linkage
        import uuid
        interaction_id = str(uuid.uuid4())[:8]

        # Enrich result with RL metadata
        result["interaction_id"] = interaction_id
        result["category"] = category
        result["routing"] = bandit_info
        result["routed_to"] = selected
        result["feedback_hint"] = f"Rate this with POST /feedback: interaction_id={interaction_id}, reward=0.0-1.0"

        # Log interaction (without reward — that comes from feedback)
        log_interaction({
            "interaction_id": interaction_id,
            "question": question,
            "agent": selected,
            "category": category,
            "top_k": top_k,
            "num_sources": result.get("num_sources", 0),
        })

        return result

    async def compare(self, question: str, agent_names: list[str], top_k: int = 5) -> dict:
        category = classify_question(question)
        tasks = [self.agents[n].run(question, top_k) for n in agent_names if n in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        comparison = {}
        for name, result in zip(agent_names, results):
            if isinstance(result, Exception):
                comparison[name] = {"error": str(result)}
            else:
                comparison[name] = result

        return {"question": question, "category": category, "comparisons": comparison}

    def submit_feedback(self, interaction_id: str, reward: float) -> dict:
        """Process user feedback and update bandit."""
        from reward_store import load_all_interactions

        # Find the original interaction
        interactions = load_all_interactions()
        original = None
        for i in interactions:
            if i.get("interaction_id") == interaction_id:
                original = i
                break

        if not original:
            return {"error": f"Interaction {interaction_id} not found"}

        # Update bandit
        self.bandit.update(
            category=original["category"],
            agent=original["agent"],
            reward=reward,
        )

        # Log the feedback
        log_interaction({
            "interaction_id": interaction_id,
            "type": "feedback",
            "agent": original["agent"],
            "category": original["category"],
            "question": original.get("question", ""),
            "reward": reward,
        })

        return {
            "status": "feedback_recorded",
            "interaction_id": interaction_id,
            "agent": original["agent"],
            "category": original["category"],
            "reward": reward,
            "updated_agent_stats": self.bandit.state.get(
                original["category"], {}
            ).get(original["agent"], {}),
        }

