# ======================== FILE: agents/graph_rag.py ========================
"""
AGENT 7: GRAPH RAG
--------------------
WHAT IT DOES: Extracts entities/relations into a knowledge graph, traverses for context.
WHY LEARN IT: Captures relationships that flat text chunks miss entirely.
KEY INSIGHT: "Who does X work with?" needs graph traversal, not vector similarity.
"""
from agents.base import BaseRAGAgent
from graph_store import KnowledgeGraph
from vectorstore import store_manager
from llm import llm_call, llm_json

class GraphRAGAgent(BaseRAGAgent):
    name = "graph_rag"
    description = "Uses knowledge graph to answer questions about entities and relationships."

    COLLECTION = "naive_rag"  # also uses text store for fallback

    def __init__(self):
        self.kg = KnowledgeGraph()

    async def retrieve(self, query, top_k=5):
        # Step 1: Extract entities from the query
        result = llm_json(
            "Extract key entities from this question. Return: {\"entities\": [\"entity1\", \"entity2\"]}",
            query
        )
        entities = result.get("entities", [])

        # Step 2: Get subgraph around those entities
        subgraph = self.kg.query_subgraph(entities, depth=2)

        # Step 3: Also get text chunks for hybrid approach
        text_docs = store_manager.search(self.COLLECTION, query, k=top_k)

        return text_docs, subgraph

    async def generate(self, query, context):
        # context is (text_docs, subgraph) tuple from retrieve
        text_docs, subgraph = context
        text_ctx = "\n\n".join([d.page_content for d in text_docs])

        graph_ctx = "Knowledge Graph:\n"
        for node in subgraph.get("nodes", []):
            graph_ctx += f"  Entity: {node['name']} (type: {node.get('type', '?')}): {node.get('description', '')}\n"
        for edge in subgraph.get("edges", []):
            graph_ctx += f"  Relation: {edge['source']} --[{edge.get('relation', '')}]--> {edge['target']}\n"

        return llm_call(
            "Answer using both the text context and the knowledge graph. "
            "The graph shows entities and their relationships. Prefer graph data for relationship questions.",
            f"Text Context:\n{text_ctx}\n\n{graph_ctx}\n\nQuestion: {query}"
        )

    async def run(self, query: str, top_k: int = 5) -> dict:
        text_docs, subgraph = await self.retrieve(query, top_k)
        answer = await self.generate(query, (text_docs, subgraph))
        return {
            "agent": self.name,
            "agent_description": self.description,
            "answer": answer,
            "sources": [{"content": d.page_content[:300], "metadata": d.metadata} for d in text_docs],
            "graph_context": subgraph,
            "num_sources": len(text_docs),
        }

