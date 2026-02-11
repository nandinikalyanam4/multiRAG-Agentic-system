
# ======================== FILE: graph_store.py ========================
"""
Knowledge graph for Graph RAG.
LEARNING: Entities + relationships capture structure that flat chunks miss.
"""
import networkx as nx
import pickle
from config import settings
from llm import llm_json

class KnowledgeGraph:
    def __init__(self):
        self.path = settings.base_dir / "knowledge_graph.pkl"
        self.graph = nx.DiGraph()
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path, "rb") as f:
                self.graph = pickle.load(f)

    def _save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.graph, f)

    def extract_and_add(self, text: str, source: str):
        """Use LLM to extract entities and relationships from text."""
        result = llm_json(
            system="""Extract entities and relationships from this text.
Return JSON: {
  "entities": [{"name": "...", "type": "person|org|concept|location|event", "description": "..."}],
  "relationships": [{"source": "...", "target": "...", "relation": "...", "description": "..."}]
}
Extract ALL meaningful entities and relationships. Be thorough.""",
            user=text[:3000]  # limit to avoid token overflow
        )

        for entity in result.get("entities", []):
            self.graph.add_node(
                entity["name"].lower(),
                type=entity.get("type", "unknown"),
                description=entity.get("description", ""),
                source=source,
            )

        for rel in result.get("relationships", []):
            self.graph.add_edge(
                rel["source"].lower(),
                rel["target"].lower(),
                relation=rel.get("relation", "related_to"),
                description=rel.get("description", ""),
                source=source,
            )
        self._save()

    def query_subgraph(self, entities: list[str], depth: int = 2) -> dict:
        """Get subgraph around entities up to N hops."""
        relevant_nodes = set()
        for entity in entities:
            entity_lower = entity.lower()
            # Fuzzy match: find nodes containing the entity name
            matches = [n for n in self.graph.nodes if entity_lower in n or n in entity_lower]
            for match in matches:
                # BFS up to `depth` hops
                for d in range(depth + 1):
                    neighbors = nx.single_source_shortest_path_length(self.graph, match, cutoff=d)
                    relevant_nodes.update(neighbors.keys())

        subgraph = self.graph.subgraph(relevant_nodes)
        nodes = [{"name": n, **self.graph.nodes[n]} for n in subgraph.nodes]
        edges = [{"source": u, "target": v, **d} for u, v, d in subgraph.edges(data=True)]
        return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}

    def get_stats(self) -> dict:
        return {"nodes": self.graph.number_of_nodes(), "edges": self.graph.number_of_edges()}
