from abc import ABC, abstractmethod

class BaseRAGAgent(ABC):
    name: str = "base"
    description: str = ""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> list:
        pass

    @abstractmethod
    async def generate(self, query: str, context: list) -> str:
        pass

    async def run(self, query: str, top_k: int = 5) -> dict:
        docs = await self.retrieve(query, top_k)
        answer = await self.generate(query, docs)
        return {
            "agent": self.name,
            "agent_description": self.description,
            "answer": answer,
            "sources": [
                {"content": d.page_content[:300], "metadata": {k: v for k, v in d.metadata.items()
                 if k not in ("parent_content", "window_text", "image_b64_preview")}}
                for d in docs
            ],
            "num_sources": len(docs),
        }
