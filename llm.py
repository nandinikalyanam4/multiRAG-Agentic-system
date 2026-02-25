"""
=============================================================================
COMPLETE MULTI-AGENT RAG SYSTEM
=============================================================================
All RAG types in one unified system with multi-agent orchestration.

RAG Agents:
  1. Naive RAG          — basic chunk + retrieve + generate (baseline)
  2. Sentence Window RAG — retrieves small, returns surrounding context
  3. Parent-Child RAG   — retrieves child chunks, returns parent docs
  4. Agentic RAG        — self-correcting with reflection loop
  5. Multimodal RAG     — images + text with vision LLM
  6. Table RAG          — structured data with text-to-pandas
  7. Graph RAG          — entity extraction + knowledge graph traversal
  8. Hybrid RAG         — combines dense + sparse (BM25) retrieval
  9. HyDE RAG           — hypothetical document embeddings
  10. Corrective RAG    — web fallback when local retrieval fails

Endpoints:
  POST /upload          — upload files (auto-routed to correct processors)
  POST /query           — query with auto-agent-selection
  POST /query/{agent}   — query a specific agent
  POST /compare         — run same query on multiple agents, compare
  GET  /agents          — list all available agents
  GET  /collections     — list all vector store collections + stats

Run: uvicorn main:app --reload --port 8000
=============================================================================
"""


# ======================== FILE: llm.py ========================
from openai import OpenAI
from config import settings
import json

client = OpenAI(api_key=settings.openai_api_key)

def llm_call(system: str, user: str, json_mode: bool = False, model: str = None) -> str:
    """Unified LLM call helper."""
    kwargs = {
        "model": model or settings.llm_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["messages"][0]["content"] += " Respond in JSON."
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def llm_json(system: str, user: str) -> dict:
    """LLM call that returns parsed JSON."""
    raw = llm_call(system, user, json_mode=True)
    return json.loads(raw)

def llm_vision(image_b64: str, prompt: str) -> str:
    """Vision LLM call for images."""
    response = client.chat.completions.create(
        model=settings.vision_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]
        }],
        max_tokens=800,
    )
    return response.choices[0].message.content

