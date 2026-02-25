## RL-Enhanced Multi-Agent RAG System

An end-to-end, production-style Retrieval-Augmented Generation (RAG) system:

- **10+ specialized RAG agents** (naive, sentence-window, parent-child, agentic, multimodal, table, graph, hybrid, HyDE, corrective).
- **RL-based multi-armed bandit router** that learns which agent works best per question type.
- **Feedback loop + self-improvement**: user rewards update the bandit, and an offline analyzer recommends parameter/agent changes.
- **Hybrid retrieval stack**: dense vector search (Chroma) + BM25 + knowledge graph.
- **FastAPI backend** with clean, documented endpoints and Docker deployment.
 - **Offline eval + A/B testing**: `evals.py` runs multi-agent comparisons with optional LLM-as-judge scoring.
 - **Model adaptation pipeline**: documented loop for LoRA/SFT, distillation, quantization, and RL-based routing.
---

### High-Level Architecture

- **Ingestion layer**
  - `PDFProcessor`, `ImageProcessor`, `CSVProcessor` convert raw files into `Document` objects with rich metadata.
  - Multiple chunking strategies: standard chunks, sentence-level windows, parent/child hierarchy, table schema/row groups.
  - Indexes into:
    - Multiple **Chroma** collections (per RAG agent type),
    - **BM25** store for sparse retrieval,
    - **Knowledge graph** built from extracted entities/relations.

- **Agent layer (`agents/*.py`)**
  - Each agent encapsulates a retrieval strategy + prompting style.
  - Examples:
    - **SentenceWindowRAG**: retrieves sentence-level matches and expands with surrounding context.
    - **ParentChildRAG**: retrieves fine-grained child chunks but returns parent-level passages.
    - **GraphRAG**: traverses entities and relationships from the knowledge graph.
    - **MultimodalRAG**: indexes images via vision model descriptions.
    - **CorrectiveRAG**: detects when local retrieval is weak and falls back to external search.

- **RL Orchestrator (`rl_orchestrator.py`)**
  - Wraps all agents and exposes:
    - `query`: route a question to the best agent (or a user-selected agent).
    - `compare`: run multiple agents in parallel on the same question.
    - `submit_feedback`: update the bandit from user rewards.
  - Uses:
    - `BanditRouter` (UCB1 multi-armed bandit) for agent selection.
    - `classify_question` for cheap, non-LLM categorization (factual/analytical/relational/procedural/vague/visual).
    - `reward_store` to log all interactions and feedback.
    - `SelfImprover` to analyze historical data and recommend changes.

- **RL Bandit Router (`bandit_router.py`)**
  - Implements UCB1:
    - State: question **category**.
    - Actions: choose a **RAG agent**.
    - Reward: user feedback \(0.0 – 1.0\).
  - Uses **smart priors** so the system behaves well even before much data:
    - e.g. table queries initially favor `table_rag`, visual queries favor `multimodal_rag`, etc.
  - Persists state to `bandit_state.json` for continuity across restarts.

- **Feedback + Self-Improvement**
  - All interactions and rewards stream into `reward_log.jsonl`.
  - `SelfImprover` runs offline analysis:
    - Finds struggling vs. high-performing agents per category.
    - Learns which `top_k` values correlate with better rewards.
    - Detects hardest question categories and query-length effects.
    - Produces human-readable recommendations and parameter suggestions.

- **API Layer (`main.py`)**
  - `POST /upload` — upload PDFs/CSVs/images; auto-routed to correct processor; indexes into all relevant stores.
  - `POST /query` — ask a question; RL bandit chooses the best agent.
  - `POST /query/{agent}` — force a specific agent.
  - `POST /compare` — run the same question on multiple agents and compare outputs.
  - `POST /feedback` — send reward feedback (0.0–1.0) tied to an `interaction_id`.
  - `GET /rl/stats` — inspect bandit state (per-category performance).
  - `GET /rl/leaderboard` — global agent leaderboard.
  - `POST /rl/optimize` — run self-improvement analysis.
  - `GET /agents`, `GET /collections`, `GET /health` — observability and introspection.

For a deeper dive, see:
- `docs/ARCHITECTURE.md` — system architecture.
- `docs/MODEL_ADAPTATION.md` — LoRA/SFT, distillation, quantization, and eval loop.

---

### Running Locally (Dev)

- **1. Create and activate a virtualenv (optional)**

```bash
python -m venv venv
source venv/bin/activate  # on macOS/Linux
# .\venv\Scripts\activate  # on Windows
```

- **2. Install dependencies**

```bash
pip install -r requirements.txt
```

- **3. Configure environment**

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
```

- **4. Run the API**

```bash
uvicorn main:app --reload --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

---

### Example Workflow

- **Upload documents**

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/path/to/your.pdf"
```

- **Ask a question (bandit auto-routing)**

```bash
curl -X POST "http://localhost:8000/query" \
  -F "question=Summarize the key risks in this document." \
  -F "top_k=5"
```

- **Give feedback**

Take the `interaction_id` from the previous response and:

```bash
curl -X POST "http://localhost:8000/feedback" \
  -F "interaction_id=abcd1234" \
  -F "reward=0.9"
```

- **Inspect RL learning**

```bash
curl "http://localhost:8000/rl/stats"
curl "http://localhost:8000/rl/leaderboard"
```

---


