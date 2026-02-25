## Architecture Deep Dive

This document is written for **staff+ / founding engineers** reviewing the project. It focuses on design choices and trade-offs rather than just API usage.

---

### 1. Problem Framing

Goal: build a **general-purpose, multi-agent RAG system** that:

- Works across **documents, tables, and images**.
- Adapts over time via **reinforcement learning from human feedback (RLHF-style)**.
- Makes it easy to **add new agents** and compare them in production.

Instead of hard-coding “one RAG pipeline”, the system treats each RAG strategy as an **agent**, and uses a **multi-armed bandit** to learn which agent works best for which type of question.

---

### 2. Ingestion & Storage

- **File processors (`processors.py`)**
  - **PDFs**
    - Chunked with `RecursiveCharacterTextSplitter` into:
      - Standard chunks (for generic dense retrieval).
      - Sentence-level chunks with **windowed context** (for high-precision local answers).
      - Parent/child hierarchy (large parents, small children) to separate *retrieval granularity* from *answer granularity*.
    - Full text is also returned for **graph extraction**.
  - **Images**
    - Converted to base64 and passed through a **vision LLM** to generate a rich textual description.
    - Resulting `Document` is stored in multiple collections so text-only agents can still leverage image content.
  - **CSVs / Excel**
    - DataFrame-based ingestion:
      - A **schema chunk** (columns, dtypes, head, describe) that is always relevant.
      - Row-group chunks (25-row windows) to allow local reasoning over subsets.
    - The raw file path is preserved for potential text-to-pandas execution agents.

- **Vector stores**
  - A `store_manager` (see `vectorstore.py`) maintains **separate collections per agent type**, e.g.:
    - `naive_rag`, `sentence_window_rag`, `parent_child_rag_children`, `parent_child_rag_parents`, `multimodal_rag`, `table_rag`, etc.
  - This design makes it easy to:
    - Tune index parameters per agent (e.g. different similarity metrics or chunk sizes).
    - Drop / rebuild specific agents without disrupting others.

- **BM25 store (`bm25_store.py`)**
  - Sparse keyword-based retrieval via `rank-bm25`.
  - Used directly by `hybrid_rag` and as a fallback signal for agents that need stronger keyword grounding.

- **Knowledge graph (`graph_store.py`)**
  - Uses an LLM-based entity/relation extraction pass on `full_text`.
  - Stores entities and edges in a `networkx` graph.
  - Enables:
    - “Why/how are X and Y related?”-style reasoning.
    - Traversal-based retrieval beyond pure lexical similarity.

---

### 3. Agent Layer

Each agent in `agents/*.py` has the same interface:

- `name`: human-readable identifier.
- `description`: natural language summary (used for routing / docs).
- `run(question: str, top_k: int) -> dict`: main execution entrypoint.

Examples of implemented strategies:

- **NaiveRAGAgent**
  - Classic “retrieve top_k chunks + generate answer”.
  - Serves as a **baseline** for measuring value of more complex agents.

- **SentenceWindowRAGAgent**
  - Retrieves sentence-level chunks and auto-expands context via `window_text`.
  - Good for pinpointing **high-precision facts** located in dense text.

- **ParentChildRAGAgent**
  - Retrieves fine-grained child chunks but returns their larger parent context.
  - Decouples *retrieval resolution* from *answer readability*.

- **AgenticRAGAgent**
  - Implements a **reflection loop**:
    - Draft answer → check for gaps/conflicts → refine using additional retrieval.
  - Configured with `settings.max_agentic_retries`.

- **MultimodalRAGAgent**
  - Operates over **image-derived textual embeddings**.
  - Can mix image and text context in a unified prompt.

- **TableRAGAgent**
  - Uses table schema and row-group chunks to answer aggregation and analytics questions.
  - Can be extended with executable Python / pandas evaluation if desired.

- **GraphRAGAgent**
  - Retrieves entities and paths from the knowledge graph.
  - Prompts the LLM to explain relationships using the graph context.

- **HyDERAGAgent**
  - Generates a **hypothetical document** from the question and embeds that for retrieval (**HyDE**).
  - In practice, performs well for vague / underspecified queries.

- **CorrectiveRAGAgent**
  - Detects low-retrieval-confidence scenarios.
  - Can escalate to **web search / external APIs** to avoid hallucinated answers.

The system is intentionally **open for extension**: adding a new agent is primarily:

1. Implementing a small class in `agents/`.
2. Registering it in `RLOrchestrator.agents` and (optionally) in `PRIOR_REWARDS`.

---

### 4. RL-Orchestrated Routing

#### 4.1 Question Classification (`question_classifier.py`)

- Uses **regex-based heuristics** (no LLM calls) to assign each question to:
  - `factual`, `analytical`, `relational`, `procedural`, `vague`, `visual`.
- This keeps routing cheap and deterministic while still being semantically meaningful.

#### 4.2 Bandit Router (`bandit_router.py`)

- Implements **UCB1** for each (category, agent) pair:

\[
\text{score} = \overline{r} + C \sqrt{\frac{\ln N}{n}}
\]

Where:

- \(\overline{r}\) = average reward for this agent in this category.
- \(N\) = total pulls in this category.
- \(n\) = pulls for this agent in this category.
- \(C\) = exploration constant (default \(\sqrt{2}\)).

- Design details:
  - **Smart priors** encoded in `PRIOR_REWARDS`:
    - e.g. `("analytical", "table_rag") → (3, 2.4)` meaning “3 prior pulls, total reward 2.4 (avg 0.8)”.
  - Default prior for others is `(1, 0.5)` to prevent division by zero and encode a neutral expectation.
  - State is persisted to `bandit_state.json`.

#### 4.3 RL Orchestrator (`rl_orchestrator.py`)

- For each `/query`:
  1. Classify question → `category`.
  2. If `agent_name` explicitly provided, use it (and annotate as `manual_override`).
  3. Else, call `BanditRouter.select_agent(category)` to pick the agent.
  4. Run agent’s `run(...)`.
  5. Attach:
     - `interaction_id` (short UUID).
     - `category`, `routing` (full bandit debug info).
     - `feedback_hint` explaining how to call `/feedback`.
  6. Log the interaction to `reward_log.jsonl` via `reward_store.log_interaction`.

- For `/feedback`:
  - Look up the original interaction by `interaction_id`.
  - Call `BanditRouter.update(category, agent, reward)`.
  - Log a new feedback entry for analysis.

This is a **minimal but production-realistic RL loop**: it optimizes agent selection policy directly from user rewards with no manual labeling.

---

### 5. Self-Improvement (`self_improver.py`)

The `SelfImprover` runs offline analytics over the interaction log:

- **Agent performance by category**
  - Detects agents with significantly low/high average reward per category.
  - Produces insight objects like:
    - “`graph_rag` struggles with `analytical` questions (avg reward 0.32, n=5).”
    - “`table_rag` excels at `analytical` questions (avg 0.88, n=7).”

- **Top-k effectiveness**
  - Computes average reward per `top_k` and suggests a `recommended_top_k`.

- **Category difficulty**
  - Finds the hardest category (lowest avg reward) and suggests uploading better domain data.

- **Query-length analysis**
  - Compares short vs long queries to surface behavioral differences.

- **Actionable recommendations**
  - Combines the above with bandit stats to generate human-readable recommendations about:
    - Prioritizing certain agents.
    - Potentially deprecating or reworking underperformers.

The `/rl/optimize` endpoint surfaces this as a JSON report suitable for dashboards or ops tooling.

---

### 6. API & Deployment

- **FastAPI (`main.py`)**
  - Clean, form-based endpoints for file upload and querying.
  - CORS enabled for easy integration with any front-end.

- **Dockerfile**
  - Simple `python:3.11-slim` image, installs `requirements.txt`, sets up `/app/data`.
  - Exposes Gunicorn+Uvicorn worker for production-style serving.

- **Config (`config.py`)**
  - Uses `pydantic-settings` for typed, environment-driven configuration.
  - All secrets (e.g. `OPENAI_API_KEY`) are expected from env / `.env` and **not** hard-coded.

---

### 7. Extensibility & Next Steps

Some natural extensions (if you want to push the project even further):

- **Front-end dashboard**
  - File upload + chat UI that:
    - Visualizes which agent was chosen.
    - Shows bandit scores and agent leaderboard in real time.
    - Provides a slider / thumbs-up/down to submit feedback.

- **Eval harness**
  - A small eval suite that:
    - Runs a set of held-out Q&A pairs through all agents.
    - Scores them with an LLM-based evaluator.
    - Compares offline eval metrics to online RL rewards.

- **Pluggable tools**
  - Expose an interface for agents to call tools (e.g. code execution, SQL) and evaluate the impact on reward.

Even without these, the current system already demonstrates:

- Non-trivial **RL + multi-agent orchestration**.
- Thoughtful **retrieval engineering** (chunking, hybrid search, graph).
- Realistic **observability** and **self-improvement** loops.

