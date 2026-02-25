## Model Adaptation & Training Loop

This project is designed to showcase not only retrieval/orchestration, but also how you would **adapt models over time** using the interaction data it generates.

The loop has four stages:

1. **Log interactions + rewards** (already implemented)
2. **Offline evaluation & A/B testing** (implemented in `evals.py`)
3. **Supervised fine-tuning with LoRA + distillation** (conceptual + scaffolding)
4. **Quantized deployment + online RL bandit routing** (conceptual + configuration)

---

### 1. Data: Interaction & Reward Logs

- All queries and feedback are stored by `reward_store.py` in `data/reward_log.jsonl`.
- Each record contains:
  - `interaction_id`
  - `question`
  - `agent`
  - `category`
  - `top_k`
  - `num_sources` (if available)
  - For feedback entries: `reward` in \[0.0, 1.0\]
- This serves as the **training and evaluation dataset** for:
  - Offline evals (see `evals.py`)
  - Supervised fine-tuning / distillation

---

### 2. Runnable training pipeline (`training/`)

From the project root, install optional deps and run:

```bash
pip install -r requirements-training.txt
python training/prepare_sft_data.py
python training/train_lora.py --steps 15
python training/run_quantized.py --model distilgpt2
```

- **prepare_sft_data.py** — Builds `training/data/sft_train.jsonl` from reward log or demo examples.
- **distill_teacher.py** — (Optional) Calls OpenAI to generate teacher answers; writes `teacher_sft.jsonl`.
- **train_lora.py** — Trains LoRA adapters (default: distilgpt2), saves to `training/adapters/`.
- **run_quantized.py** — Loads model (and optional adapters), runs one inference with timing.

The **lora_rag** agent uses `training/adapters` when `LORA_ADAPTERS_PATH` is set and falls back to OpenAI otherwise. See `training/README.md` for full steps.

---

### 3. Offline Evaluation & A/B Testing (`evals.py`)

- `evals.py` is a **generic evaluation harness**:
  - Input: a set of questions (`data/eval/queries.jsonl` if present, otherwise a small built-in demo set).
  - For each question:
    - Runs a list of agents (e.g. `naive_rag,agentic_rag,hybrid_rag`) through `RLOrchestrator.compare`.
    - Collects each agent’s answer.
  - Two modes:
    - `--judge none`: writes raw answers only.
    - `--judge llm`: uses **LLM-as-a-judge** to score each answer 0–1 and pick a winner per question.
  - Aggregates per-agent statistics:
    - Average score
    - Number of scored samples
    - Win counts in pairwise comparison

Usage:

```bash
python evals.py \
  --agents naive_rag,agentic_rag,hybrid_rag \
  --top-k 5 \
  --max-questions 10 \
  --judge none
```

or, using LLM-as-judge (requires `OPENAI_API_KEY`):

```bash
python evals.py \
  --agents naive_rag,agentic_rag,hybrid_rag \
  --top-k 5 \
  --max-questions 10 \
  --judge llm
```

Results are written to `data/eval/results.json` and printed as a summary.

This provides **offline A/B testing** without changing the production serving endpoint.

---

### 4. Supervised Fine-Tuning with LoRA + Distillation (conceptual)

Pipeline (implemented in `training/`; see section 2):

1. **Build a training dataset** from logs:
   - Source: `reward_log.jsonl`.
   - For each interaction with positive feedback (`reward` close to 1.0):
     - Input text:
       - Question.
       - Retrieved context that the agent saw (you can re-run retrieval or log it at serving time).
     - Target text:
       - The agent’s best-rated answer, or
       - A **teacher answer** produced offline by a larger model (distillation).
   - Save as JSONL suitable for `datasets.load_dataset("json", ...)`.

2. **Teacher–Student Distillation**:
   - Use a powerful model (e.g. a larger GPT variant) offline to re-answer historical questions with the same context.
   - Train a smaller open-weight model (e.g. LLaMA/Mistral derivative) to mimic the teacher’s answers.
   - This is standard **sequence-to-sequence distillation**.

3. **LoRA Fine-Tuning**:
   - Apply **LoRA adapters** on top of the base student model:
     - Configure low-rank adapters on attention and/or MLP layers.
     - Train only the adapters on your constructed dataset (cheap and parameter-efficient).
   - Save only the LoRA weights as an artifact.

4. **Integration into this repo**:
   - Load the LoRA-adapted student model in a separate service or in a new agent, e.g. `LoraRAGAgent`:
     - Retrieval logic stays the same (reuse `vectorstore.py` and `bm25_store.py`).
     - Only the **generator** changes from OpenAI to your fine-tuned student model.
   - Register the new agent in `RLOrchestrator.agents`.
   - Add appropriate priors into `PRIOR_REWARDS` in `bandit_router.py` if you want to bias experiments.

This demonstrates a realistic path from **online data → offline SFT/LoRA → new agent**.

---

### 5. Quantization & Deployment

For deployment-focused optimization, you would:

1. Take the **distilled + LoRA-adapted student model**.
2. Load it in 8-bit or 4-bit using libraries like `bitsandbytes` (or model-specific quantization tooling).
3. Benchmark:
   - Latency per token / per request.
   - Memory footprint.
   - Quality via `evals.py --judge llm`.
4. Wrap the quantized model behind a simple generation API and plug it in as:
   - Either a dedicated agent (`QuantizedRAGAgent`), or
   - A drop-in replacement LLM backend in `llm.py` for local runs.

The **multi-armed bandit router** will then naturally treat this quantized agent as another arm:

- If its quality is close to the unquantized version, it will get comparable rewards and be chosen more often.
- If the quality is worse, the bandit will gradually reduce its traffic share.

---

### 6. Online RL Bandit Integration

All of the above feeds back into the bandit:

- As you deploy new agents (LoRA / distilled / quantized):
  - They are added to the candidate set in `bandit_router.AGENTS`.
  - Their priors are configured in `PRIOR_REWARDS` based on expectations.
- As users give feedback via `/feedback`:
  - `BanditRouter.update` adjusts pulls and reward totals.
  - `/rl/stats` and `/rl/leaderboard` show how each variant performs.

This closes the loop:

1. Deploy new model variants as agents.
2. Collect online rewards per question category.
3. Use offline eval (`evals.py`) plus self-analysis (`self_improver.py`) to understand behavior.
4. Iterate on training (LoRA/SFT/distillation) and deployment (quantization + routing).

