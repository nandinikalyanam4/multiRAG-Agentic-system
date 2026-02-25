# Runnable training pipeline

Install optional deps (from project root):

```bash
pip install -r requirements-training.txt
```

Then run in order:

1. **Prepare SFT data** (demo data if no reward log):
   ```bash
   python training/prepare_sft_data.py
   ```
   Writes `training/data/sft_train.jsonl`.

2. **(Optional) Distill teacher** — use OpenAI to generate target answers:
   ```bash
   export OPENAI_API_KEY=sk-...
   python training/distill_teacher.py
   ```
   Writes `training/data/teacher_sft.jsonl`. Then use `--data training/data/teacher_sft.jsonl` in step 3.

3. **Train LoRA adapters**:
   ```bash
   python training/train_lora.py --data training/data/sft_train.jsonl --steps 15
   ```
   Saves to `training/adapters/`. Uses `distilgpt2` by default (runs on CPU, ~1–2 min).

4. **Run quantized inference** (optional; no adapters):
   ```bash
   python training/run_quantized.py --model distilgpt2 --prompt "What is RAG? Answer in one sentence."
   ```
   With adapters:
   ```bash
   python training/run_quantized.py --model distilgpt2 --adapters training/adapters
   ```

5. **Use LoRA in the API** — set adapters path and start the API:
   ```bash
   export LORA_ADAPTERS_PATH=training/adapters
   uvicorn main:app --reload --port 8000
   ```
   Then query with agent `lora_rag` or let the bandit route to it.
