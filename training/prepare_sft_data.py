"""
Prepare SFT (supervised fine-tuning) data from reward log or demo set.
Outputs training/data/sft_train.jsonl for use by train_lora.py.
"""
import json
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = DATA_DIR / "sft_train.jsonl"

# Demo examples so training always has data (no reward_log required)
DEMO_SFT = [
    {
        "instruction": "Answer based only on the context. If the context does not contain the answer, say so.",
        "input": "Context:\nRAG systems retrieve documents and then generate answers using an LLM.\n\nQuestion: What is RAG?",
        "output": "RAG stands for Retrieval-Augmented Generation. It is a system that retrieves relevant documents and then uses an LLM to generate answers based on that context.",
    },
    {
        "instruction": "Answer based only on the context. If the context does not contain the answer, say so.",
        "input": "Context:\nMulti-armed bandits balance exploration and exploitation. UCB1 adds an exploration bonus.\n\nQuestion: What is UCB1?",
        "output": "UCB1 is an algorithm used in multi-armed bandits. It balances exploration and exploitation by adding an exploration bonus to the expected reward.",
    },
    {
        "instruction": "Answer based only on the context. If the context does not contain the answer, say so.",
        "input": "Context:\nLoRA trains low-rank adapters on top of a frozen base model, reducing memory and cost.\n\nQuestion: Why use LoRA?",
        "output": "LoRA is used to fine-tune models efficiently by training only low-rank adapters while keeping the base model frozen, which reduces memory use and training cost.",
    },
]


def load_reward_log_answers() -> list[dict]:
    """Load interactions that have stored answers (if we ever log them)."""
    from config import settings
    log_path = settings.base_dir / "reward_log.jsonl"
    if not log_path.exists():
        return []
    out = []
    with open(log_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                # Use only feedback rows or rows that include an answer
                if row.get("answer") and row.get("question"):
                    out.append({
                        "instruction": "Answer based only on the context. If the context does not contain the answer, say so.",
                        "input": f"Context:\n{row.get('context', '')}\n\nQuestion: {row['question']}",
                        "output": row["answer"],
                    })
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = load_reward_log_answers()
    if not records:
        records = DEMO_SFT
    with open(OUT_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} SFT examples to {OUT_PATH}")
    return OUT_PATH


if __name__ == "__main__":
    main()
