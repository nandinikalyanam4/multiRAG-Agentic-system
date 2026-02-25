
"""
Stores all interactions and feedback for learning.
Uses a simple JSON-lines file — no database needed.
"""
import json, os, time
from pathlib import Path
from config import settings
from threading import Lock

REWARD_LOG_PATH = settings.base_dir / "reward_log.jsonl"
lock = Lock()


def log_interaction(data: dict):
    """Append an interaction record to the log."""
    data["timestamp"] = time.time()
    with lock:
        with open(REWARD_LOG_PATH, "a") as f:
            f.write(json.dumps(data) + "\n")


def load_all_interactions() -> list[dict]:
    """Load all logged interactions."""
    if not REWARD_LOG_PATH.exists():
        return []
    with open(REWARD_LOG_PATH, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_interactions_for_agent(agent_name: str) -> list[dict]:
    """Load interactions for a specific agent."""
    return [i for i in load_all_interactions() if i.get("agent") == agent_name]
