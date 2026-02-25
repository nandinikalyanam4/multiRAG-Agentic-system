# ======================== FILE: config.py ========================
import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Keys (must be provided via environment or .env)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

    # Directories
    base_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    chroma_dir: Path = Path("./data/chroma")

    # Optional: path to LoRA adapters for local generation (set LORA_ADAPTERS_PATH in .env)
    lora_adapters_path: Optional[Path] = None

    # Models
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    vision_model: str = "gpt-4o-mini"

    # Chunking defaults
    chunk_size: int = 512
    chunk_overlap: int = 50
    sentence_window_size: int = 3  # sentences around match

    # Retrieval
    default_top_k: int = 5
    relevance_threshold: float = 0.7
    max_agentic_retries: int = 3

    class Config:
        env_file = ".env"

settings = Settings()
for d in [settings.upload_dir, settings.chroma_dir]:
    d.mkdir(parents=True, exist_ok=True)