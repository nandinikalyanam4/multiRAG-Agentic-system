# ======================== FILE: config.py ========================
import os
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API Keys
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "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")

    # Directories
    base_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    chroma_dir: Path = Path("./data/chroma")

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