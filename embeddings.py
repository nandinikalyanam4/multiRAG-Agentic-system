
# ======================== FILE: embeddings.py ========================
from langchain_openai import OpenAIEmbeddings
from config import settings


def get_embeddings():
    # Pass API key explicitly so we do not rely on environment variable loading
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
