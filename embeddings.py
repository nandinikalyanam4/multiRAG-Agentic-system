
# ======================== FILE: embeddings.py ========================
from langchain_openai import OpenAIEmbeddings
from config import settings

def get_embeddings():
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )
