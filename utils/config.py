"""Configuration management for the RAG application."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"  # Cheapest model: $0.0015/1K input, $0.002/1K output
    
    # Embeddings Settings
    embeddings_model: str = "sentence-transformers"  # or "openai"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"  # Fast and efficient
    
    # Vector Database Settings
    vector_db_type: str = "chroma"  # or "qdrant"
    chroma_persist_directory: str = "./chroma_db"
    
    # Document Processing Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # RAG Settings
    top_k_retrieval: int = 3
    temperature: float = 0.3  # Lower = more focused, cheaper (less randomness)
    max_tokens: int = 300  # Reduced to save costs (shorter answers)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

