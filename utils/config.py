"""Configuration management for the RAG application."""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Settings
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"  # Cheapest model: $0.0015/1K input, $0.002/1K output
    
    # Embeddings Settings
    embeddings_model: str = "openai"  # Default to OpenAI if API key available, else "sentence-transformers"
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
    
    # Hybrid Search Settings
    use_hybrid_search: bool = True
    semantic_weight: float = 0.7  # Weight for semantic search (0-1)
    keyword_weight: float = 0.3   # Weight for keyword search (0-1)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

