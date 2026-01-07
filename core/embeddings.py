"""Embedding generation utilities."""
from typing import List
import os
from langchain.embeddings.base import Embeddings
from langchain_openai import OpenAIEmbeddings
from loguru import logger

from utils.config import settings

# Lazy loading - only import sentence_transformers when needed
_sentence_transformer_model = None


class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper for Sentence Transformers to work with LangChain."""
    
    def __init__(self, model_name: str = None):
        global _sentence_transformer_model
        model_name = model_name or settings.sentence_transformer_model
        logger.info(f"Loading Sentence Transformer model: {model_name}")
        try:
            # Suppress TensorFlow warnings/errors temporarily
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            import warnings
            warnings.filterwarnings('ignore')
            
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            _sentence_transformer_model = self.model
        except Exception as e:
            logger.error(f"Error loading SentenceTransformer: {e}")
            error_msg = (
                f"Failed to load SentenceTransformer. This may be due to TensorFlow/DLL issues. "
                f"Error: {str(e)}. "
                f"Solutions: 1) Use OpenAI embeddings (set OPENAI_API_KEY and EMBEDDINGS_MODEL=openai), "
                f"2) Install Visual C++ Build Tools, or 3) Use a different Python environment."
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()


def get_embeddings() -> Embeddings:
    """
    Get embeddings instance based on configuration.
    Prefers OpenAI if API key is available, falls back to sentence-transformers.
    
    Returns:
        Embeddings instance (OpenAI or Sentence Transformers)
    """
    # Prefer OpenAI if API key is available
    if settings.openai_api_key and (settings.embeddings_model == "openai" or settings.embeddings_model == "sentence-transformers"):
        try:
            logger.info("Using OpenAI embeddings (API key available)")
            return OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
        except Exception as e:
            logger.warning(f"Failed to use OpenAI embeddings: {e}. Falling back to sentence-transformers.")
    
    # Fall back to sentence-transformers
    if settings.embeddings_model == "sentence-transformers" or not settings.openai_api_key:
        logger.info("Using Sentence Transformers embeddings")
        return SentenceTransformerEmbeddings()
    
    # If explicitly set to OpenAI but no key, raise error
    if settings.embeddings_model == "openai" and not settings.openai_api_key:
        raise ValueError(
            "OpenAI API key required when using OpenAI embeddings. "
            "Set OPENAI_API_KEY environment variable or use sentence-transformers."
        )
    
    # Default fallback
    logger.info("Using Sentence Transformers embeddings (default)")
    return SentenceTransformerEmbeddings()

