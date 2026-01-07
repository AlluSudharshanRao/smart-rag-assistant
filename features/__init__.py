"""Advanced features for RAG system."""
from features.multi_document import DocumentCollection, CollectionManager
from features.evaluation import RAGEvaluator, EvaluationResult

__all__ = ["DocumentCollection", "CollectionManager", "RAGEvaluator", "EvaluationResult"]

