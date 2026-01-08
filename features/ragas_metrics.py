"""RAGAS evaluation metrics implementation."""
from typing import List, Dict, Optional
from loguru import logger

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logger.warning("RAGAS not installed. Install with: pip install ragas datasets")


class RAGASEvaluator:
    """RAGAS metrics evaluation for RAG systems."""
    
    def __init__(self):
        """Initialize RAGAS evaluator."""
        self.available = RAGAS_AVAILABLE
        if not RAGAS_AVAILABLE:
            logger.warning("RAGAS metrics will not be available")
    
    def calculate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str]
    ) -> float:
        """
        Calculate faithfulness score.
        Measures if the answer is grounded in the provided context.
        
        Returns:
            Score between 0 and 1 (1 = fully faithful to context)
        """
        if not self.available or not contexts:
            return 0.0
        
        try:
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts]
            })
            
            result = evaluate(
                dataset,
                metrics=[faithfulness],
            )
            
            score = result["faithfulness"]
            return float(score) if score is not None else 0.0
        except Exception as e:
            logger.error(f"Error calculating faithfulness: {e}")
            return 0.0
    
    def calculate_answer_relevancy(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Calculate answer relevancy score.
        Measures how relevant the answer is to the question.
        
        Returns:
            Score between 0 and 1 (1 = highly relevant)
        """
        if not self.available:
            return 0.0
        
        try:
            dataset = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [[""]]  # Not needed for this metric
            })
            
            result = evaluate(
                dataset,
                metrics=[answer_relevancy],
            )
            
            score = result["answer_relevancy"]
            return float(score) if score is not None else 0.0
        except Exception as e:
            logger.error(f"Error calculating answer relevancy: {e}")
            return 0.0
    
    def calculate_context_precision(
        self,
        question: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Calculate context precision.
        Measures the precision of retrieved contexts.
        
        Returns:
            Score between 0 and 1 (1 = high precision)
        """
        if not self.available or not contexts:
            return 0.0
        
        try:
            # Use question as ground truth if not provided
            ground_truth = ground_truth or question
            
            dataset = Dataset.from_dict({
                "question": [question],
                "contexts": [contexts],
                "ground_truth": [ground_truth]
            })
            
            result = evaluate(
                dataset,
                metrics=[context_precision],
            )
            
            score = result["context_precision"]
            return float(score) if score is not None else 0.0
        except Exception as e:
            logger.error(f"Error calculating context precision: {e}")
            return 0.0
    
    def calculate_context_recall(
        self,
        question: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Calculate context recall.
        Measures the recall of retrieved contexts.
        
        Returns:
            Score between 0 and 1 (1 = high recall)
        """
        if not self.available or not contexts:
            return 0.0
        
        try:
            # Use question as ground truth if not provided
            ground_truth = ground_truth or question
            
            dataset = Dataset.from_dict({
                "question": [question],
                "contexts": [contexts],
                "ground_truth": [ground_truth]
            })
            
            result = evaluate(
                dataset,
                metrics=[context_recall],
            )
            
            score = result["context_recall"]
            return float(score) if score is not None else 0.0
        except Exception as e:
            logger.error(f"Error calculating context recall: {e}")
            return 0.0
    
    def evaluate_all(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate all RAGAS metrics at once.
        
        Returns:
            Dictionary with all metric scores
        """
        if not self.available:
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
            }
        
        return {
            "faithfulness": self.calculate_faithfulness(question, answer, contexts),
            "answer_relevancy": self.calculate_answer_relevancy(question, answer),
            "context_precision": self.calculate_context_precision(question, contexts, ground_truth),
            "context_recall": self.calculate_context_recall(question, contexts, ground_truth),
        }

