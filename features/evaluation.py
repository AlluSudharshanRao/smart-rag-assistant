"""Evaluation metrics for RAG system."""
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger


@dataclass
class EvaluationResult:
    """Stores evaluation metrics."""
    question: str
    expected_answer: str
    actual_answer: str
    retrieved_docs: int
    relevance_score: float
    answer_quality: float
    response_time: float = 0.0  # Response time in seconds
    timestamp: str = ""


class RAGEvaluator:
    """Evaluates RAG system performance."""
    
    def __init__(self):
        self.results: List[EvaluationResult] = []
        self.metrics = {
            "total_queries": 0,
            "avg_relevance": 0.0,
            "avg_quality": 0.0,
            "avg_response_time": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    
    def evaluate(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
        retrieved_docs: int,
        relevance_score: float = 0.0,
        answer_quality: float = 0.0,
        response_time: float = 0.0,
    ) -> EvaluationResult:
        """Evaluate a single query."""
        from datetime import datetime
        
        result = EvaluationResult(
            question=question,
            expected_answer=expected_answer,
            actual_answer=actual_answer,
            retrieved_docs=retrieved_docs,
            relevance_score=relevance_score,
            answer_quality=answer_quality,
            response_time=response_time,
            timestamp=datetime.now().isoformat(),
        )
        
        self.results.append(result)
        self._update_metrics()
        
        return result
    
    def _update_metrics(self):
        """Update aggregate metrics."""
        if not self.results:
            return
        
        self.metrics["total_queries"] = len(self.results)
        
        # Calculate averages with proper rounding
        avg_rel = sum(r.relevance_score for r in self.results) / len(self.results)
        avg_qual = sum(r.answer_quality for r in self.results) / len(self.results)
        rt_values = [r.response_time for r in self.results if hasattr(r, 'response_time') and r.response_time and r.response_time > 0]
        avg_rt = (sum(rt_values) / len(rt_values)) if rt_values else 0.0
        
        self.metrics["avg_relevance"] = round(avg_rel, 4)
        self.metrics["avg_quality"] = round(avg_qual, 4)
        self.metrics["avg_response_time"] = round(avg_rt, 4)
    
    def get_summary(self) -> Dict:
        """Get evaluation summary."""
        summary = {
            **self.metrics,
            "total_evaluations": len(self.results),
        }
        # Ensure avg_response_time is included
        if "avg_response_time" not in summary:
            rt_values = [r.response_time for r in self.results if hasattr(r, 'response_time') and r.response_time and r.response_time > 0]
            summary["avg_response_time"] = round((sum(rt_values) / len(rt_values)) if rt_values else 0.0, 4)
        return summary
    
    def export_results(self, filepath: str):
        """Export results to JSON."""
        data = {
            "metrics": self.metrics,
            "results": [
                {
                    "question": r.question,
                    "expected": r.expected_answer,
                    "actual": r.actual_answer,
                    "retrieved_docs": r.retrieved_docs,
                    "relevance": r.relevance_score,
                    "quality": r.answer_quality,
                    "response_time": r.response_time,
                    "timestamp": r.timestamp,
                }
                for r in self.results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(self.results)} evaluation results to {filepath}")

    # ------------------------
    # Persistence helpers
    # ------------------------
    def save_to_disk(self, filepath: str):
        """Persist metrics and results to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.export_results(str(path))
        logger.info(f"Saved evaluator state to {path}")

    def load_from_disk(self, filepath: str):
        """Load metrics and results from disk if present."""
        path = Path(filepath)
        if not path.exists():
            logger.info(f"No evaluator file at {path}, starting fresh")
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            results = data.get("results", [])
            self.results = [
                EvaluationResult(
                    question=r.get("question", ""),
                    expected_answer=r.get("expected", ""),
                    actual_answer=r.get("actual", ""),
                    retrieved_docs=r.get("retrieved_docs", 0),
                    relevance_score=r.get("relevance", 0.0),
                    answer_quality=r.get("quality", 0.0),
                    response_time=r.get("response_time", 0.0) or 0.0,
                    timestamp=r.get("timestamp", ""),
                )
                for r in results
            ]
            # Load metrics if available; recompute to stay consistent
            self._update_metrics()
            logger.info(f"Loaded evaluator state from {path} ({len(self.results)} results)")
        except Exception as e:
            logger.error(f"Failed to load evaluator from {path}: {e}")
    
    def calculate_precision_recall(self, threshold: float = 0.7) -> Tuple[float, float]:
        """
        Calculate precision and recall based on relevance scores.
        
        Precision: Of all retrieved documents, what % are relevant?
        Recall: Of all relevant documents, what % were retrieved?
        """
        if not self.results:
            return 0.0, 0.0
        
        # Count relevant results (above threshold)
        relevant_retrieved = sum(1 for r in self.results if r.relevance_score >= threshold)
        total_retrieved = len(self.results)
        
        # Precision: relevant retrieved / total retrieved
        precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0.0
        
        # For recall, we need to estimate total relevant documents
        # We'll use the average relevance score as a proxy
        # If avg relevance is high, we're retrieving most relevant docs
        avg_relevance = sum(r.relevance_score for r in self.results) / len(self.results) if self.results else 0.0
        
        # Estimate: if we have high relevance scores, we're likely retrieving most relevant docs
        # This is a simplified recall calculation
        # In a real system, you'd need ground truth of all relevant docs
        estimated_total_relevant = total_retrieved * (avg_relevance / threshold) if threshold > 0 else total_retrieved
        estimated_total_relevant = max(total_retrieved, estimated_total_relevant)  # At least what we retrieved
        
        recall = relevant_retrieved / estimated_total_relevant if estimated_total_relevant > 0 else 0.0
        recall = min(1.0, recall)  # Cap at 100%
        
        self.metrics["precision"] = precision
        self.metrics["recall"] = recall
        
        return precision, recall

