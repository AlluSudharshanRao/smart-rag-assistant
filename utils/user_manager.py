"""User management and identification utilities."""
import hashlib
import streamlit as st
from pathlib import Path
from typing import Optional


def get_user_id() -> str:
    """
    Get or create a unique user ID for the current session.
    Uses a combination of session state and browser-based identification.
    """
    if "user_id" not in st.session_state:
        # Try multiple methods to get a stable user identifier
        user_id = None
        
        # Method 1: Try to get Streamlit's session ID (works in some deployments)
        try:
            import streamlit.runtime.scriptrunner.script_runner as script_runner
            if hasattr(script_runner, 'get_script_run_ctx'):
                ctx = script_runner.get_script_run_ctx()
                if ctx and hasattr(ctx, 'session_id'):
                    session_id = ctx.session_id
                    if session_id:
                        user_id = hashlib.md5(session_id.encode()).hexdigest()[:12]
        except:
            pass
        
        # Method 2: Try runtime instance (alternative approach)
        if not user_id:
            try:
                runtime = st.runtime.get_instance()
                if hasattr(runtime, '_session_mgr'):
                    sessions = runtime._session_mgr.list_sessions()
                    if sessions:
                        session_id = sessions[0].id
                        if session_id:
                            user_id = hashlib.md5(session_id.encode()).hexdigest()[:12]
            except:
                pass
        
        # Method 3: Fallback - generate random ID and store in session
        if not user_id:
            import secrets
            user_id = secrets.token_hex(6)
        
        st.session_state.user_id = user_id
    
    return st.session_state.user_id


def get_user_data_dir(base_dir: Path, user_id: str) -> Path:
    """Get user-specific data directory path."""
    user_dir = base_dir / "users" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def get_user_chat_path(base_dir: Path, user_id: str) -> Optional[Path]:
    """Get user-specific chat history file path."""
    if base_dir is None:
        return None
    user_dir = get_user_data_dir(base_dir, user_id)
    return user_dir / "chat_history.json"


def get_user_evaluations_path(base_dir: Path, user_id: str) -> Optional[Path]:
    """Get user-specific evaluations file path."""
    if base_dir is None:
        return None
    user_dir = get_user_data_dir(base_dir, user_id)
    return user_dir / "evaluations.json"


def get_user_chromadb_dir(base_dir: Path, user_id: str) -> Path:
    """Get user-specific ChromaDB directory path."""
    if base_dir is None:
        # Fallback to relative path if base_dir is None
        return Path(f"./chroma_db/users/{user_id}")
    user_dir = get_user_data_dir(base_dir, user_id)
    chroma_dir = user_dir / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chroma_dir


def get_user_collections_path(base_dir: Path, user_id: str) -> Optional[Path]:
    """Get user-specific collections list file path."""
    if base_dir is None:
        return None
    user_dir = get_user_data_dir(base_dir, user_id)
    return user_dir / "collections.json"


def get_user_collection_name(user_id: str, collection_name: str) -> str:
    """Get user-scoped collection name."""
    # Format: user_{user_id}_{collection_name}
    # This ensures each user has isolated collections
    return f"user_{user_id}_{collection_name}"


def get_all_user_dirs(base_dir: Path) -> list:
    """Get all user directories for admin/overall metrics."""
    if base_dir is None:
        return []
    users_dir = base_dir / "users"
    if not users_dir.exists():
        return []
    return [d for d in users_dir.iterdir() if d.is_dir()]


def load_all_user_evaluations(base_dir: Path):
    """Load evaluation results from all users for aggregate metrics."""
    from features.evaluation import RAGEvaluator, EvaluationResult
    from loguru import logger
    import json
    
    all_results = []
    user_dirs = get_all_user_dirs(base_dir)
    
    for user_dir in user_dirs:
        eval_path = user_dir / "evaluations.json"
        if eval_path.exists():
            try:
                with open(eval_path, 'r') as f:
                    data = json.load(f)
                    results = data.get("results", [])
                    for r in results:
                        # Add user_id to each result for tracking
                        result_dict = {
                            "question": r.get("question", ""),
                            "expected_answer": r.get("expected", ""),
                            "actual_answer": r.get("actual", ""),
                            "retrieved_docs": r.get("retrieved_docs", 0),
                            "relevance_score": r.get("relevance", 0.0),
                            "answer_quality": r.get("quality", 0.0),
                            "response_time": r.get("response_time", 0.0) or 0.0,
                            "timestamp": r.get("timestamp", ""),
                            "user_id": user_dir.name,
                        }
                        all_results.append(result_dict)
            except Exception as e:
                from loguru import logger
                logger.error(f"Failed to load evaluations from {eval_path}: {e}")
    
    return all_results


def get_overall_metrics(base_dir: Path) -> dict:
    """Calculate aggregate metrics across all users."""
    all_results = load_all_user_evaluations(base_dir)
    
    if not all_results:
        return {
            "total_users": 0,
            "total_queries": 0,
            "avg_relevance": 0.0,
            "avg_quality": 0.0,
            "avg_response_time": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    
    # Get unique user count
    unique_users = set(r.get("user_id", "") for r in all_results if r.get("user_id"))
    
    # Calculate aggregate metrics
    total_queries = len(all_results)
    avg_relevance = sum(r.get("relevance_score", 0.0) for r in all_results) / total_queries if total_queries > 0 else 0.0
    avg_quality = sum(r.get("answer_quality", 0.0) for r in all_results) / total_queries if total_queries > 0 else 0.0
    
    rt_values = [r.get("response_time", 0.0) for r in all_results if r.get("response_time", 0.0) > 0]
    avg_rt = (sum(rt_values) / len(rt_values)) if rt_values else 0.0
    
    # Calculate precision and recall
    threshold = 0.7
    relevant_retrieved = sum(1 for r in all_results if r.get("relevance_score", 0.0) >= threshold)
    precision = relevant_retrieved / total_queries if total_queries > 0 else 0.0
    
    avg_relevance_for_recall = avg_relevance
    estimated_total_relevant = total_queries * (avg_relevance_for_recall / threshold) if threshold > 0 else total_queries
    estimated_total_relevant = max(total_queries, estimated_total_relevant)
    recall = relevant_retrieved / estimated_total_relevant if estimated_total_relevant > 0 else 0.0
    recall = min(1.0, recall)
    
    return {
        "total_users": len(unique_users),
        "total_queries": total_queries,
        "avg_relevance": round(avg_relevance, 4),
        "avg_quality": round(avg_quality, 4),
        "avg_response_time": round(avg_rt, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
    }

