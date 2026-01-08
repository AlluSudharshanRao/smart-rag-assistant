"""User management and identification utilities."""
import hashlib
import streamlit as st
from pathlib import Path
from typing import Optional
from loguru import logger


def get_user_id() -> str:
    """
    Get or create a persistent user ID that survives page refreshes.
    Uses URL query parameters for persistence across sessions.
    """
    # First, check if we have it in session state (fast access)
    if "user_id" in st.session_state and st.session_state.get("user_id_persisted", False):
        return st.session_state.user_id
    
    # Method 1: Check URL query parameters FIRST (persists across refreshes)
    user_id = None
    try:
        # Try to get user_id from query parameters
        query_params = st.query_params
        if "user_id" in query_params:
            user_id = query_params["user_id"]
            # Validate it's a reasonable format (alphanumeric, 12-24 chars)
            if user_id and user_id.isalnum() and 12 <= len(user_id) <= 24:
                st.session_state.user_id = user_id
                st.session_state.user_id_persisted = True
                return user_id
            else:
                # Invalid format, ignore it and generate new one
                user_id = None
    except Exception as e:
        logger.debug(f"Error reading query params: {e}")
    
    # Method 2: Try to get Streamlit's session ID (for first-time users)
    if not user_id:
        try:
            import streamlit.runtime.scriptrunner.script_runner as script_runner
            if hasattr(script_runner, 'get_script_run_ctx'):
                ctx = script_runner.get_script_run_ctx()
                if ctx and hasattr(ctx, 'session_id'):
                    session_id = ctx.session_id
                    if session_id:
                        user_id = hashlib.md5(session_id.encode()).hexdigest()[:12]
        except Exception:
            pass
    
    # Method 3: Generate a new random ID
    if not user_id:
        import secrets
        user_id = secrets.token_hex(6)  # 12 characters
    
    # Store in session state
    st.session_state.user_id = user_id
    
    # Persist to URL query parameters so it survives refresh
    # Only do this if not already persisted to avoid infinite reruns
    if not st.session_state.get("user_id_persisted", False):
        try:
            # Check if query params need updating
            current_params = st.query_params
            if "user_id" not in current_params or current_params.get("user_id") != user_id:
                # Use st.query_params.update() which is safer
                st.query_params.update(user_id=user_id)
                st.session_state.user_id_persisted = True
        except Exception as e:
            # If query params update fails, that's okay - will try again next time
            logger.debug(f"Could not persist user_id to query params: {e}")
    
    return user_id


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

