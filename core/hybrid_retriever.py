"""Hybrid retriever combining semantic and keyword search."""
from typing import List, Dict, Optional
from langchain_core.documents import Document
from loguru import logger

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    logger.warning("rank-bm25 not installed. Install with: pip install rank-bm25")

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

from langchain_community.vectorstores import Chroma


class HybridRetriever:
    """Combines semantic (vector) and keyword (BM25) search."""
    
    def __init__(
        self,
        vectorstore: Chroma,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 3
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            vectorstore: Chroma vector store for semantic search
            semantic_weight: Weight for semantic search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            top_k: Number of documents to retrieve
        """
        self.vectorstore = vectorstore
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k
        
        # Initialize BM25 if available
        self.bm25_index = None
        self.documents = []
        self.document_ids = []
        self.document_metadatas = []
        self._build_bm25_index()
        
        # Semantic retriever
        self.semantic_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": top_k * 2}  # Get more for re-ranking
        )
    
    def _build_bm25_index(self):
        """Build BM25 index from documents in vectorstore."""
        if BM25Okapi is None:
            logger.warning("BM25 not available. Using semantic search only.")
            return
        
        try:
            # Get all documents from vectorstore
            # Note: This is a simplified approach - in production, you'd want to cache this
            collection = self.vectorstore._collection
            results = collection.get(include=["documents", "ids", "metadatas"])
            
            if results and "documents" in results and results["documents"]:
                self.documents = results["documents"]
                self.document_ids = results.get("ids", [])
                self.document_metadatas = results.get("metadatas", [])
                # Tokenize documents for BM25
                tokenized_docs = [doc.lower().split() if doc else [] for doc in self.documents]
                if tokenized_docs:
                    self.bm25_index = BM25Okapi(tokenized_docs)
                    logger.info(f"Built BM25 index with {len(self.documents)} documents")
                else:
                    self.bm25_index = None
            else:
                logger.warning("No documents found in vectorstore for BM25 indexing")
                self.documents = []
                self.document_ids = []
                self.document_metadatas = []
        except Exception as e:
            logger.error(f"Error building BM25 index: {e}")
            self.bm25_index = None
            self.documents = []
            self.document_ids = []
            self.document_metadatas = []
    
    def _keyword_search(self, query: str, top_k: int) -> List[tuple]:
        """
        Perform keyword search using BM25.
        
        Returns:
            List of (score, doc_index) tuples
        """
        if self.bm25_index is None or not self.documents:
            return []
        
        try:
            tokenized_query = query.lower().split()
            scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top K documents with scores
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]
            
            return [(scores[i], i) for i in top_indices]
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _semantic_search(self, query: str) -> List[Document]:
        """Perform semantic search using vector store."""
        try:
            docs = self.semantic_retriever.invoke(query)
            return docs
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Hybrid retrieval combining semantic and keyword search.
        
        Args:
            query: Search query
            
        Returns:
            List of ranked documents
        """
        # Get semantic results
        semantic_docs = self._semantic_search(query)
        
        # Get keyword results
        keyword_results = self._keyword_search(query, self.top_k * 2)
        
        # If no BM25, return semantic results
        if not self.bm25_index or not keyword_results:
            logger.debug("Using semantic search only (BM25 not available)")
            return semantic_docs[:self.top_k]
        
        # Combine and re-rank using better matching strategy
        # Create a map of semantic docs by content hash for matching
        semantic_doc_map = {}
        for doc in semantic_docs:
            # Use first 500 chars as key for matching
            content_key = doc.page_content[:500].lower().strip()
            semantic_doc_map[content_key] = doc
        
        # Score documents (using content as key)
        doc_scores: Dict[str, float] = {}
        
        # Score semantic results
        for i, doc in enumerate(semantic_docs):
            content_key = doc.page_content[:500].lower().strip()
            # Normalize rank (first doc gets highest score)
            semantic_score = (len(semantic_docs) - i) / len(semantic_docs) if semantic_docs else 0
            doc_scores[content_key] = doc_scores.get(content_key, 0) + semantic_score * self.semantic_weight
        
        # Score keyword results and match to semantic docs
        keyword_doc_contents = set()
        for score, doc_idx in keyword_results:
            if doc_idx < len(self.documents):
                doc_content = self.documents[doc_idx]
                if not doc_content:
                    continue
                content_key = doc_content[:500].lower().strip()
                keyword_doc_contents.add(content_key)
                
                # If this doc is in semantic results, add keyword score
                if content_key in semantic_doc_map:
                    # Normalize BM25 score (log scale normalization)
                    normalized_score = min(score / (score + 10), 1.0)  # Better normalization
                    doc_scores[content_key] = doc_scores.get(content_key, 0) + normalized_score * self.keyword_weight
                # If not in semantic, we'll add it separately if needed
        
        # Re-rank semantic docs by hybrid scores
        def get_score(doc: Document) -> float:
            content_key = doc.page_content[:500].lower().strip()
            return doc_scores.get(content_key, 0)
        
        # Sort semantic docs by hybrid score
        ranked_semantic_docs = sorted(semantic_docs, key=get_score, reverse=True)
        
        # Start with top-ranked semantic docs
        result_docs = ranked_semantic_docs[:self.top_k]
        
        # Add high-scoring keyword-only docs if we have space and they're not duplicates
        remaining = self.top_k - len(result_docs)
        if remaining > 0 and keyword_doc_contents:
            existing_content_keys = {doc.page_content[:500].lower().strip() for doc in result_docs}
            # Get keyword-only docs that aren't already included
            for doc_idx in [idx for score, idx in sorted(keyword_results, reverse=True)[:self.top_k * 2]]:
                if doc_idx < len(self.documents) and remaining > 0:
                    doc_content = self.documents[doc_idx]
                    if not doc_content:
                        continue
                    content_key = doc_content[:500].lower().strip()
                    if content_key not in existing_content_keys:
                        # Get metadata if available
                        metadata = self.document_metadatas[doc_idx] if doc_idx < len(self.document_metadatas) else {}
                        result_docs.append(Document(
                            page_content=doc_content,
                            metadata={**metadata, "source": "bm25", "retrieval_method": "keyword"}
                        ))
                        existing_content_keys.add(content_key)
                        remaining -= 1
        
        logger.debug(f"Hybrid retrieval: {len(result_docs)} documents (semantic: {len(semantic_docs)}, keyword matches: {len(keyword_results)})")
        return result_docs[:self.top_k]

