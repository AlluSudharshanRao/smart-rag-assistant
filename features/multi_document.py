"""Multi-document collection management."""
from typing import List, Dict, Optional, Tuple
from langchain_community.vectorstores import Chroma
from loguru import logger
import re

from core.embeddings import get_embeddings
from utils.config import settings


class DocumentCollection:
    """Manages multiple document collections."""
    
    @staticmethod
    def validate_collection_name(name: str) -> Tuple[bool, str]:
        """
        Validate collection name according to ChromaDB rules.
        
        Rules:
        - 3-512 characters
        - Only [a-zA-Z0-9._-]
        - Must start and end with [a-zA-Z0-9]
        
        Returns:
            (is_valid, error_message)
        """
        if not name:
            return False, "Collection name cannot be empty"
        
        if len(name) < 3:
            return False, f"Collection name must be at least 3 characters (got {len(name)})"
        
        if len(name) > 512:
            return False, f"Collection name must be 512 characters or less (got {len(name)})"
        
        # Check if starts and ends with alphanumeric
        if not name[0].isalnum() or not name[-1].isalnum():
            return False, "Collection name must start and end with a letter or number (a-z, A-Z, 0-9)"
        
        # Check for invalid characters
        if not re.match(r'^[a-zA-Z0-9._-]+$', name):
            return False, "Collection name can only contain letters, numbers, dots, underscores, and hyphens (a-z, A-Z, 0-9, ., _, -)"
        
        return True, ""
    
    def __init__(self, collection_name: str = "default"):
        """
        Initialize a document collection.
        
        Args:
            collection_name: Name of the collection
        """
        # Validate collection name
        is_valid, error_msg = self.validate_collection_name(collection_name)
        if not is_valid:
            raise ValueError(f"Invalid collection name: {error_msg}")
        
        self.collection_name = collection_name
        self.embeddings = get_embeddings()
        self.vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=settings.chroma_persist_directory,
            embedding_function=self.embeddings,
        )
        logger.info(f"Initialized collection: {collection_name}")
    
    def add_documents(self, documents: List, metadata: Optional[Dict] = None):
        """Add documents to collection."""
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)
        
        self.vectorstore.add_documents(documents)
        self.vectorstore.persist()
        logger.info(f"Added {len(documents)} documents to {self.collection_name}")
    
    def get_retriever(self, top_k: int = None):
        """Get retriever for this collection."""
        top_k = top_k or settings.top_k_retrieval
        return self.vectorstore.as_retriever(
            search_kwargs={"k": top_k}
        )
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            return {
                "name": self.collection_name,
                "document_count": count,
                "persist_directory": settings.chroma_persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "document_count": 0,
                "error": str(e),
            }
    
    def delete_documents_by_source(self, source_file: str) -> int:
        """
        Delete all documents with a specific source file.
        
        Args:
            source_file: The source file name to delete
            
        Returns:
            Number of documents deleted
        """
        try:
            collection = self.vectorstore._collection
            
            # Get all documents with this source file
            results = collection.get(
                where={"source_file": source_file}
            )
            
            if results and len(results.get("ids", [])) > 0:
                ids_to_delete = results["ids"]
                collection.delete(ids=ids_to_delete)
                self.vectorstore.persist()
                logger.info(f"Deleted {len(ids_to_delete)} documents with source_file={source_file} from {self.collection_name}")
                return len(ids_to_delete)
            else:
                logger.warning(f"No documents found with source_file={source_file}")
                return 0
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            raise
    
    def delete_documents_by_ids(self, ids: List[str]) -> int:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        try:
            collection = self.vectorstore._collection
            collection.delete(ids=ids)
            self.vectorstore.persist()
            logger.info(f"Deleted {len(ids)} documents from {self.collection_name}")
            return len(ids)
        except Exception as e:
            logger.error(f"Error deleting documents by IDs: {e}")
            raise
    
    def get_documents_by_source(self) -> Dict[str, Dict]:
        """
        Get all documents grouped by source file.
        
        Returns:
            Dictionary mapping source_file to document info
        """
        try:
            collection = self.vectorstore._collection
            all_docs = collection.get()
            
            source_files = {}
            if all_docs and len(all_docs.get("ids", [])) > 0:
                for i, doc_id in enumerate(all_docs.get("ids", [])):
                    metadata = all_docs.get("metadatas", [{}])[i] if all_docs.get("metadatas") else {}
                    source_file = metadata.get("source_file", "Unknown")
                    
                    if source_file not in source_files:
                        source_files[source_file] = {
                            "count": 0,
                            "ids": [],
                            "metadata": metadata,
                        }
                    source_files[source_file]["count"] += 1
                    source_files[source_file]["ids"].append(doc_id)
            
            return source_files
        except Exception as e:
            logger.error(f"Error getting documents by source: {e}")
            return {}
    
    @staticmethod
    def list_collections() -> List[str]:
        """List all available collections."""
        # This would require accessing Chroma's collection manager
        # For now, return default
        return ["default"]


class CollectionManager:
    """Manages multiple document collections."""
    
    def __init__(self):
        self.collections: Dict[str, DocumentCollection] = {}
        self.current_collection: Optional[str] = None
    
    def create_collection(self, name: str) -> DocumentCollection:
        """Create a new collection."""
        if name in self.collections:
            logger.warning(f"Collection {name} already exists")
            return self.collections[name]
        
        collection = DocumentCollection(collection_name=name)
        self.collections[name] = collection
        return collection
    
    def get_collection(self, name: str) -> DocumentCollection:
        """Get a collection by name."""
        if name not in self.collections:
            self.collections[name] = DocumentCollection(collection_name=name)
        return self.collections[name]
    
    def switch_collection(self, name: str):
        """Switch to a different collection."""
        self.current_collection = name
        logger.info(f"Switched to collection: {name}")
    
    def get_current_collection(self) -> Optional[DocumentCollection]:
        """Get the current active collection."""
        if not self.current_collection:
            self.current_collection = "default"
        
        return self.get_collection(self.current_collection)
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete a collection completely from ChromaDB.
        
        Args:
            name: Name of the collection to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if name == "default":
                raise ValueError("Cannot delete the default collection")
            
            # Get the collection to delete
            collection = self.get_collection(name)
            
            # Delete all documents in the collection
            collection_obj = collection.vectorstore._collection
            all_docs = collection_obj.get()
            
            if all_docs and len(all_docs.get("ids", [])) > 0:
                collection_obj.delete(ids=all_docs["ids"])
            
            # Delete the collection from ChromaDB
            # Note: ChromaDB doesn't have a direct delete_collection method
            # The system requires using the client to delete it
            try:
                client = collection.vectorstore._client
                client.delete_collection(name=name)
                logger.info(f"Deleted collection {name} from ChromaDB")
            except Exception as e:
                logger.warning(f"Could not delete collection from ChromaDB client: {e}")
                # Collection will be recreated if accessed again, but documents are deleted
            
            # Remove from manager
            if name in self.collections:
                del self.collections[name]
            
            # Switch to default if current collection was deleted
            if self.current_collection == name:
                self.current_collection = "default"
            
            logger.info(f"Successfully deleted collection: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {name}: {e}")
            raise

