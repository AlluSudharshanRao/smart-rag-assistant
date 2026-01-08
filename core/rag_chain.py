"""RAG (Retrieval-Augmented Generation) chain implementation."""
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

# Handle vector store imports
try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

from utils.config import settings
from core.embeddings import get_embeddings


class RAGChain:
    """Manages the RAG pipeline: retrieval + generation."""
    
    def __init__(self, vectorstore: Chroma):
        """
        Initialize RAG chain with vector store.
        
        Args:
            vectorstore: Chroma vector store instance
        """
        self.vectorstore = vectorstore
        self.embeddings = get_embeddings()
        
        # Initialize LLM
        if settings.openai_api_key:
            self.llm = ChatOpenAI(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
        else:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable."
            )
        
        # Store retriever for manual RAG implementation
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": settings.top_k_retrieval}
        )
        
        # Create prompt template (optimized for cost - concise)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Answer questions using ONLY the provided context. Be brief and accurate. If context is insufficient, say so."""),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        logger.info("RAG chain initialized")
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Processing question: {question}")
        
        try:
            # Retrieve relevant documents
            docs = self.retriever.invoke(question)
            
            # Combine document content
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Format prompt
            messages = self.prompt.format_messages(question=question, context=context)
            
            # Get answer from LLM
            response = self.llm.invoke(messages)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content[:500],  # First 500 chars
                        "metadata": doc.metadata,
                    }
                    for doc in docs
                ],
            }
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": f"An error was encountered: {str(e)}",
                "source_documents": [],
            }

