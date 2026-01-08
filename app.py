"""Streamlit frontend for the RAG Document Assistant."""
import os
import tempfile
import streamlit as st
from pathlib import Path

try:
    from langchain.vectorstores import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma
from loguru import logger

from core.document_processor import DocumentProcessor
from core.embeddings import get_embeddings
from core.rag_chain import RAGChain
from utils.config import settings
from utils.user_manager import (
    get_user_id, 
    get_user_chat_path, 
    get_user_evaluations_path,
    get_overall_metrics
)
from features.multi_document import CollectionManager
from features.evaluation import RAGEvaluator

# Page configuration
st.set_page_config(
    page_title="Smart RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Smart RAG Document Assistant - AI-powered document Q&A system"
    }
)

# Custom CSS for modern, professional UI
st.markdown("""
<style>
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1f2937;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    h3 {
        color: #374151;
        font-weight: 600;
        margin-top: 1rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #1f2937;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.25);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #6b7280;
        font-weight: 500;
    }
    
    /* Info boxes */
    .stInfo {
        background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%);
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 4px solid #10b981;
        border-radius: 8px;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        border-radius: 8px;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #ef4444;
        border-radius: 8px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f9fafb;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: #f9fafb;
        border-radius: 6px;
        font-weight: 600;
    }
    
    /* Chat message styling */
    [data-testid="stChatMessage"] {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        padding: 2rem;
        background: #f8fafc;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f0f4ff;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Code blocks */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }
    
    /* JSON viewer */
    pre {
        border-radius: 8px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #e5e7eb, transparent);
        margin: 2rem 0;
    }
    
    /* Card-like containers */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Radio button styling */
    .stRadio > div {
        gap: 1rem;
    }
    
    .stRadio > div > label {
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .stRadio > div > label:hover {
        border-color: #667eea;
        background: #f0f4ff;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.25);
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.4);
    }
    
    /* Sidebar header */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1f2937;
    }
    
    /* Caption styling */
    .stCaption {
        color: #6b7280;
        font-size: 0.85rem;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
# For cloud deployment, data persistence may not be available
# Use try/except to handle cases where file system is read-only
try:
    DATA_DIR = Path("./data")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Test write access
    test_file = DATA_DIR / ".test_write"
    test_file.write_text("test")
    test_file.unlink()
except (PermissionError, OSError, Exception) as e:
    # Cloud environment may not allow file creation
    # Use temporary in-memory storage instead
    logger.warning(f"File system not writable, using in-memory storage: {e}")
    DATA_DIR = None

# Get or create user ID for this session
try:
    user_id = get_user_id()
except Exception as e:
    logger.error(f"Error generating user ID: {e}")
    # Fallback to simple random ID
    import secrets
    user_id = secrets.token_hex(6)
    if "user_id" not in st.session_state:
        st.session_state.user_id = user_id

# Get user-specific paths
EVALS_PATH = get_user_evaluations_path(DATA_DIR, user_id) if DATA_DIR else None
CHAT_PATH = get_user_chat_path(DATA_DIR, user_id) if DATA_DIR else None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "collection_manager" not in st.session_state:
    st.session_state.collection_manager = CollectionManager()
if "evaluator" not in st.session_state:
    st.session_state.evaluator = RAGEvaluator()
if "current_collection" not in st.session_state:
    st.session_state.current_collection = "default"
if "collections_list" not in st.session_state:
    st.session_state.collections_list = ["default"]
if "state_loaded" not in st.session_state:
    # Load persisted evaluator for this user (if file system is available)
    if EVALS_PATH:
        try:
            st.session_state.evaluator.load_from_disk(str(EVALS_PATH))
        except Exception:
            pass
    # Load persisted chat history for this user (if file system is available)
    if CHAT_PATH:
        try:
            if CHAT_PATH.exists():
                import json as _json
                st.session_state.chat_history = _json.loads(CHAT_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    st.session_state.state_loaded = True


def initialize_vectorstore():
    """Initialize or get existing vector store."""
    if st.session_state.vectorstore is None:
        try:
            embeddings = get_embeddings()
            st.session_state.vectorstore = Chroma(
                persist_directory=settings.chroma_persist_directory,
                embedding_function=embeddings,
            )
        except Exception as e:
            st.error(f"❌ Error initializing embeddings: {str(e)}")
            st.info("💡 **Solution:** Set `EMBEDDINGS_MODEL=openai` and `OPENAI_API_KEY` in the `.env` file to use OpenAI embeddings instead.")
            raise


def process_uploaded_file(uploaded_file, collection_name: str = None):
    """Process uploaded file and add to vector store."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Process document
        processor = DocumentProcessor()
        chunks = processor.process_file(tmp_path)
        
        # Use collection manager
        collection_name = collection_name or st.session_state.current_collection
        collection = st.session_state.collection_manager.get_collection(collection_name)
        
        # Add metadata including collection name
        file_size = len(uploaded_file.read())
        uploaded_file.seek(0)  # Reset file pointer
        metadata = {
            "source_file": uploaded_file.name,
            "file_size": file_size,
            "collection_name": collection_name,  # Store collection name with each document
        }
        
        # Add to collection
        collection.add_documents(chunks, metadata=metadata)
        
        # Update vectorstore reference
        st.session_state.vectorstore = collection.vectorstore
        
        # Reinitialize RAG chain
        st.session_state.rag_chain = RAGChain(st.session_state.vectorstore)
        
        # Cleanup
        os.unlink(tmp_path)
        
        return True, len(chunks)
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return False, str(e)


# UI Layout with tabs
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1.5rem 0; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 16px; margin-bottom: 2rem;">
    <h1 style="margin-bottom: 0.5rem; font-size: 2.8rem;">📚 Smart RAG Document Assistant</h1>
    <p style="font-size: 1.15rem; color: #6b7280; margin: 0; font-weight: 500;">AI-Powered Document Intelligence • Ask Questions, Get Instant Answers</p>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💬 Chat", 
    "📊 Analytics", 
    "📁 Collections", 
    "📄 Documents", 
    "💾 Export/Import"
])

# Sidebar for document upload
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; margin-bottom: 1.5rem; border-bottom: 2px solid #e5e7eb;">
        <h2 style="margin: 0; color: #1f2937;">📄 Document Upload</h2>
        <p style="margin: 0.5rem 0 0 0; color: #6b7280; font-size: 0.9rem;">Manage your documents and collections</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Collection selector
    st.markdown("### 📁 Collection Management")
    st.markdown("""
    <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem;">
        <p style="margin: 0; font-size: 0.85rem; color: #92400e;"><strong>⚠️ Important:</strong> Queries search ONLY the selected collection!</p>
    </div>
    """, unsafe_allow_html=True)
    
    collection_options = st.session_state.collections_list
    selected_collection = st.selectbox(
        "Active Collection (for queries)",
        options=collection_options,
        index=0 if st.session_state.current_collection in collection_options else 0,
        help="⚠️ This is the collection that will be searched when you ask questions. Switch collections to query different document sets.",
        key="collection_selector_sidebar"
    )
    if selected_collection != st.session_state.current_collection:
        st.session_state.current_collection = selected_collection
        st.session_state.collection_manager.switch_collection(selected_collection)
        collection = st.session_state.collection_manager.get_current_collection()
        st.session_state.vectorstore = collection.vectorstore
        st.session_state.rag_chain = None  # Reset RAG chain
        st.success(f"✅ Switched to collection: **{selected_collection}**")
        st.info(f"💡 All new queries will search the **{selected_collection}** collection")
        st.rerun()
    
    # Create new collection
    with st.expander("➕ Create New Collection"):
        new_collection_name = st.text_input(
            "Collection Name", 
            placeholder="e.g., research-papers (min 3 chars, a-z, 0-9, ., _, -)",
            help="Collection name must be 3-512 characters, start/end with letter/number, and contain only: a-z, A-Z, 0-9, ., _, -"
        )
        if st.button("Create Collection", key="create_collection_sidebar"):
            if new_collection_name:
                # Validate collection name
                from features.multi_document import DocumentCollection
                is_valid, error_msg = DocumentCollection.validate_collection_name(new_collection_name)
                
                if not is_valid:
                    st.error(f"❌ Invalid collection name: {error_msg}")
                elif new_collection_name in st.session_state.collections_list:
                    st.error("❌ Collection already exists!")
                else:
                    try:
                        st.session_state.collection_manager.create_collection(new_collection_name)
                        st.session_state.collections_list.append(new_collection_name)
                        st.session_state.current_collection = new_collection_name
                        st.success(f"✅ Created collection: {new_collection_name}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error creating collection: {str(e)}")
            else:
                st.warning("⚠️ Please enter a collection name")
    
    # Collection info with better styling
    if st.session_state.current_collection:
        collection = st.session_state.collection_manager.get_current_collection()
        info = collection.get_collection_info()
        doc_count = info.get('document_count', 0)
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e0e7ff 0%, #f3e8ff 100%); padding: 1rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #667eea;">
            <p style="margin: 0; font-size: 0.9rem; color: #1f2937;"><strong>📊 Current Collection Stats</strong></p>
            <p style="margin: 0.5rem 0 0 0; font-size: 1.5rem; font-weight: 700; color: #667eea;">{}</p>
            <p style="margin: 0; font-size: 0.85rem; color: #6b7280;">Documents indexed</p>
        </div>
        """.format(doc_count), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Batch upload option
    upload_mode = st.radio(
        "Upload Mode",
        ["Single File", "Batch Upload"],
        help="Process one file or multiple files at once"
    )
    
    if upload_mode == "Single File":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "txt", "md"],
            help="Supported formats: PDF, DOCX, TXT, MD",
        )
        
        if uploaded_file is not None:
            st.info(f"📎 **File ready:** {uploaded_file.name}")
            if st.button("🔄 Process & Index Document", type="primary", use_container_width=True):
                with st.spinner("Processing document..."):
                    success, result = process_uploaded_file(uploaded_file, st.session_state.current_collection)
                    if success:
                        st.success(f"✅ Successfully processed! Created {result} chunks.")
                        st.balloons()
                        collection = st.session_state.collection_manager.get_current_collection()
                        info = collection.get_collection_info()
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {result}")
            else:
                st.caption("👆 Click the button above to process the document")
    else:
        uploaded_files = st.file_uploader(
            "Choose multiple files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Upload multiple files to process at once",
        )
        
        if uploaded_files:
            st.info(f"📎 **{len(uploaded_files)} file(s) ready**")
            if st.button("🔄 Process All Documents", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                results = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {i+1}/{len(uploaded_files)}: {file.name}")
                    success, result = process_uploaded_file(file, st.session_state.current_collection)
                    results.append((file.name, success, result))
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Show results
                successful = sum(1 for _, s, _ in results if s)
                st.success(f"✅ Processed {successful}/{len(uploaded_files)} files successfully!")
                if successful < len(uploaded_files):
                    st.warning(f"⚠️ {len(uploaded_files) - successful} file(s) failed")
                
                # Show details
                with st.expander("📋 View Processing Details"):
                    for name, success, result in results:
                        if success:
                            st.success(f"✅ {name}: {result} chunks")
                        else:
                            st.error(f"❌ {name}: {result}")
                
                st.rerun()
    
    st.markdown("---")
    st.markdown("### 👤 Session Info")
    st.markdown(f"""
    <div style="background: #f9fafb; padding: 0.75rem; border-radius: 8px; margin-bottom: 1rem; border: 1px solid #e5e7eb;">
        <p style="margin: 0; font-size: 0.85rem; color: #6b7280;"><strong>User ID:</strong> <code style="background: #e5e7eb; padding: 0.2rem 0.4rem; border-radius: 4px;">{user_id[:8]}...</code></p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; color: #9ca3af;">💡 Each user has isolated chat history and metrics</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Settings")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); padding: 1rem; border-radius: 10px; border-left: 4px solid #0ea5e9;">
        <p style="margin: 0.5rem 0;"><strong>Chunk Size:</strong> <span style="color: #0ea5e9; font-weight: 600;">{settings.chunk_size}</span></p>
        <p style="margin: 0.5rem 0;"><strong>Embeddings:</strong> <span style="color: #0ea5e9; font-weight: 600;">{settings.embeddings_model}</span></p>
        <p style="margin: 0.5rem 0;"><strong>Top K Retrieval:</strong> <span style="color: #0ea5e9; font-weight: 600;">{settings.top_k_retrieval}</span></p>
    </div>
    """, unsafe_allow_html=True)

# Chat input - MUST be outside tabs (Streamlit requirement)
if prompt := st.chat_input("Ask a question about the documents..."):
    # Check if vectorstore is initialized
    if st.session_state.vectorstore is None:
        st.warning("⚠️ **Please upload and process a document first!**")
        st.info("💡 **Steps:** 1) Upload a file in the sidebar → 2) Click 'Process & Index Document' → 3) Then ask questions here!")
    else:
        # Initialize RAG chain if not already done
        if st.session_state.rag_chain is None:
            collection = st.session_state.collection_manager.get_current_collection()
            st.session_state.rag_chain = RAGChain(collection.vectorstore)
        
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Get response with timing
        import time
        start_time = time.time()
        
        try:
            response = st.session_state.rag_chain.query(prompt)
            response_time = time.time() - start_time
            
            # Extract answer and sources
            answer = response.get("answer", "An answer could not be generated.")
            sources = response.get("source_documents", [])
            
            # Get current collection name for tracking and add to source metadata
            current_collection_name = st.session_state.current_collection
            
            # Ensure all sources have collection_name in metadata
            for source in sources:
                if "metadata" not in source:
                    source["metadata"] = {}
                # Add collection name if missing (for backward compatibility)
                if "collection_name" not in source["metadata"]:
                    source["metadata"]["collection_name"] = current_collection_name
            
            # Add assistant message to history with collection info
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "response_time": response_time,
                "collection_name": current_collection_name,  # Track which collection was queried
            })
            
            # Auto-evaluate the response
            try:
                collection = st.session_state.collection_manager.get_current_collection()
                info = collection.get_collection_info()
                retrieved_docs = len(sources)
                
                # Simple auto-evaluation (can be improved)
                relevance_score = min(0.95, 0.7 + (retrieved_docs / max(1, info.get('document_count', 1))) * 0.25)
                quality_score = min(0.95, 0.75 + (len(answer) / 500) * 0.2) if len(answer) > 50 else 0.7
                
                st.session_state.evaluator.evaluate(
                    question=prompt,
                    expected_answer="",  # Auto-evaluation
                    actual_answer=answer,
                    retrieved_docs=retrieved_docs,
                    relevance_score=relevance_score,
                    answer_quality=quality_score,
                    response_time=response_time
                )
                
                # Save evaluator state
                if EVALS_PATH:
                    try:
                        st.session_state.evaluator.save_to_disk(str(EVALS_PATH))
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Error in auto-evaluation: {e}")
            
            # Save chat history
            if CHAT_PATH:
                try:
                    import json as _json
                    CHAT_PATH.write_text(_json.dumps(st.session_state.chat_history, indent=2), encoding="utf-8")
                except Exception:
                    pass
            
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error generating response: {str(e)}")
            logger.error(f"RAG query error: {e}")

# Tab 1: Chat Interface
with tab1:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="margin-bottom: 1rem;">💬 Chat with Documents</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display current collection info prominently
    collection = st.session_state.collection_manager.get_current_collection()
    collection_info = collection.get_collection_info()
    doc_count = collection_info.get("document_count", 0)
    
    if doc_count > 0:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%); padding: 1.25rem; border-radius: 12px; margin-bottom: 1.5rem; border-left: 5px solid #3b82f6; box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15);">
            <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;">
                <div>
                    <p style="margin: 0; font-size: 0.9rem; color: #1e40af; font-weight: 600;">📁 ACTIVE COLLECTION</p>
                    <p style="margin: 0.25rem 0 0 0; font-size: 1.3rem; font-weight: 700; color: #1e3a8a;">{st.session_state.current_collection}</p>
                </div>
                <div style="flex: 1; min-width: 200px;">
                    <p style="margin: 0; font-size: 0.9rem; color: #1e40af; font-weight: 600;">📄 DOCUMENTS INDEXED</p>
                    <p style="margin: 0.25rem 0 0 0; font-size: 1.3rem; font-weight: 700; color: #1e3a8a;">{doc_count}</p>
                </div>
                <div>
                    <p style="margin: 0; font-size: 0.85rem; color: #3b82f6;">💡 All queries will search this collection</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1.25rem; border-radius: 12px; margin-bottom: 1.5rem; border-left: 5px solid #f59e0b; box-shadow: 0 2px 8px rgba(245, 158, 11, 0.15);">
            <p style="margin: 0; font-size: 1rem; color: #92400e; font-weight: 600;">⚠️ <strong>Active Collection:</strong> `{st.session_state.current_collection}` | 📄 <strong>No documents</strong></p>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem; color: #78350f;">Please upload and process documents first in the sidebar!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                # Show collection info if available
                collection_used = message.get("collection_name", "Unknown")
                with st.expander(f"📚 View Sources (from collection: {collection_used})"):
                    if not message["sources"]:
                        st.info("No sources retrieved for this response.")
                    else:
                        for i, source in enumerate(message["sources"], 1):
                            source_metadata = source.get("metadata", {})
                            source_file = source_metadata.get("source_file", "Unknown")
                            source_collection = source_metadata.get("collection_name", collection_used)
                            chunk_id = source_metadata.get("chunk_id", "N/A")
                            
                            # Display source info with better styling
                            st.markdown(f"""
                            <div style="background: #f9fafb; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #667eea;">
                                <h4 style="margin: 0 0 0.75rem 0; color: #1f2937;">Source {i}</h4>
                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.75rem; margin-bottom: 0.75rem;">
                                    <div>
                                        <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">📁 Collection</p>
                                        <p style="margin: 0.25rem 0 0 0; font-weight: 600; color: #1f2937;"><code style="background: #e5e7eb; padding: 0.2rem 0.5rem; border-radius: 4px;">{source_collection}</code></p>
                                    </div>
                                    <div>
                                        <p style="margin: 0; font-size: 0.8rem; color: #6b7280;">📄 Document</p>
                                        <p style="margin: 0.25rem 0 0 0; font-weight: 600; color: #1f2937;"><code style="background: #e5e7eb; padding: 0.2rem 0.5rem; border-radius: 4px;">{source_file}</code></p>
                                    </div>
                                    {f'<div><p style="margin: 0; font-size: 0.8rem; color: #6b7280;">📍 Chunk ID</p><p style="margin: 0.25rem 0 0 0; font-weight: 600; color: #1f2937;">{chunk_id}</p></div>' if chunk_id != "N/A" else ''}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown("**📝 Content Preview:**")
                            st.markdown(f"""
                            <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb; font-family: 'Courier New', monospace; font-size: 0.9rem; line-height: 1.6; color: #374151;">
                                {source["content"][:500] + ("..." if len(source["content"]) > 500 else "")}
                            </div>
                            """, unsafe_allow_html=True)
                            with st.expander("🔍 View Full Metadata"):
                                st.json(source_metadata)
                            if i < len(message["sources"]):
                                st.markdown("<hr style='margin: 1.5rem 0; border: none; border-top: 2px solid #e5e7eb;'>", unsafe_allow_html=True)
    
    # Show message if no documents uploaded
    if not st.session_state.chat_history:
        st.info("👆 **Use the chat input at the bottom of the page to ask questions!**")
        
        # Show helpful info about collections
        if collection_info.get("document_count", 0) == 0:
            st.info("""
            **💡 How Collections Work:**
            1. **Create/Select Collection** in the sidebar
            2. **Upload & Process** documents (they'll be added to the selected collection)
            3. **Ask Questions** - RAG will search only the active collection
            4. **Switch Collections** to query different document sets
            """)

# Tab 2: Analytics Dashboard
with tab2:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="margin-bottom: 0.5rem;">📊 Analytics Dashboard</h2>
        <p style="color: #6b7280; margin: 0;">Track performance metrics and system insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metric view selector with better styling
    st.markdown("### 📈 Metrics View")
    metric_view = st.radio(
        "Select Metrics View",
        ["My Metrics", "Overall Metrics"],
        horizontal=True,
        help="View your personal metrics or aggregate metrics across all users",
        label_visibility="collapsed"
    )
    st.markdown("---")
    
    collection = st.session_state.collection_manager.get_current_collection()
    collection_info = collection.get_collection_info()
    
    # Display metrics based on selection
    if metric_view == "My Metrics":
        evaluator = st.session_state.evaluator
        eval_summary = evaluator.get_summary()
        st.caption(f"👤 User ID: {user_id[:8]}...")
        
        # Overview Metrics with enhanced styling
        st.markdown("### 📊 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        doc_count = collection_info.get("document_count", 0)
        query_count = eval_summary.get("total_queries", 0)
        avg_rel = eval_summary.get("avg_relevance", 0.0)
        avg_qual = eval_summary.get("avg_quality", 0.0)
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid #93c5fd; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.1);">
                <p style="margin: 0; font-size: 0.85rem; color: #1e40af; font-weight: 600;">📚 Total Documents</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700; color: #1e3a8a;">{doc_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #e0e7ff 0%, #c7d2fe 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid #a5b4fc; box-shadow: 0 4px 6px rgba(99, 102, 241, 0.1);">
                <p style="margin: 0; font-size: 0.85rem; color: #4338ca; font-weight: 600;">💬 Total Queries</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700; color: #3730a3;">{query_count}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid #fcd34d; box-shadow: 0 4px 6px rgba(245, 158, 11, 0.1);">
                <p style="margin: 0; font-size: 0.85rem; color: #92400e; font-weight: 600;">⭐ Avg Relevance</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700; color: #78350f;">{avg_rel:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); padding: 1.5rem; border-radius: 12px; text-align: center; border: 1px solid #6ee7b7; box-shadow: 0 4px 6px rgba(16, 185, 129, 0.1);">
                <p style="margin: 0; font-size: 0.85rem; color: #065f46; font-weight: 600;">🎯 Avg Quality</p>
                <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700; color: #047857;">{avg_qual:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Usage Statistics
        st.subheader("📈 Usage Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Collection Stats")
            st.info(
                f"**Collection Name:** {collection_info.get('name', 'N/A')}\n\n"
                f"**Document Count:** {collection_info.get('document_count', 0)}\n\n"
                f"**Storage:** {collection_info.get('persist_directory', 'N/A')}"
            )
        
        with col2:
            st.markdown("### Query Performance")
            if eval_summary.get("total_queries", 0) > 0:
                avg_rt = eval_summary.get("avg_response_time", 0.0)
                rt_text = f"{avg_rt:.2f}s" if avg_rt > 0 else "N/A"
                st.info(
                    f"**Total Queries:** {eval_summary.get('total_queries', 0)}\n\n"
                    f"**Average Relevance:** {eval_summary.get('avg_relevance', 0.0):.1%}\n\n"
                    f"**Average Quality:** {eval_summary.get('avg_quality', 0.0):.1%}\n\n"
                    f"**Avg Response Time:** {rt_text}"
                )
            else:
                st.info("No queries yet. Start chatting to see statistics!")
        
        st.markdown("---")
        
        # Metrics Summary Section
        st.subheader("📄 Metrics Summary")
        metrics_data = {
            "Documents Processed": collection_info.get("document_count", 0),
            "Total Queries": eval_summary.get("total_queries", 0),
            "Avg Relevance": f"{eval_summary.get('avg_relevance', 0.0):.1%}",
            "Avg Quality": f"{eval_summary.get('avg_quality', 0.0):.1%}",
            "Avg Response Time": f"{eval_summary.get('avg_response_time', 0.0):.2f}s" if eval_summary.get('avg_response_time', 0.0) > 0 else "N/A",
            "Collections": len(st.session_state.collections_list),
        }
        
        # Display metrics in a nice format
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Key Metrics:**")
            for key, value in list(metrics_data.items())[:3]:
                st.write(f"- **{key}:** {value}")
        with col2:
            st.markdown("**Additional Metrics:**")
            for key, value in list(metrics_data.items())[3:]:
                st.write(f"- **{key}:** {value}")
        
        # Download Metrics
        if collection_info.get("document_count", 0) > 0 and eval_summary.get("total_queries", 0) > 0:
            st.markdown("---")
            import json
            metrics_export = {
                "metrics": metrics_data,
                "user_id": user_id,
                "generated_date": str(__import__("datetime").datetime.now()),
            }
            metrics_json = json.dumps(metrics_export, indent=2)
            st.download_button(
                "📥 Download My Metrics (JSON)",
                metrics_json,
                file_name=f"my_metrics_{__import__('datetime').datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("📊 Process some documents and ask questions to generate metrics!")
        
        st.markdown("---")
        
        # Query History (Last 10) - User-specific
        if evaluator.results:
            st.subheader("📋 My Recent Query History")
            recent_results = evaluator.results[-10:]
            
            for i, result in enumerate(reversed(recent_results), 1):
                with st.expander(f"Query {i}: {result.question[:60]}..."):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Relevance", f"{result.relevance_score:.1%}")
                    with col2:
                        st.metric("Quality", f"{result.answer_quality:.1%}")
                    with col3:
                        st.metric("Docs Retrieved", result.retrieved_docs)
                    with col4:
                        if hasattr(result, 'response_time') and result.response_time > 0:
                            st.metric("Response Time", f"{result.response_time:.2f}s")
                    st.caption(f"**Answer:** {result.actual_answer[:200]}...")
    
    else:  # Overall Metrics
        if DATA_DIR:
            overall_metrics = get_overall_metrics(DATA_DIR)
            
            st.caption("📊 Aggregate metrics across all users")
            
            # Overview Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("👥 Total Users", overall_metrics.get("total_users", 0))
            
            with col2:
                st.metric("💬 Total Queries", overall_metrics.get("total_queries", 0))
            
            with col3:
                avg_rel = overall_metrics.get("avg_relevance", 0.0)
                st.metric("⭐ Avg Relevance", f"{avg_rel:.1%}")
            
            with col4:
                avg_qual = overall_metrics.get("avg_quality", 0.0)
                st.metric("🎯 Avg Quality", f"{avg_qual:.1%}")
            
            st.markdown("---")
            
            # Overall Statistics
            st.subheader("📈 Overall System Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### System Performance")
                if overall_metrics.get("total_queries", 0) > 0:
                    avg_rt = overall_metrics.get("avg_response_time", 0.0)
                    rt_text = f"{avg_rt:.2f}s" if avg_rt > 0 else "N/A"
                    st.info(
                        f"**Total Active Users:** {overall_metrics.get('total_users', 0)}\n\n"
                        f"**Total Queries (All Users):** {overall_metrics.get('total_queries', 0)}\n\n"
                        f"**Average Relevance:** {overall_metrics.get('avg_relevance', 0.0):.1%}\n\n"
                        f"**Average Quality:** {overall_metrics.get('avg_quality', 0.0):.1%}\n\n"
                        f"**Avg Response Time:** {rt_text}"
                    )
                else:
                    st.info("No system-wide queries yet.")
            
            with col2:
                st.markdown("### Evaluation Metrics")
                if overall_metrics.get("total_queries", 0) > 0:
                    st.info(
                        f"**Precision:** {overall_metrics.get('precision', 0.0):.1%}\n\n"
                        f"**Recall:** {overall_metrics.get('recall', 0.0):.1%}\n\n"
                        f"**F1 Score:** {(2 * overall_metrics.get('precision', 0.0) * overall_metrics.get('recall', 0.0) / (overall_metrics.get('precision', 0.0) + overall_metrics.get('recall', 0.0))):.1%}" if (overall_metrics.get('precision', 0.0) + overall_metrics.get('recall', 0.0)) > 0 else "N/A"
                    )
                else:
                    st.info("No evaluation data available.")
            
            st.markdown("---")
            
            # Overall Metrics Summary
            st.subheader("📄 Overall Metrics Summary")
            overall_metrics_data = {
                "Total Users": overall_metrics.get("total_users", 0),
                "Total Queries": overall_metrics.get("total_queries", 0),
                "Avg Relevance": f"{overall_metrics.get('avg_relevance', 0.0):.1%}",
                "Avg Quality": f"{overall_metrics.get('avg_quality', 0.0):.1%}",
                "Avg Response Time": f"{overall_metrics.get('avg_response_time', 0.0):.2f}s" if overall_metrics.get('avg_response_time', 0.0) > 0 else "N/A",
                "Precision": f"{overall_metrics.get('precision', 0.0):.1%}",
                "Recall": f"{overall_metrics.get('recall', 0.0):.1%}",
            }
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Key Metrics:**")
                for key, value in list(overall_metrics_data.items())[:3]:
                    st.write(f"- **{key}:** {value}")
            with col2:
                st.markdown("**Additional Metrics:**")
                for key, value in list(overall_metrics_data.items())[3:]:
                    st.write(f"- **{key}:** {value}")
            
            # Download Overall Metrics
            if overall_metrics.get("total_queries", 0) > 0:
                st.markdown("---")
                import json
                overall_metrics_export = {
                    "metrics": overall_metrics_data,
                    "aggregate_data": overall_metrics,
                    "generated_date": str(__import__("datetime").datetime.now()),
                }
                overall_metrics_json = json.dumps(overall_metrics_export, indent=2)
                st.download_button(
                    "📥 Download Overall Metrics (JSON)",
                    overall_metrics_json,
                    file_name=f"overall_metrics_{__import__('datetime').datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.warning("⚠️ Overall metrics are not available in cloud deployment (file system access required).")

# Tab 3: Collections Management
with tab3:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="margin-bottom: 0.5rem;">📁 Document Collections</h2>
        <p style="color: #6b7280; margin: 0;">Organize and manage your document collections</p>
    </div>
    """, unsafe_allow_html=True)
    
    manager = st.session_state.collection_manager
    
    # Current Collection Info
    st.subheader("Current Collection")
    current = manager.get_current_collection()
    info = current.get_collection_info()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Collection Name", info.get("name", "N/A"))
    with col2:
        st.metric("Document Count", info.get("document_count", 0))
    with col3:
        st.metric("Total Collections", len(st.session_state.collections_list))
    
    st.markdown("---")
    
    # All Collections
    st.subheader("All Collections")
    
    for col_name in st.session_state.collections_list:
        with st.expander(f"📁 {col_name}", expanded=(col_name == st.session_state.current_collection)):
            collection = manager.get_collection(col_name)
            col_info = collection.get_collection_info()
            
            st.write(f"**Documents:** {col_info.get('document_count', 0)}")
            st.write(f"**Directory:** {col_info.get('persist_directory', 'N/A')}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(f"Switch to {col_name}", key=f"switch_{col_name}"):
                    st.session_state.current_collection = col_name
                    manager.switch_collection(col_name)
                    collection = manager.get_current_collection()
                    st.session_state.vectorstore = collection.vectorstore
                    st.session_state.rag_chain = None
                    st.success(f"✅ Switched to {col_name}")
                    st.rerun()
            
            with col2:
                if col_name != "default":
                    if st.button(f"🗑️ Delete Collection", key=f"delete_{col_name}", type="secondary"):
                        try:
                            # Use the manager's delete_collection method
                            manager.delete_collection(col_name)
                            if col_name in st.session_state.collections_list:
                                st.session_state.collections_list.remove(col_name)
                            st.success(f"✅ Successfully deleted collection: {col_name}")
                            st.rerun()
                        except ValueError as e:
                            st.error(f"❌ {str(e)}")
                        except Exception as e:
                            st.error(f"❌ Error deleting collection: {str(e)}")
    
    st.markdown("---")
    
    # Create New Collection
    st.subheader("Create New Collection")
    new_col = st.text_input(
        "Collection Name", 
        placeholder="e.g., research-papers (min 3 chars)",
        help="Must be 3-512 characters, start/end with letter/number, and contain only: a-z, A-Z, 0-9, ., _, -"
    )
    if st.button("Create Collection", key="create_collection_tab"):
        if new_col:
            # Validate collection name
            from features.multi_document import DocumentCollection
            is_valid, error_msg = DocumentCollection.validate_collection_name(new_col)
            
            if not is_valid:
                st.error(f"❌ Invalid collection name: {error_msg}")
            elif new_col in st.session_state.collections_list:
                st.error("❌ Collection already exists!")
            else:
                try:
                    manager.create_collection(new_col)
                    st.session_state.collections_list.append(new_col)
                    st.success(f"✅ Created collection: {new_col}")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error creating collection: {str(e)}")
        else:
            st.warning("⚠️ Please enter a collection name")

# Tab 4: Document Management
with tab4:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="margin-bottom: 0.5rem;">📄 Document Management</h2>
        <p style="color: #6b7280; margin: 0;">View, manage, and delete indexed documents</p>
    </div>
    """, unsafe_allow_html=True)
    
    collection = st.session_state.collection_manager.get_current_collection()
    collection_info = collection.get_collection_info()
    
    # Document Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", collection_info.get("document_count", 0))
    with col2:
        st.metric("Collection", collection_info.get("name", "N/A"))
    with col3:
        st.metric("Storage", "ChromaDB")
    
    st.markdown("---")
    
    # Document List
    st.subheader("📋 Document List")
    
    try:
        # Get documents grouped by source file using the new method
        source_files = collection.get_documents_by_source()
        
        if source_files:
            st.info(f"📊 Found {len(source_files)} unique document(s) in this collection")
            
            # Display documents
            for source_file, info in source_files.items():
                with st.expander(f"📄 {source_file} ({info['count']} chunks)", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Chunks:** {info['count']}")
                        if info.get('metadata'):
                            file_size = info['metadata'].get('file_size', 'N/A')
                            if file_size != 'N/A':
                                file_size_str = f"{file_size:,} bytes" if isinstance(file_size, int) else str(file_size)
                                st.write(f"**File Size:** {file_size_str}")
                            st.json(info['metadata'])
                    with col2:
                        delete_key = f"delete_doc_{source_file}_{hash(source_file)}"
                        if st.button(
                            "🗑️ Delete Document", 
                            key=delete_key, 
                            help="Delete all chunks from this document",
                            type="secondary"
                        ):
                            try:
                                deleted_count = collection.delete_documents_by_source(source_file)
                                st.success(f"✅ Successfully deleted {deleted_count} chunks from {source_file}")
                                # Reset RAG chain if current collection
                                if st.session_state.current_collection == collection.collection_name:
                                    st.session_state.rag_chain = None
                                    st.session_state.vectorstore = collection.vectorstore
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error deleting document: {str(e)}")
        else:
            st.info("📭 No documents in this collection. Upload documents in the sidebar!")
            
    except Exception as e:
        st.error(f"❌ Error loading documents: {str(e)}")
        st.info("💡 Try uploading a document first!")

# Tab 5: Export/Import
with tab5:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h2 style="margin-bottom: 0.5rem;">💾 Export & Import</h2>
        <p style="color: #6b7280; margin: 0;">Export your data or import from previous sessions</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Export Section
    with col1:
        st.subheader("📥 Export Data")
        
        # Export Chat History
        st.markdown("### 💬 Chat History")
        if st.session_state.chat_history:
            chat_json = {
                "chat_history": st.session_state.chat_history,
                "export_date": str(__import__("datetime").datetime.now()),
            }
            import json
            chat_json_str = json.dumps(chat_json, indent=2)
            st.download_button(
                "📥 Download Chat History (JSON)",
                chat_json_str,
                file_name=f"chat_history_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            st.caption(f"💬 {len(st.session_state.chat_history)} messages")
        else:
            st.info("No chat history to export")
        
        st.markdown("---")
        
        # Export Evaluation Results
        st.markdown("### 📊 Evaluation Results")
        evaluator = st.session_state.evaluator
        if evaluator.results:
            import tempfile
            import json
            eval_data = {
                "metrics": evaluator.metrics,
                "results": [
                    {
                        "question": r.question,
                        "expected_answer": r.expected_answer,
                        "actual_answer": r.actual_answer,
                        "retrieved_docs": r.retrieved_docs,
                        "relevance_score": r.relevance_score,
                        "answer_quality": r.answer_quality,
                        "timestamp": r.timestamp,
                    }
                    for r in evaluator.results
                ],
                "export_date": str(__import__("datetime").datetime.now()),
            }
            eval_json_str = json.dumps(eval_data, indent=2)
            st.download_button(
                "📥 Download Evaluation Results (JSON)",
                eval_json_str,
                file_name=f"evaluation_results_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            st.caption(f"📊 {len(evaluator.results)} evaluation results")
        else:
            st.info("No evaluation results to export")
        
        st.markdown("---")
        
        # Export Collection Info
        st.markdown("### 📁 Collection Info")
        collection = st.session_state.collection_manager.get_current_collection()
        collection_info = collection.get_collection_info()
        collection_data = {
            "collection_name": collection_info.get("name", "N/A"),
            "document_count": collection_info.get("document_count", 0),
            "collections_list": st.session_state.collections_list,
            "export_date": str(__import__("datetime").datetime.now()),
        }
        import json
        collection_json_str = json.dumps(collection_data, indent=2)
        st.download_button(
            "📥 Download Collection Info (JSON)",
            collection_json_str,
            file_name=f"collection_info_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Import Section
    with col2:
        st.subheader("📤 Import Data")
        
        # Import Chat History
        st.markdown("### 💬 Chat History")
        uploaded_chat = st.file_uploader(
            "Upload Chat History (JSON)",
            type=["json"],
            key="import_chat"
        )
        if uploaded_chat:
            try:
                import json
                chat_data = json.load(uploaded_chat)
                if "chat_history" in chat_data:
                    st.session_state.chat_history = chat_data["chat_history"]
                    st.success(f"✅ Imported {len(chat_data['chat_history'])} messages!")
                    st.rerun()
                else:
                    st.error("❌ Invalid chat history format")
            except Exception as e:
                st.error(f"❌ Error importing chat history: {str(e)}")
        
        st.markdown("---")
        
        # Import Evaluation Results
        st.markdown("### 📊 Evaluation Results")
        uploaded_eval = st.file_uploader(
            "Upload Evaluation Results (JSON)",
            type=["json"],
            key="import_eval"
        )
        if uploaded_eval:
            try:
                import json
                eval_data = json.load(uploaded_eval)
                if "results" in eval_data:
                    # Add results to evaluator
                    for result_data in eval_data["results"]:
                        evaluator.evaluate(
                            question=result_data.get("question", ""),
                            expected_answer=result_data.get("expected_answer", ""),
                            actual_answer=result_data.get("actual_answer", ""),
                            retrieved_docs=result_data.get("retrieved_docs", 0),
                            relevance_score=result_data.get("relevance_score", 0.0),
                            answer_quality=result_data.get("answer_quality", 0.0),
                        )
                    st.success(f"✅ Imported {len(eval_data['results'])} evaluation results!")
                    st.rerun()
                else:
                    st.error("❌ Invalid evaluation results format")
            except Exception as e:
                st.error(f"❌ Error importing evaluation results: {str(e)}")
        
        st.info("💡 **Note:** Collection data is stored in ChromaDB and cannot be imported via JSON. Use the Collections tab to manage collections.")

