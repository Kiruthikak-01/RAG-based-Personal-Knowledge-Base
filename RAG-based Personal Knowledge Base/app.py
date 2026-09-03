"""
Streamlit application for the RAG-based Personal Knowledge Base.

Features:
- File upload for PDFs and text/Markdown documents
- Document ingestion and embedding
- Chat interface with streaming responses
- Source citations displayed with clickable links to documents
- Chat history persistence within session
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import os

# Import local modules
from config import load_config
from ingestion import ingest_document, Chunk
from embeddings import load_embedding_model, embed_query
from vectorstore import initialize_vectorstore, add_chunks_to_store, similarity_search
from retrieval import retrieve_with_enhancements
from generation import generate_with_citations, generate_with_citations_streaming

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Streamlit page configuration
st.set_page_config(
    page_title="RAG-based Personal Knowledge Base",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📚 Chat with My Notes")
st.markdown("Upload documents, chat with them, and get AI-powered answers with source citations.")


# ============================================================================
# Session State Initialization
# ============================================================================

if "config" not in st.session_state:
    st.session_state.config = load_config()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore_initialized" not in st.session_state:
    st.session_state.vectorstore_initialized = False

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


# ============================================================================
# Sidebar: Configuration & Document Upload
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Chunking strategy
    chunking_strategy = st.selectbox(
        "Chunking Strategy",
        options=["fixed", "recursive"],
        index=0 if st.session_state.config.chunking.strategy == "fixed" else 1
    )
    st.session_state.config.chunking.strategy = chunking_strategy
    
    chunk_size = st.slider(
        "Chunk Size (characters)",
        min_value=100,
        max_value=2000,
        value=st.session_state.config.chunking.chunk_size,
        step=50
    )
    st.session_state.config.chunking.chunk_size = chunk_size
    
    overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=500,
        value=st.session_state.config.chunking.overlap,
        step=10
    )
    st.session_state.config.chunking.overlap = overlap
    
    st.divider()
    
    # Retrieval configuration
    st.subheader("Retrieval Settings")
    top_k = st.slider(
        "Top-K Results",
        min_value=1,
        max_value=20,
        value=st.session_state.config.retrieval.top_k
    )
    st.session_state.config.retrieval.top_k = top_k
    
    use_reranking = st.checkbox(
        "Use Cross-Encoder Re-ranking",
        value=st.session_state.config.retrieval.use_reranking
    )
    st.session_state.config.retrieval.use_reranking = use_reranking
    
    use_hybrid = st.checkbox(
        "Use Hybrid Search (BM25 + Dense)",
        value=st.session_state.config.retrieval.use_hybrid_search
    )
    st.session_state.config.retrieval.use_hybrid_search = use_hybrid
    
    use_hyde = st.checkbox(
        "Use HyDE Query Transformation",
        value=st.session_state.config.retrieval.use_hyde
    )
    st.session_state.config.retrieval.use_hyde = use_hyde
    
    st.divider()
    
    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.uploaded_documents:
                    # TODO: Save uploaded file to temporary location
                    # TODO: Call ingest_document()
                    # TODO: Embed chunks via embeddings.embed_texts()
                    # TODO: Add to vectorstore via add_chunks_to_store()
                    
                    # Placeholder
                    st.info(f"Document processing stub: {uploaded_file.name}")
                    st.session_state.uploaded_documents.append(uploaded_file.name)
    
    if st.session_state.uploaded_documents:
        st.success(f"Loaded {len(st.session_state.uploaded_documents)} document(s)")
        with st.expander("Uploaded Documents"):
            for doc in st.session_state.uploaded_documents:
                st.text(doc)


# ============================================================================
# Main Chat Interface
# ============================================================================

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If this is an assistant message, show citations
        if message["role"] == "assistant" and "citations" in message:
            with st.expander("📌 Sources"):
                for i, citation in enumerate(message["citations"], 1):
                    st.write(f"""
                    **[{i}]** `{citation.get('source_file', 'Unknown')}`
                    - Page: {citation.get('page_number', 'N/A')}
                    - Chunk ID: {citation.get('chunk_id', 'N/A')}
                    """)


# User input
if user_query := st.chat_input("Ask me something about your documents..."):
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Generate response
    with st.chat_message("assistant"):
        # TODO: Initialize vectorstore if needed
        # TODO: Embed query via embeddings.embed_query()
        # TODO: Apply retrieval enhancements via retrieval.retrieve_with_enhancements()
        # TODO: Generate answer via generation.generate_with_citations_streaming()
        # TODO: Display streaming response
        
        # Placeholder
        response_placeholder = st.empty()
        response_placeholder.markdown("*Generating response...*")
        
        # After implementation:
        # with response_placeholder.container():
        #     full_response = ""
        #     for token in generate_with_citations_streaming(...):
        #         full_response += token
        #         st.write(full_response)


# ============================================================================
# Footer
# ============================================================================

st.divider()
st.markdown("""
**How to use:**
1. Upload PDF or text documents in the sidebar
2. Ask questions about the content
3. Answers will include clickable source citations
4. Adjust retrieval and chunking settings as needed

**Limitations:**
- This is a learning project with TODO implementations
- Some features (streaming, re-ranking, HyDE, etc.) need to be filled in
""")
