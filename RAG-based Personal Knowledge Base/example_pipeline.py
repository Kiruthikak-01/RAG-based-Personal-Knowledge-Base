#!/usr/bin/env python3
"""
Complete Example: Ingest documents and query the RAG system
Run this script to see the full pipeline in action.
"""

import sys
from pathlib import Path

# Add project directory to path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

import logging
from embeddings import load_embedding_model, embed_texts, embed_query
from ingestion import ingest_document
from vectorstore import initialize_vectorstore, add_chunks_to_store, get_collection_stats
from retrieval import retrieve_with_enhancements
from generation import generate_with_citations
from config import load_config

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_full_pipeline():
    """Complete example: ingest → embed → retrieve → generate"""
    
    logger.info("=" * 70)
    logger.info("RAG-BASED PERSONAL KNOWLEDGE BASE - FULL PIPELINE EXAMPLE")
    logger.info("=" * 70)
    
    # Step 1: Load configuration
    logger.info("\n📋 Step 1: Loading Configuration...")
    config = load_config()
    logger.info(f"  - Embedding model: {config.embedding.model_name}")
    logger.info(f"  - Chunking strategy: {config.chunking.strategy}")
    logger.info(f"  - Vector store: {config.vectorstore.collection_name}")
    logger.info(f"  - Top-K results: {config.retrieval.top_k}")
    
    # Step 2: Initialize embedding model
    logger.info("\n🔤 Step 2: Loading Embedding Model...")
    model = load_embedding_model(config.embedding.model_name)
    logger.info(f"  ✓ Model loaded: {config.embedding.model_name}")
    logger.info(f"  ✓ Embedding dimension: 384")
    
    # Step 3: Initialize vector store
    logger.info("\n🗄️  Step 3: Initializing Vector Store...")
    initialize_vectorstore(
        config.vectorstore.persist_directory,
        config.vectorstore.collection_name
    )
    logger.info(f"  ✓ ChromaDB initialized at: {config.vectorstore.persist_directory}")
    
    # Step 4: Create sample documents for demonstration
    logger.info("\n📄 Step 4: Creating Sample Documents...")
    
    sample_docs = {
        "sample_doc1.txt": """
        Machine Learning is a subset of Artificial Intelligence that focuses on 
        enabling computers to learn from data without being explicitly programmed.
        
        Key concepts in machine learning include:
        - Supervised learning: Learning from labeled data
        - Unsupervised learning: Finding patterns in unlabeled data
        - Reinforcement learning: Learning through interaction and rewards
        
        Popular machine learning frameworks include TensorFlow, PyTorch, and scikit-learn.
        """,
        
        "sample_doc2.txt": """
        Python is a high-level, interpreted programming language known for its simplicity
        and readability. It was created by Guido van Rossum in 1991.
        
        Python is widely used for:
        - Web development (Django, Flask)
        - Data science (NumPy, Pandas, scikit-learn)
        - Artificial intelligence (TensorFlow, PyTorch)
        - Automation and scripting
        
        The Python community is very active and welcoming to beginners.
        """
    }
    
    # Step 5: Ingest sample documents
    logger.info("\n📥 Step 5: Ingesting Sample Documents...")
    all_chunks = []
    
    for doc_name, doc_content in sample_docs.items():
        logger.info(f"  Processing: {doc_name}")
        
        # Create temporary file
        temp_path = project_dir / doc_name
        temp_path.write_text(doc_content)
        
        try:
            # Ingest document
            chunks = ingest_document(
                str(temp_path),
                chunking_strategy=config.chunking.strategy,
                chunk_size=config.chunking.chunk_size,
                overlap=config.chunking.overlap
            )
            all_chunks.extend(chunks)
            logger.info(f"    ✓ Created {len(chunks)} chunks")
        finally:
            # Clean up temp file
            temp_path.unlink()
    
    logger.info(f"  ✓ Total chunks created: {len(all_chunks)}")
    
    # Step 6: Embed chunks
    logger.info("\n⚡ Step 6: Embedding Chunks...")
    chunk_texts = [chunk.text for chunk in all_chunks]
    embeddings = embed_texts(chunk_texts)
    logger.info(f"  ✓ Embedded {len(embeddings)} chunks")
    logger.info(f"  ✓ Embedding shape: {embeddings.shape}")
    
    # Step 7: Store chunks
    logger.info("\n💾 Step 7: Storing Chunks in Vector Store...")
    chunks_with_embeddings = [
        {
            "chunk_id": chunk.chunk_id,
            "embedding": embedding,
            "text": chunk.text,
            "metadata": {
                "source_file": chunk.source_file,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index
            }
        }
        for chunk, embedding in zip(all_chunks, embeddings)
    ]
    
    add_chunks_to_store(chunks_with_embeddings)
    
    # Step 8: Check vectorstore stats
    stats = get_collection_stats()
    logger.info(f"  ✓ Vector store statistics:")
    logger.info(f"    - Documents stored: {stats['document_count']}")
    logger.info(f"    - Metadata fields: {stats['metadata_fields']}")
    
    # Step 9: Query the system
    logger.info("\n🔍 Step 9: Querying the System...")
    queries = [
        "What is machine learning?",
        "Tell me about Python"
    ]
    
    for query in queries:
        logger.info(f"\n  Query: '{query}'")
        
        # Embed query
        query_embedding = embed_query(query)
        logger.info(f"    ✓ Query embedded (dim: {query_embedding.shape[0]})")
        
        # Retrieve
        retrieved = retrieve_with_enhancements(
            query=query,
            query_embedding=query_embedding,
            use_reranking=False,
            use_hyde=False,
            use_hybrid=False,
            top_k=config.retrieval.top_k
        )
        logger.info(f"    ✓ Retrieved {len(retrieved)} chunks")
        
        # Display retrieved chunks
        for i, chunk in enumerate(retrieved, 1):
            score = chunk.get('score', 0)
            source = chunk.get('metadata', {}).get('source_file', 'unknown')
            preview = chunk['text'][:100].replace('\n', ' ') + "..."
            logger.info(f"      [{i}] {source} (score: {score:.3f})")
            logger.info(f"          {preview}")
        
        # Step 10: Generate answer (without LLM to avoid API key requirement)
        logger.info(f"    📝 Answer generation would require:")
        logger.info(f"       - ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
        logger.info(f"       - Call: generate_with_citations(query, retrieved)")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("=" * 70)
    logger.info("\nTo generate answers with citations, you need to:")
    logger.info("  1. Set ANTHROPIC_API_KEY or OPENAI_API_KEY environment variable")
    logger.info("  2. Call: answer, citations = generate_with_citations(query, retrieved)")
    logger.info("\nTo use the Streamlit app:")
    logger.info("  1. Run: streamlit run app.py")
    logger.info("  2. Upload documents in the UI")
    logger.info("  3. Ask questions and get answers with citations")
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    try:
        example_full_pipeline()
        logger.info("\n🎉 Example completed successfully!")
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}", exc_info=True)
        sys.exit(1)
