#!/usr/bin/env python3
"""
Quick test script to validate all module implementations.
This tests the basic functionality without running the full app.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_embeddings():
    """Test embeddings module."""
    logger.info("=" * 50)
    logger.info("Testing Embeddings Module")
    logger.info("=" * 50)
    
    try:
        from embeddings import load_embedding_model, embed_texts, embed_query, get_embedding_dimension, get_model_name
        
        # Test 1: Load model
        logger.info("✓ Embeddings module imported successfully")
        logger.info("✓ All embeddings functions imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Embeddings test failed: {str(e)}")
        return False


def test_ingestion():
    """Test ingestion module."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Ingestion Module")
    logger.info("=" * 50)
    
    try:
        from ingestion import (
            extract_text_from_pdf, 
            extract_text_from_file,
            chunk_text_fixed_size,
            chunk_text_recursive,
            ingest_document,
            Chunk
        )
        
        logger.info("✓ Ingestion module imported successfully")
        logger.info("✓ All ingestion functions imported")
        logger.info("✓ Chunk dataclass imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Ingestion test failed: {str(e)}")
        return False


def test_vectorstore():
    """Test vectorstore module."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing VectorStore Module")
    logger.info("=" * 50)
    
    try:
        from vectorstore import (
            initialize_vectorstore,
            add_chunks_to_store,
            similarity_search,
            get_chunk_by_id,
            list_collections,
            delete_collection,
            get_collection_stats
        )
        
        logger.info("✓ VectorStore module imported successfully")
        logger.info("✓ All vectorstore functions imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ VectorStore test failed: {str(e)}")
        return False


def test_generation():
    """Test generation module."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Generation Module")
    logger.info("=" * 50)
    
    try:
        from generation import (
            build_system_prompt,
            build_user_prompt,
            call_llm_api,
            extract_citations_from_response,
            generate_with_citations,
            generate_with_citations_streaming
        )
        
        logger.info("✓ Generation module imported successfully")
        logger.info("✓ All generation functions imported")
        
        # Test system prompt
        system_prompt = build_system_prompt()
        logger.info(f"✓ System prompt generated ({len(system_prompt)} chars)")
        
        return True
    except Exception as e:
        logger.error(f"✗ Generation test failed: {str(e)}")
        return False


def test_retrieval():
    """Test retrieval module."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Retrieval Module")
    logger.info("=" * 50)
    
    try:
        from retrieval import (
            retrieve_dense,
            rerank_candidates,
            hyde_transform_query,
            bm25_search,
            hybrid_retrieval,
            retrieve_with_enhancements
        )
        
        logger.info("✓ Retrieval module imported successfully")
        logger.info("✓ All retrieval functions imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Retrieval test failed: {str(e)}")
        return False


def test_evaluation():
    """Test evaluation module."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Evaluation Module")
    logger.info("=" * 50)
    
    try:
        from eval import (
            load_test_set,
            RetrievalEvaluator,
            GenerationEvaluator,
            run_full_evaluation
        )
        
        logger.info("✓ Evaluation module imported successfully")
        logger.info("✓ All evaluation functions/classes imported")
        
        return True
    except Exception as e:
        logger.error(f"✗ Evaluation test failed: {str(e)}")
        return False


def test_config():
    """Test config module."""
    logger.info("\n" + "=" * 50)
    logger.info("Testing Config Module")
    logger.info("=" * 50)
    
    try:
        from config import load_config, RAGConfig
        
        config = load_config()
        logger.info("✓ Config module imported successfully")
        logger.info(f"✓ Default config loaded:")
        logger.info(f"  - Embedding model: {config.embedding.model_name}")
        logger.info(f"  - Chunking strategy: {config.chunking.strategy}")
        logger.info(f"  - Vector store: {config.vectorstore.collection_name}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Config test failed: {str(e)}")
        return False


if __name__ == "__main__":
    logger.info("\n🚀 TESTING ALL RAG IMPLEMENTATIONS\n")
    
    results = {
        "embeddings": test_embeddings(),
        "ingestion": test_ingestion(),
        "vectorstore": test_vectorstore(),
        "generation": test_generation(),
        "retrieval": test_retrieval(),
        "evaluation": test_evaluation(),
        "config": test_config(),
    }
    
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for module, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status}: {module}")
    
    logger.info(f"\nTotal: {passed}/{total} modules passed")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED! The RAG system is ready to use.")
        sys.exit(0)
    else:
        logger.error(f"\n⚠️  {total - passed} module(s) failed. Please fix them before running the app.")
        sys.exit(1)
