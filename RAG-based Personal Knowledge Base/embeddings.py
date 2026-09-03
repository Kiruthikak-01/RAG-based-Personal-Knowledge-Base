"""
Embeddings module for encoding text into dense vector representations.

This module handles:
- Loading and caching a sentence-transformers embedding model
- Embedding chunks for ingestion
- Embedding queries at retrieval time
- Enforcing consistency: same model must be used for both chunks and queries
"""

from typing import List, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global cache for the embedding model to enforce single-model consistency
_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_NAME = None


def load_embedding_model(model_name: str):
    """
    Load a sentence-transformers model and cache it globally.
    
    CRITICAL: This function enforces model consistency. All embeddings in the system
    must use the SAME model. Once a model is loaded, attempting to load a different
    model should raise an AssertionError.
    
    This function should:
    1. Check if a model is already cached
    2. If cached model name differs from requested name, raise AssertionError
    3. Load the model using sentence-transformers library
    4. Store in global cache
    
    Args:
        model_name: Identifier for the sentence-transformers model
                   (e.g., "all-MiniLM-L6-v2", "all-mpnet-base-v2")
                   
    Raises:
        AssertionError: If attempting to load a different model than already cached
        
    TODO: Implement model loading
    - Use sentence_transformers.SentenceTransformer
    - Cache model globally
    - Enforce single-model constraint with assertion and error message
    - Log model loading events
    """
    global _EMBEDDING_MODEL, _EMBEDDING_MODEL_NAME
    
    # Check consistency: if model already loaded, ensure same name
    if _EMBEDDING_MODEL is not None:
        assert _EMBEDDING_MODEL_NAME == model_name, \
            f"Model mismatch! Already loaded: {_EMBEDDING_MODEL_NAME}, " \
            f"trying to load: {model_name}. Embeddings must use the same model for chunks and queries."
        return _EMBEDDING_MODEL
    
    # Load the model
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading embedding model: {model_name}")
    _EMBEDDING_MODEL = SentenceTransformer(model_name)
    _EMBEDDING_MODEL_NAME = model_name
    logger.info(f"Model loaded. Embedding dimension: {_EMBEDDING_MODEL.get_sentence_embedding_dimension()}")
    
    return _EMBEDDING_MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts into vectors.
    
    This function should:
    1. Verify that a model is loaded (call load_embedding_model if needed)
    2. Use the cached model to encode all texts
    3. Return numpy array of shape (len(texts), embedding_dim)
    
    Args:
        texts: List of text strings to embed
        
    Returns:
        numpy array of shape (len(texts), embedding_dimension)
        Each row is the embedding vector for the corresponding text
        
    Raises:
        RuntimeError: If embedding model failed to load
        
    TODO: Implement text embedding
    - Ensure model is loaded via load_embedding_model
    - Batch encode texts for efficiency
    - Return numpy array
    - Handle empty input gracefully
    """
    if not texts:
        return np.array([])
    
    # Ensure model is loaded (lazy-load if needed)
    if _EMBEDDING_MODEL is None:
        load_embedding_model("all-MiniLM-L6-v2")
    
    # Embed all texts
    embeddings = _EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string into a vector.
    
    This is a convenience wrapper around embed_texts for single queries.
    Uses the SAME model that was used for chunk embeddings.
    
    Args:
        query: Single query string
        
    Returns:
        1D numpy array (embedding vector)
        
    TODO: Implement single query embedding
    - Call embed_texts with single-item list
    - Return the first (and only) row as 1D array
    """
    embeddings = embed_texts([query])
    return embeddings[0]  # Return 1D array


def get_embedding_dimension() -> int:
    """
    Get the dimensionality of the embedding space for the loaded model.
    
    Useful for setting up vector stores or debugging embedding shapes.
    
    Returns:
        Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        
    Raises:
        RuntimeError: If no model is loaded
        
    TODO: Implement dimension retrieval
    - Verify model is loaded
    - Return the embedding dimension
    """
    if _EMBEDDING_MODEL is None:
        raise RuntimeError("No embedding model loaded. Call load_embedding_model() first.")
    return _EMBEDDING_MODEL.get_sentence_embedding_dimension()


def get_model_name() -> str:
    """
    Get the name of the currently loaded embedding model.
    
    Useful for validation and logging to verify consistency.
    
    Returns:
        Name of the loaded model
        
    Raises:
        RuntimeError: If no model is loaded
        
    TODO: Return the cached model name
    """
    if _EMBEDDING_MODEL_NAME is None:
        raise RuntimeError("No embedding model loaded. Call load_embedding_model() first.")
    return _EMBEDDING_MODEL_NAME


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Embeddings module loaded. Awaiting implementation of model loading and embedding logic.")
