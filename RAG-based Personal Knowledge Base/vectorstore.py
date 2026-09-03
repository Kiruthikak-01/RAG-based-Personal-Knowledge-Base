"""
Vector store module for persistent storage and similarity search.

This module handles:
- ChromaDB collection management
- Storing chunks with embeddings and metadata
- Similarity search with optional metadata filtering
- Retrieving metadata for citations
"""

from typing import List, Dict, Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Global ChromaDB client (lazy-loaded)
_CHROMA_CLIENT = None


def initialize_vectorstore(persist_directory: str, collection_name: str):
    """
    Initialize or load a ChromaDB vector store collection.
    
    This function should:
    1. Initialize a persistent ChromaDB client if not already done
    2. Get or create a collection with the specified name
    3. Store collection reference globally for later use
    
    Args:
        persist_directory: Path where ChromaDB data will be persisted
        collection_name: Name of the collection (e.g., "knowledge_base")
        
    Returns:
        The ChromaDB collection object
        
    TODO: Implement ChromaDB initialization
    - Import chromadb library
    - Create a persistent client pointing to persist_directory
    - Get or create collection with collection_name
    - Cache collection globally
    - Log initialization events
    """
    global _CHROMA_CLIENT
    import chromadb
    from pathlib import Path
    
    logger.info(f"Initializing ChromaDB vectorstore at {persist_directory}")
    
    # Create persist directory if it doesn't exist
    Path(persist_directory).mkdir(parents=True, exist_ok=True)
    
    # Initialize persistent client
    _CHROMA_CLIENT = chromadb.PersistentClient(path=persist_directory)
    
    # Get or create collection
    collection = _CHROMA_CLIENT.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info(f"ChromaDB vectorstore initialized with collection '{collection_name}'")
    return collection


def add_chunks_to_store(
    chunks_with_embeddings: List[Dict[str, Any]]
) -> None:
    """
    Add embedded chunks to the vector store.
    
    Each item in chunks_with_embeddings should have:
    - chunk_id: Unique identifier
    - embedding: numpy array or list of floats
    - text: The chunk text
    - metadata: Dict with source_file, page_number, chunk_index, etc.
    
    This function should:
    1. Prepare data for ChromaDB insert
    2. Insert embeddings with metadata
    3. Handle duplicate IDs by updating or skipping
    
    Args:
        chunks_with_embeddings: List of dicts with chunk_id, embedding, text, metadata
        
    Raises:
        ValueError: If required fields are missing
        RuntimeError: If ChromaDB collection is not initialized
        
    TODO: Implement chunk storage
    - Verify collection is initialized
    - Extract chunk_id, embedding, text, metadata from each item
    - Call ChromaDB collection.upsert or .add
    - Handle metadata serialization for ChromaDB
    """
    if _CHROMA_CLIENT is None:
        raise RuntimeError("ChromaDB vectorstore not initialized. Call initialize_vectorstore() first.")
    
    if not chunks_with_embeddings:
        logger.warning("No chunks to add to vectorstore")
        return
    
    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for chunk in chunks_with_embeddings:
        if "chunk_id" not in chunk or "embedding" not in chunk or "text" not in chunk:
            raise ValueError("Each chunk must have 'chunk_id', 'embedding', and 'text' fields")
        
        ids.append(chunk["chunk_id"])
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append(chunk.get("metadata", {}))
    
    # Get current collection
    collection = _CHROMA_CLIENT.get_collection(name="knowledge_base")
    
    # Add to ChromaDB using upsert (handles duplicates gracefully)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    
    logger.info(f"Added {len(chunks_with_embeddings)} chunks to vectorstore")


def similarity_search(
    query_embedding: np.ndarray,
    top_k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k most similar chunks for a query embedding.
    
    This function should:
    1. Query ChromaDB collection with the embedding
    2. Optionally filter by metadata (e.g., source document)
    3. Return the top-k results with scores and metadata
    
    Args:
        query_embedding: Query vector (numpy array or list)
        top_k: Number of results to return
        metadata_filter: Optional metadata filter (e.g., {"source_file": "notes.pdf"})
                        ChromaDB uses {"$eq": value} syntax for exact match
        
    Returns:
        List of dicts with:
        - chunk_id: Identifier
        - text: Chunk text content
        - distance/score: Similarity score
        - metadata: Dict with source, page number, etc.
        
    Raises:
        RuntimeError: If ChromaDB collection is not initialized
        ValueError: If query_embedding has wrong shape
        
    TODO: Implement similarity search
    - Call ChromaDB collection.query
    - Format and return results
    - Handle metadata filtering
    """
    if _CHROMA_CLIENT is None:
        raise RuntimeError("ChromaDB vectorstore not initialized. Call initialize_vectorstore() first.")
    
    collection = _CHROMA_CLIENT.get_collection(name="knowledge_base")
    
    # Convert numpy array to list if needed
    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()
    
    # Build where filter if provided
    where_filter = None
    if metadata_filter:
        where_filter = {}
        for key, value in metadata_filter.items():
            where_filter[key] = {"$eq": value}
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter
    )
    
    # Format results
    retrieved_chunks = []
    if results["ids"] and len(results["ids"]) > 0:
        for i, chunk_id in enumerate(results["ids"][0]):
            retrieved_chunks.append({
                "chunk_id": chunk_id,
                "text": results["documents"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
            })
    
    logger.info(f"Retrieved {len(retrieved_chunks)} chunks via similarity search")
    return retrieved_chunks


def get_chunk_by_id(chunk_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single chunk by its ID (useful for citation lookup).
    
    Args:
        chunk_id: The unique chunk identifier
        
    Returns:
        Dict with text, metadata, etc., or None if not found
        
    TODO: Implement chunk retrieval by ID
    - Query ChromaDB for document with matching chunk_id
    - Return the document data
    """
    if _CHROMA_CLIENT is None:
        raise RuntimeError("ChromaDB vectorstore not initialized. Call initialize_vectorstore() first.")
    
    collection = _CHROMA_CLIENT.get_collection(name="knowledge_base")
    
    try:
        results = collection.get(ids=[chunk_id])
        if results["ids"] and len(results["ids"]) > 0:
            return {
                "chunk_id": results["ids"][0],
                "text": results["documents"][0],
                "metadata": results["metadatas"][0] if results["metadatas"] else {}
            }
    except Exception as e:
        logger.warning(f"Failed to retrieve chunk {chunk_id}: {str(e)}")
    
    return None


def list_collections() -> List[str]:
    """
    List all available collections in the vector store.
    
    Returns:
        List of collection names
        
    TODO: Implement collection listing
    - Query ChromaDB client for all collections
    """
    if _CHROMA_CLIENT is None:
        raise RuntimeError("ChromaDB vectorstore not initialized. Call initialize_vectorstore() first.")
    
    collections = _CHROMA_CLIENT.list_collections()
    collection_names = [c.name for c in collections]
    logger.info(f"Available collections: {collection_names}")
    return collection_names


def delete_collection(collection_name: str) -> None:
    """
    Delete an entire collection from the vector store.
    
    Args:
        collection_name: Name of collection to delete
        
    TODO: Implement collection deletion
    """
    if _CHROMA_CLIENT is None:
        raise RuntimeError("ChromaDB vectorstore not initialized. Call initialize_vectorstore() first.")
    
    _CHROMA_CLIENT.delete_collection(name=collection_name)
    logger.info(f"Deleted collection: {collection_name}")


def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the current collection (useful for debugging).
    
    Returns:
        Dict with:
        - document_count: Total number of chunks stored
        - collection_name: Name of collection
        - metadata_fields: List of metadata field names
        
    TODO: Implement stats retrieval
    - Count documents in collection
    - Inspect metadata fields
    """
    if _CHROMA_CLIENT is None:
        raise RuntimeError("ChromaDB vectorstore not initialized. Call initialize_vectorstore() first.")
    
    collection = _CHROMA_CLIENT.get_collection(name="knowledge_base")
    doc_count = collection.count()
    
    # Try to get metadata field names by sampling
    metadata_fields = set()
    try:
        sample = collection.get(limit=10)
        if sample["metadatas"]:
            for metadata in sample["metadatas"]:
                metadata_fields.update(metadata.keys())
    except:
        pass
    
    return {
        "document_count": doc_count,
        "collection_name": "knowledge_base",
        "metadata_fields": list(metadata_fields)
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("VectorStore module loaded. Awaiting implementation of ChromaDB integration.")
