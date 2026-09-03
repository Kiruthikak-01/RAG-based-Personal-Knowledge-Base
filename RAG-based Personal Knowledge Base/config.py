"""
Configuration management for the RAG-based Personal Knowledge Base.

This module centralizes all configurable parameters including:
- Chunking strategy and parameters
- Embedding model selection
- Vector store settings
- Retrieval parameters
- LLM configuration
- Evaluation settings
"""

from dataclasses import dataclass
from typing import Literal
import os


@dataclass
class ChunkingConfig:
    """
    Configuration for document chunking strategy.
    
    Attributes:
        strategy: "fixed" for fixed-size chunking or "recursive" for structure-aware chunking
        chunk_size: Size of each chunk in tokens/characters
        overlap: Number of overlapping tokens/characters between chunks
    """
    strategy: Literal["fixed", "recursive"] = "fixed"
    chunk_size: int = 512
    overlap: int = 50


@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding model.
    
    Attributes:
        model_name: Name of the sentence-transformers model to use
                   (e.g., "all-MiniLM-L6-v2", "all-mpnet-base-v2")
    """
    model_name: str = "all-MiniLM-L6-v2"


@dataclass
class VectorStoreConfig:
    """
    Configuration for ChromaDB vector store.
    
    Attributes:
        persist_directory: Path where ChromaDB persists data
        collection_name: Name of the collection to use
    """
    persist_directory: str = "./chroma_db"
    collection_name: str = "knowledge_base"


@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval enhancements.
    
    Attributes:
        top_k: Number of top results to return from similarity search
        use_reranking: Whether to use cross-encoder re-ranking
        reranker_model: Cross-encoder model name (used if use_reranking=True)
        use_hyde: Whether to use HyDE query transformation
        use_hybrid_search: Whether to combine BM25 + dense retrieval
        candidate_k: Number of candidates to retrieve before re-ranking
    """
    top_k: int = 5
    use_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_hyde: bool = False
    use_hybrid_search: bool = False
    candidate_k: int = 20


@dataclass
class GenerationConfig:
    """
    Configuration for LLM-based generation.
    
    Attributes:
        llm_provider: "claude" for Anthropic or "openai" for OpenAI
        model_name: Model identifier (e.g., "claude-3-sonnet-20240229" or "gpt-4")
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 to 1.0)
    """
    llm_provider: Literal["claude", "openai"] = "claude"
    model_name: str = "claude-3-sonnet-20240229"
    max_tokens: int = 1024
    temperature: float = 0.3


@dataclass
class EvaluationConfig:
    """
    Configuration for evaluation scripts.
    
    Attributes:
        test_set_path: Path to CSV file with test questions and correct sources
        use_ragas: Whether to use RAGAS library for generation evaluation
    """
    test_set_path: str = "./eval/test_set.csv"
    use_ragas: bool = True


@dataclass
class RAGConfig:
    """
    Top-level configuration aggregating all sub-configs.
    """
    chunking: ChunkingConfig = None
    embedding: EmbeddingConfig = None
    vectorstore: VectorStoreConfig = None
    retrieval: RetrievalConfig = None
    generation: GenerationConfig = None
    evaluation: EvaluationConfig = None
    
    def __post_init__(self):
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.vectorstore is None:
            self.vectorstore = VectorStoreConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()


def load_config() -> RAGConfig:
    """
    Load configuration from environment or use defaults.
    
    Environment variables can override defaults with format: RAG_<SECTION>_<KEY>
    Example: RAG_EMBEDDING_MODEL_NAME=all-mpnet-base-v2
    """
    config = RAGConfig()
    
    # TODO: Implement environment variable overrides
    # Scan os.environ for RAG_* prefixed variables and apply them to the config
    
    return config


if __name__ == "__main__":
    config = load_config()
    print("Default RAG Configuration:")
    print(f"  Chunking: strategy={config.chunking.strategy}, size={config.chunking.chunk_size}")
    print(f"  Embedding model: {config.embedding.model_name}")
    print(f"  VectorStore: {config.vectorstore.collection_name}")
    print(f"  Retrieval top-k: {config.retrieval.top_k}")
    print(f"  LLM: {config.generation.llm_provider}/{config.generation.model_name}")
