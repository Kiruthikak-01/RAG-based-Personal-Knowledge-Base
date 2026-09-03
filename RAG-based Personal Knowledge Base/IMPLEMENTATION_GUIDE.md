# Implementation Guide: Getting Started with RAG Components

This guide walks through implementing one component to understand the pattern and dependencies.

## Starting Point: Embeddings Module

The **embeddings module** is a good starting point because:
1. It has few external dependencies (just sentence-transformers)
2. It's used by both ingestion and retrieval modules
3. It's self-contained and testable independently

### Step 1: Implement `load_embedding_model()`

```python
# embeddings.py

def load_embedding_model(model_name: str):
    """Load a sentence-transformers model and cache it globally."""
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
```

### Step 2: Implement `embed_texts()`

```python
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts into vectors."""
    if not texts:
        return np.array([])
    
    # Ensure model is loaded (lazy-load if needed)
    if _EMBEDDING_MODEL is None:
        load_embedding_model("all-MiniLM-L6-v2")
    
    # Embed all texts
    embeddings = _EMBEDDING_MODEL.encode(texts, convert_to_numpy=True)
    return embeddings
```

### Step 3: Implement `embed_query()`

```python
def embed_query(query: str) -> np.ndarray:
    """Embed a single query string into a vector."""
    embeddings = embed_texts([query])
    return embeddings[0]  # Return 1D array
```

### Step 4: Test Your Implementation

```python
# test_embeddings.py
from embeddings import load_embedding_model, embed_texts, embed_query
import numpy as np

# Test 1: Load model
print("Test 1: Loading model...")
load_embedding_model("all-MiniLM-L6-v2")
print("✓ Model loaded")

# Test 2: Embed texts
print("\nTest 2: Embedding texts...")
texts = ["Hello world", "Python programming", "Machine learning"]
embeddings = embed_texts(texts)
assert embeddings.shape == (3, 384)  # all-MiniLM-L6-v2 outputs 384-dim
print(f"✓ Embedded {embeddings.shape[0]} texts, dim: {embeddings.shape[1]}")

# Test 3: Embed query
print("\nTest 3: Embedding query...")
query_embedding = embed_query("What is AI?")
assert query_embedding.shape == (384,)
print("✓ Query embedding correct shape")

# Test 4: Consistency check
print("\nTest 4: Consistency check...")
try:
    load_embedding_model("all-mpnet-base-v2")  # Different model
    print("✗ Should have raised AssertionError!")
except AssertionError as e:
    print(f"✓ Correctly rejected different model: {e}")

print("\n✅ All embeddings tests passed!")
```

Run: `python test_embeddings.py`

---

## Next: Ingestion Module

Once embeddings work, implement ingestion:

### Step 1: Extract Text from PDF

```python
# ingestion.py
import pdfplumber

def extract_text_from_pdf(file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
    """Extract text from a PDF file with page-level information."""
    full_text = ""
    page_info = []
    
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text += text + "\n\n"
                page_info.append((text, page_num))
    
    return full_text, page_info
```

### Step 2: Implement Fixed-Size Chunking

```python
def chunk_text_fixed_size(
    text: str,
    source_file: str,
    chunk_size: int = 512,
    overlap: int = 50,
    page_number: int = None
) -> List[Chunk]:
    """Split text into fixed-size overlapping chunks."""
    chunks = []
    chunk_index = 0
    
    # Start with overlapping windows
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        
        if len(chunk_text.strip()) == 0:
            continue  # Skip empty chunks
        
        # Create chunk with metadata
        chunk = Chunk(
            text=chunk_text,
            source_file=source_file,
            page_number=page_number,
            chunk_index=chunk_index,
            chunk_id=f"{source_file}_{page_number}_{chunk_index}" if page_number else f"{source_file}_{chunk_index}"
        )
        chunks.append(chunk)
        chunk_index += 1
    
    return chunks
```

### Step 3: Test Ingestion

```python
# test_ingestion.py
from ingestion import extract_text_from_pdf, chunk_text_fixed_size, ingest_document

# Create a test PDF or use an existing one
file_path = "test.pdf"

# Test extraction
print("Test 1: Extract PDF text...")
full_text, page_info = extract_text_from_pdf(file_path)
print(f"✓ Extracted {len(full_text)} characters from {len(page_info)} pages")

# Test chunking
print("\nTest 2: Chunk text...")
chunks = chunk_text_fixed_size(full_text, "test.pdf", chunk_size=256, overlap=30)
print(f"✓ Created {len(chunks)} chunks")
for chunk in chunks[:2]:
    print(f"  - Chunk {chunk.chunk_index}: {len(chunk.text)} chars, ID: {chunk.chunk_id}")

# Test full ingestion
print("\nTest 3: Full ingestion pipeline...")
chunks = ingest_document(file_path, chunking_strategy="fixed", chunk_size=256)
print(f"✓ Ingested {len(chunks)} chunks with metadata")

print("\n✅ All ingestion tests passed!")
```

---

## Next: Vector Store (ChromaDB)

```python
# vectorstore.py
import chromadb

def initialize_vectorstore(persist_directory: str, collection_name: str):
    """Initialize ChromaDB collection."""
    global _CHROMA_CLIENT
    
    # Create persistent client
    _CHROMA_CLIENT = chromadb.PersistentClient(path=persist_directory)
    logger.info(f"Initialized ChromaDB at {persist_directory}")
    
    # Get or create collection
    collection = _CHROMA_CLIENT.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )
    logger.info(f"Collection '{collection_name}' ready")
    
    return collection


def add_chunks_to_store(chunks_with_embeddings: List[Dict[str, Any]]) -> None:
    """Add embedded chunks to the vector store."""
    from vectorstore import initialize_vectorstore
    from config import load_config
    
    cfg = load_config()
    collection = initialize_vectorstore(
        cfg.vectorstore.persist_directory,
        cfg.vectorstore.collection_name
    )
    
    # Prepare data for ChromaDB
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for item in chunks_with_embeddings:
        ids.append(item["chunk_id"])
        embeddings.append(item["embedding"].tolist())  # Convert numpy to list
        documents.append(item["text"])
        metadatas.append(item["metadata"])
    
    # Insert into collection
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    logger.info(f"Stored {len(ids)} chunks in vector database")


def similarity_search(
    query_embedding: np.ndarray,
    top_k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Retrieve top-k most similar chunks."""
    from vectorstore import initialize_vectorstore
    from config import load_config
    
    cfg = load_config()
    collection = initialize_vectorstore(
        cfg.vectorstore.persist_directory,
        cfg.vectorstore.collection_name
    )
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=metadata_filter  # None if no filter
    )
    
    # Format results
    retrieved_chunks = []
    for i, (chunk_id, distance, text, metadata) in enumerate(
        zip(results["ids"][0], results["distances"][0], 
            results["documents"][0], results["metadatas"][0])
    ):
        retrieved_chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "distance": distance,
            "metadata": metadata
        })
    
    return retrieved_chunks
```

---

## Testing the Full Pipeline

Once you've implemented embeddings, ingestion, and vectorstore:

```python
# test_rag_pipeline.py
from ingestion import ingest_document
from embeddings import embed_texts, embed_query, load_embedding_model
from vectorstore import initialize_vectorstore, add_chunks_to_store, similarity_search
from config import load_config
import numpy as np

# Load config
cfg = load_config()

# Step 1: Ingest document
print("Step 1: Ingesting document...")
chunks = ingest_document("sample.pdf", chunking_strategy="fixed")
print(f"✓ Ingested {len(chunks)} chunks")

# Step 2: Initialize embedding model
print("\nStep 2: Loading embedding model...")
load_embedding_model(cfg.embedding.model_name)
print("✓ Embedding model loaded")

# Step 3: Embed chunks
print("\nStep 3: Embedding chunks...")
chunk_texts = [c.text for c in chunks]
embeddings = embed_texts(chunk_texts)
chunks_with_embeddings = [
    {
        "chunk_id": c.chunk_id,
        "text": c.text,
        "embedding": embeddings[i],
        "metadata": {
            "source_file": c.source_file,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index
        }
    }
    for i, c in enumerate(chunks)
]
print(f"✓ Embedded {len(chunks_with_embeddings)} chunks")

# Step 4: Store in vector database
print("\nStep 4: Storing in vector database...")
initialize_vectorstore(cfg.vectorstore.persist_directory, cfg.vectorstore.collection_name)
add_chunks_to_store(chunks_with_embeddings)
print("✓ Chunks stored in ChromaDB")

# Step 5: Retrieve by similarity
print("\nStep 5: Testing retrieval...")
query = "What is the main topic?"
query_embedding = embed_query(query)
results = similarity_search(query_embedding, top_k=5)
print(f"✓ Retrieved {len(results)} results for query: '{query}'")
for i, result in enumerate(results, 1):
    print(f"  [{i}] Distance: {result['distance']:.3f}")
    print(f"      Source: {result['metadata']['source_file']}")
    print(f"      Text: {result['text'][:50]}...")

print("\n✅ Full RAG pipeline test passed!")
```

---

## Tips for Successful Implementation

1. **Start small:** Test each function independently before integrating
2. **Use print statements:** Debug by printing intermediate values
3. **Handle edge cases:** Empty inputs, malformed data, etc.
4. **Log everything:** Use `logger.info()` to track execution
5. **Validate assumptions:** Check shapes of numpy arrays, list lengths
6. **Read docstrings carefully:** They specify exactly what you need to do
7. **Test incrementally:** Implement one TODO at a time, test, then move on

## Common Pitfalls

- ❌ **Mixing embedding models:** Always enforce single model via global cache
- ❌ **Forgetting metadata:** Every chunk needs source_file, page_number, chunk_id
- ❌ **Hardcoding config:** Use `config.py` for all parameters
- ❌ **Not handling empty inputs:** Add guards for edge cases
- ❌ **API key in code:** Always read from environment variables

---

Good luck implementing! 🚀
