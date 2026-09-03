# RAG-based Personal Knowledge Base: "Chat with My Notes"

A modular Python project for building a **Retrieval-Augmented Generation (RAG)** system that lets you upload documents (PDFs, text, Markdown) and chat with them using an LLM (Claude or OpenAI) with source citations.

This is a **learning-focused scaffolding project** where core RAG components are left as TODO implementations for you to build and understand the internals.

## 🎯 Project Structure

```
RAG-based Personal Knowledge Base/
├── config.py              # Configuration management (chunking, embedding, LLM, etc.)
├── ingestion.py           # Document extraction & chunking (TODO: implement)
├── embeddings.py          # Embedding model wrapper (TODO: implement)
├── vectorstore.py         # ChromaDB vector store (TODO: implement)
├── retrieval.py           # Retrieval with enhancements (TODO: implement)
├── generation.py          # LLM prompt construction & generation (TODO: implement)
├── eval.py                # Evaluation metrics (TODO: implement)
├── app.py                 # Streamlit web interface (stubs for integration)
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── README.md              # This file
```

## 🔧 Key Components

### 1. **Ingestion (`ingestion.py`)**
- Extract text from PDFs (pdfplumber or PyMuPDF)
- Load plain text and Markdown files
- **Configurable chunking strategies:**
  - **Fixed-size**: Split into overlapping windows of fixed size
  - **Recursive**: Hierarchical splitting respecting document structure (paragraphs → sentences → words)
- Attach metadata to every chunk: source file, page number, chunk index

### 2. **Embeddings (`embeddings.py`)**
- Wrap sentence-transformers model (e.g., `all-MiniLM-L6-v2`)
- **Critical:** Enforce single-model consistency
  - Same model must embed both chunks (ingestion) and queries (retrieval)
  - Attempting to load a different model raises `AssertionError`
- Embed collections and single queries

### 3. **Vector Store (`vectorstore.py`)**
- Persistent ChromaDB database
- Store chunks with embeddings and metadata
- Support similarity search with optional metadata filtering
- Retrieve chunks by ID for citation lookups

### 4. **Retrieval (`retrieval.py`)**
- **Dense retrieval:** Semantic similarity search
- **Optional re-ranking:** Cross-encoder re-ranking of top-K candidates
- **Optional HyDE:** Hypothetical Document Embeddings query transformation
- **Optional hybrid search:** Combine BM25 (sparse) + dense (semantic) retrieval
- Configurable via `config.py`

### 5. **Generation (`generation.py`)**
- Build system and user prompts with retrieved context
- **Critical instruction:** "Say 'I don't know' if not in context"
- ID-based citation tagging in prompts (e.g., `[CITE: chunk_id]`)
- LLM API calls: Claude (Anthropic) or OpenAI (configurable)
- Extract citations from response and map to source metadata
- Support both streaming and non-streaming responses

### 6. **Evaluation (`eval.py`)**
- **Retrieval evaluation:** Recall@K, Precision@K against manually labeled test set (CSV format)
- **Generation evaluation:** RAGAS library integration for:
  - Faithfulness: Is answer grounded in context?
  - Answer Relevance: Does answer address the question?

### 7. **Streamlit App (`app.py`)**
- File upload for documents
- Configuration sidebar for chunking, retrieval, LLM parameters
- Chat interface with message history
- Stream LLM responses token-by-token
- Display source citations with document name, page number, chunk ID

## 🚀 Setup Instructions

### Prerequisites
- Python 3.9+
- API keys for Claude (Anthropic) and/or OpenAI (see `.env.example`)

### Installation

1. **Clone or set up the project:**
   ```bash
   cd "RAG-based Personal Knowledge Base"
   ```

2. **Create and activate a Python virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env and add your API keys:
   # - ANTHROPIC_API_KEY (for Claude)
   # - OPENAI_API_KEY (for OpenAI)
   ```

5. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```
   The app will open at `http://localhost:8501`

## 📋 Configuration

All configurable parameters are in `config.py`:

| Parameter | Default | Options |
|-----------|---------|---------|
| **Chunking Strategy** | `fixed` | `fixed`, `recursive` |
| **Chunk Size** | 512 | 100–2000 (characters) |
| **Chunk Overlap** | 50 | 0–500 (characters) |
| **Embedding Model** | `all-MiniLM-L6-v2` | Any sentence-transformers model |
| **Top-K Results** | 5 | 1–20 |
| **Use Re-ranking** | False | True/False |
| **Use HyDE** | False | True/False |
| **Use Hybrid Search** | False | True/False |
| **LLM Provider** | `claude` | `claude`, `openai` |
| **Model Name** | `claude-3-sonnet-20240229` | Claude or OpenAI model IDs |

Override defaults via:
- Sidebar in Streamlit app
- `.env` file environment variables (format: `RAG_<SECTION>_<KEY>`)
- Programmatically in `config.py`

## 📝 TODO Implementations

This project intentionally leaves core logic as TODO stubs for learning. Here's what you need to implement:

### High Priority (Core RAG Loop)
- [ ] `ingestion.extract_text_from_pdf()` — Extract text from PDFs
- [ ] `ingestion.chunk_text_fixed_size()` — Fixed-size chunking with overlap
- [ ] `ingestion.chunk_text_recursive()` — Recursive structure-aware chunking
- [ ] `embeddings.load_embedding_model()` — Load sentence-transformers model
- [ ] `embeddings.embed_texts()` — Embed text collections
- [ ] `vectorstore.initialize_vectorstore()` — Set up ChromaDB
- [ ] `vectorstore.add_chunks_to_store()` — Store chunks with embeddings
- [ ] `vectorstore.similarity_search()` — Retrieve by similarity
- [ ] `generation.build_system_prompt()` — Define assistant role
- [ ] `generation.build_user_prompt()` — Format context for LLM
- [ ] `generation.call_llm_api()` — Call Claude or OpenAI API
- [ ] `generation.extract_citations_from_response()` — Parse [CITE: ...] tags

### Medium Priority (Retrieval Enhancements)
- [ ] `retrieval.rerank_candidates()` — Cross-encoder re-ranking
- [ ] `retrieval.hyde_transform_query()` — HyDE query transformation
- [ ] `retrieval.bm25_search()` — BM25 keyword search
- [ ] `retrieval.hybrid_retrieval()` — Combine dense + sparse

### Lower Priority (Evaluation & Polish)
- [ ] `eval.load_test_set()` — Load CSV test set
- [ ] `eval.RetrievalEvaluator.evaluate_single_query()` — Eval one query
- [ ] `eval.RetrievalEvaluator.evaluate_dataset()` — Eval full test set
- [ ] `eval.GenerationEvaluator.evaluate_single_generation()` — RAGAS scoring
- [ ] `app.py` integration — Connect UI to implemented modules

## 📊 Evaluation Template

Create a test set in `eval/test_set.csv`:

```csv
question,correct_source_1,correct_source_2
"What is machine learning?","ml_intro.pdf","wikipedia.txt"
"How does backpropagation work?","ml_intro.pdf","deeplearning_book.pdf"
"Explain transformers.","papers.pdf"
```

Run evaluation:
```python
from eval import run_full_evaluation
from ingestion import ingest_document
from generation import generate_with_citations

metrics = run_full_evaluation(
    test_set_path="eval/test_set.csv",
    retrieval_fn=lambda q: retrieve_with_enhancements(q, ...),
    generation_fn=lambda q: generate_with_citations(q, ...)
)

print(f"Retrieval - Recall@5: {metrics['retrieval_metrics']['recall@5']:.2%}")
print(f"Generation - Faithfulness: {metrics['generation_metrics']['faithfulness']:.2f}")
```

## 🔑 Environment Variables

**Required for LLM calls:**
- `ANTHROPIC_API_KEY` — Get from [console.anthropic.com](https://console.anthropic.com)
- `OPENAI_API_KEY` — Get from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

**Optional:**
- `CHROMA_DB_PATH` — Where to persist vector DB (default: `./chroma_db`)
- `RAG_EMBEDDING_MODEL_NAME` — Override embedding model
- `RAG_GENERATION_LLM_PROVIDER` — Override LLM provider

⚠️ **Never commit `.env` with actual keys to version control!**

## 📚 Learning Goals

This project is structured to teach:

1. **Document Processing:** Extract and chunk text while preserving metadata
2. **Embeddings:** Understand dense representations and model consistency
3. **Vector Search:** Query embeddings for semantic retrieval
4. **Prompt Engineering:** Craft prompts that ground LLM outputs
5. **Citation Mapping:** Track retrieved chunks back to sources
6. **Retrieval Optimization:** Re-ranking, HyDE, hybrid search
7. **Evaluation:** Measure quality with retrieval and generation metrics

Each TODO is designed to stand alone — you can implement them in any order and test incrementally.

## 🛠️ Development Workflow

1. Pick a TODO implementation (start with `ingestion` → `embeddings` → `vectorstore`)
2. Write code following the docstring specifications
3. Test with a small document first
4. Integrate into `app.py` once working
5. Run evaluation scripts to verify quality

## 📖 Useful References

- **Sentence-Transformers:** [huggingface.co/sentence-transformers](https://huggingface.co/sentence-transformers)
- **ChromaDB:** [docs.trychroma.com](https://docs.trychroma.com)
- **LangChain:** [python.langchain.com](https://python.langchain.com)
- **Claude API:** [docs.anthropic.com](https://docs.anthropic.com)
- **OpenAI API:** [platform.openai.com/docs](https://platform.openai.com/docs)
- **RAGAS:** [github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)
- **Streamlit:** [docs.streamlit.io](https://docs.streamlit.io)

## 📄 License

This is a learning project. Feel free to modify and extend!

---

**Happy learning! Build something amazing! 🚀**
