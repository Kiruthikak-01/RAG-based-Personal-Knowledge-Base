"""
Document ingestion module for extracting text from various document formats.

This module handles:
- PDF text extraction (using pdfplumber or PyMuPDF)
- Plain text file loading
- Markdown file parsing
- Metadata attachment (source, page number, chunk index)
"""

from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """
    Represents a single text chunk with metadata.
    
    Attributes:
        text: The actual text content of the chunk
        source_file: Name of the source document (e.g., "notes.pdf")
        page_number: Page number if applicable (None for plain text/markdown)
        chunk_index: Index of this chunk within the document
        chunk_id: Unique identifier for the chunk (should be set during chunking)
        metadata: Additional metadata dictionary (e.g., section title, keywords)
    """
    text: str
    source_file: str
    page_number: int = None
    chunk_index: int = None
    chunk_id: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def extract_text_from_pdf(file_path: str) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Extract text from a PDF file with page-level information.
    
    This function should:
    1. Use pdfplumber or PyMuPDF to read the PDF
    2. Extract text from each page
    3. Return full text and a list of (text_chunk, page_number) tuples
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Tuple of:
        - full_text: Concatenated text from all pages
        - page_info: List of (page_text, page_number) for page-level tracking
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ValueError: If the PDF is corrupted or unreadable
        
    TODO: Implement PDF text extraction
    - Handle multi-column layouts if applicable
    - Preserve structure (headings, lists) where possible
    - Return page-level metadata for citation purposes
    """
    import pdfplumber
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    logger.info(f"Extracting text from PDF: {file_path}")
    
    full_text = ""
    page_info = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    full_text += page_text + "\n"
                    page_info.append((page_text, page_num))
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF {file_path}: {str(e)}")
    
    logger.info(f"Extracted {len(page_info)} pages from PDF")
    return full_text, page_info


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a plain text or Markdown file.
    
    This function should:
    1. Read the file with UTF-8 encoding
    2. Return the raw text content
    3. Handle common encoding issues gracefully
    
    Args:
        file_path: Path to the text or Markdown file
        
    Returns:
        Text content of the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        UnicodeDecodeError: If file encoding is not UTF-8 compatible
        
    TODO: Implement plain text file extraction
    - Handle UTF-8 encoding errors with fallback encoding
    - Optionally normalize whitespace
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Text file not found: {file_path}")
    
    logger.info(f"Extracting text from file: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 encoding
        logger.warning(f"UTF-8 decoding failed, trying latin-1 for {file_path}")
        with open(file_path, 'r', encoding='latin-1') as f:
            text = f.read()
    
    return text


def chunk_text_fixed_size(
    text: str,
    source_file: str,
    chunk_size: int = 512,
    overlap: int = 50,
    page_number: int = None
) -> List[Chunk]:
    """
    Split text into fixed-size chunks with overlapping windows.
    
    This is the simpler chunking strategy: divide text into chunks of ~chunk_size
    characters/tokens with ~overlap characters/tokens between consecutive chunks.
    
    This function should:
    1. Split text into overlapping windows of specified size
    2. Create a Chunk object for each window
    3. Assign sequential chunk_index and generate unique chunk_ids
    4. Attach source metadata
    
    Args:
        text: The text to chunk
        source_file: Name of the source document
        chunk_size: Target size of each chunk (in characters or tokens)
        overlap: Overlap size between consecutive chunks
        page_number: Optional page number for citations
        
    Returns:
        List of Chunk objects with metadata attached
        
    TODO: Implement fixed-size chunking
    - Decide if chunk_size is in characters or tokens (recommend characters for simplicity)
    - Implement overlapping window logic
    - Assign chunk_ids (e.g., f"{source_file}_{page}_{index}")
    - Ensure every chunk has source_file, chunk_index, and page_number set
    """
    chunks = []
    stride = chunk_size - overlap  # Step size for sliding window
    chunk_index = 0
    
    for start_idx in range(0, len(text), stride):
        end_idx = start_idx + chunk_size
        chunk_text = text[start_idx:end_idx]
        
        if len(chunk_text.strip()) > 0:  # Only create non-empty chunks
            chunk_id = f"{source_file.replace('.', '_')}_{page_number or 'all'}_{chunk_index}"
            chunk = Chunk(
                text=chunk_text,
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_id=chunk_id,
                metadata={"strategy": "fixed", "chunk_size": chunk_size}
            )
            chunks.append(chunk)
            chunk_index += 1
    
    logger.info(f"Fixed-size chunking: created {len(chunks)} chunks from {source_file}")
    return chunks


def chunk_text_recursive(
    text: str,
    source_file: str,
    chunk_size: int = 512,
    overlap: int = 50,
    page_number: int = None
) -> List[Chunk]:
    """
    Split text into semantically-aware chunks using recursive splitting.
    
    This is the more sophisticated chunking strategy: recursively split on
    structural boundaries (paragraphs, sentences, words) to create chunks that
    respect document structure rather than just splitting at fixed positions.
    
    This function should:
    1. Try splitting on double newlines (paragraph breaks) first
    2. If chunks are still too large, split on single newlines
    3. If still too large, split on ". " (sentence boundaries)
    4. If still too large, split on " " (word boundaries)
    5. Create Chunk objects with the same metadata discipline as fixed-size
    
    Args:
        text: The text to chunk
        source_file: Name of the source document
        chunk_size: Target size of each chunk (in characters/tokens)
        overlap: Overlap size between recursive chunks
        page_number: Optional page number for citations
        
    Returns:
        List of Chunk objects with structure-aware splitting
        
    TODO: Implement recursive chunking
    - Define a sequence of separators to try: ["\n\n", "\n", ". ", " "]
    - Implement recursive splitting logic
    - Maintain overlap between chunks when splitting hierarchically
    - Assign chunk_ids consistently
    - Ensure metadata is complete for every chunk
    """
    def recursive_split(text, separators):
        """Recursively split text on separators until chunks are small enough."""
        good_splits = []
        separator = separators[-1]
        
        for _i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                break
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        good_splits = []
        for s in splits:
            if len(s) < chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged_text = separator.join(good_splits)
                    good_splits = [merged_text] if merged_text else []
                
                if len(s) < chunk_size:
                    good_splits.append(s)
                else:
                    other_splits = recursive_split(s, separators)
                    good_splits.extend(other_splits)
        
        return good_splits
    
    separators = ["\n\n", "\n", ". ", " ", ""]
    split_texts = recursive_split(text, separators)
    
    # Create chunks with overlap
    chunks = []
    chunk_index = 0
    
    for i, chunk_text in enumerate(split_texts):
        if len(chunk_text.strip()) > 0:
            # Build overlap from previous chunk
            if i > 0 and overlap > 0:
                overlap_text = split_texts[i-1][-overlap:] if len(split_texts[i-1]) > overlap else split_texts[i-1]
                chunk_text = overlap_text + chunk_text
            
            chunk_id = f"{source_file.replace('.', '_')}_{page_number or 'all'}_{chunk_index}"
            chunk = Chunk(
                text=chunk_text,
                source_file=source_file,
                page_number=page_number,
                chunk_index=chunk_index,
                chunk_id=chunk_id,
                metadata={"strategy": "recursive"}
            )
            chunks.append(chunk)
            chunk_index += 1
    
    logger.info(f"Recursive chunking: created {len(chunks)} chunks from {source_file}")
    return chunks


def ingest_document(
    file_path: str,
    chunking_strategy: str = "fixed",
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Chunk]:
    """
    Main entry point: ingest a document and return chunked content with metadata.
    
    This orchestration function should:
    1. Detect file type (PDF, .txt, .md)
    2. Extract text using appropriate extractor
    3. Apply the specified chunking strategy
    4. Return chunks with complete metadata
    
    Args:
        file_path: Path to the document to ingest
        chunking_strategy: "fixed" or "recursive"
        chunk_size: Size of chunks (passed to chunker)
        overlap: Overlap between chunks (passed to chunker)
        
    Returns:
        List of Chunk objects ready for embedding
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file type is not supported or chunking_strategy is invalid
        
    TODO: Implement document ingestion pipeline
    - Determine file type from extension
    - Call appropriate text extraction function
    - Call appropriate chunking function based on strategy
    - Return list of Chunks with all metadata fields populated
    - Add error handling and logging
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    
    if chunking_strategy not in ["fixed", "recursive"]:
        raise ValueError(f"Invalid chunking_strategy: {chunking_strategy}. Must be 'fixed' or 'recursive'.")
    
    logger.info(f"Ingesting document: {file_path}")
    
    # Extract text based on file type
    if file_path.suffix.lower() == ".pdf":
        full_text, page_info = extract_text_from_pdf(str(file_path))
        text = full_text
    elif file_path.suffix.lower() in [".txt", ".md"]:
        text = extract_text_from_file(str(file_path))
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}. Supported types: .pdf, .txt, .md")
    
    # Apply chunking strategy
    if chunking_strategy == "fixed":
        chunks = chunk_text_fixed_size(text, file_path.name, chunk_size, overlap)
    else:  # recursive
        chunks = chunk_text_recursive(text, file_path.name, chunk_size, overlap)
    
    logger.info(f"Ingestion complete: {len(chunks)} chunks created from {file_path.name}")
    return chunks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Ingestion module loaded. Awaiting implementation of extraction/chunking logic.")
