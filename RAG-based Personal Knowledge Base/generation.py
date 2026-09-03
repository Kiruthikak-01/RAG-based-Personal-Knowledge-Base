"""
Generation module for prompt construction and LLM-based answer generation.

This module handles:
- Building system and user prompts with context and citations
- LLM API calls (Claude or OpenAI, configurable)
- Mapping cited chunk IDs back to source metadata for display
- Streaming LLM responses token-by-token
"""

from typing import List, Dict, Iterator, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


def build_system_prompt() -> str:
    """
    Build the system prompt that instructs the LLM on its role and constraints.
    
    The system prompt should:
    1. Describe the assistant's role: answer questions based on retrieved context
    2. Include explicit instruction: "Say 'I don't know' if the answer is not in the context"
    3. Instruct on citation format (e.g., reference chunks by ID with [CITE: chunk_id])
    4. Define output format expectations
    
    Returns:
        System prompt string
        
    TODO: Implement system prompt
    - Define role and constraints
    - Include "I don't know" instruction
    - Specify citation format (must be parseable to extract chunk IDs later)
    - Recommend: return a clear, concise system message
    """
    return """You are a helpful assistant that answers questions based on provided context.

IMPORTANT INSTRUCTIONS:
1. Answer the user's question based ONLY on the provided context.
2. If the answer is not in the context, say "I don't know" or "This information is not in the provided context."
3. When citing information from the context, use the format [CITE: chunk_id].
4. Be concise and direct in your answers.
5. Always provide citations for factual claims.
"""


def build_user_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    """
    Build the user prompt that includes the query and retrieved context.
    
    This function should:
    1. Format retrieved chunks into a readable context section
    2. Include chunk IDs and metadata (source, page number) in a parseable format
    3. Append the user's query
    4. Make it clear where context ends and query begins
    
    Args:
        query: User's question
        retrieved_chunks: List of chunks retrieved via retrieval module
                         Each has: text, chunk_id, metadata (source_file, page_number, etc.)
        
    Returns:
        Formatted user prompt string
        
    Example format:
        ---
        CONTEXT:
        
        [CITE: chunk_id_1] Source: notes.pdf (Page 3)
        <chunk text>
        
        [CITE: chunk_id_2] Source: notes.pdf (Page 5)
        <chunk text>
        
        ---
        
        QUESTION:
        <user query>
        ---
        
    TODO: Implement user prompt construction
    - Iterate through retrieved_chunks
    - Format each chunk with [CITE: chunk_id] and metadata
    - Append query at the end
    - Return formatted string
    """
    context_text = "CONTEXT:\n\n"
    
    for chunk in retrieved_chunks:
        chunk_id = chunk.get("chunk_id", "unknown")
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        source_file = metadata.get("source_file", "unknown")
        page_number = metadata.get("page_number", None)
        
        # Build metadata line
        if page_number:
            metadata_line = f"[CITE: {chunk_id}] Source: {source_file} (Page {page_number})"
        else:
            metadata_line = f"[CITE: {chunk_id}] Source: {source_file}"
        
        context_text += f"{metadata_line}\n{text}\n\n"
    
    user_prompt = f"""{context_text}---

QUESTION:
{query}

---"""
    
    return user_prompt


def call_llm_api(
    system_prompt: str,
    user_prompt: str,
    llm_provider: str = "claude",
    model_name: str = "claude-3-sonnet-20240229",
    max_tokens: int = 1024,
    temperature: float = 0.3,
    stream: bool = False
) -> str | Iterator[str]:
    """
    Call the LLM API (Claude or OpenAI) with the given prompts.
    
    This function should:
    1. Read API key from environment variable (ANTHROPIC_API_KEY or OPENAI_API_KEY)
    2. Initialize the appropriate API client
    3. Make an API call with the prompts
    4. Return the response (or iterator if streaming)
    
    Args:
        system_prompt: System prompt (role and constraints)
        user_prompt: User prompt with context and query
        llm_provider: "claude" or "openai"
        model_name: Model identifier
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        stream: If True, return an iterator of tokens; else return full response
        
    Returns:
        str: Full response text (if stream=False)
        Iterator[str]: Token-by-token iterator (if stream=True)
        
    Raises:
        ValueError: If llm_provider not supported
        RuntimeError: If API key not found in environment
        
    TODO: Implement LLM API calls
    - Check llm_provider is "claude" or "openai"
    - Read API key from environment
    - Initialize appropriate client (anthropic.Anthropic or openai.OpenAI)
    - Make message-based API call with system + user prompts
    - If stream=True: return streaming iterator
    - If stream=False: collect full response and return
    - Handle API errors with informative messages
    """
    import os
    
    if llm_provider == "claude":
        from anthropic import Anthropic
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not found in environment variables")
        
        client = Anthropic(api_key=api_key)
        
        if stream:
            # Return a generator for streaming
            def stream_tokens():
                with client.messages.stream(
                    model=model_name,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=temperature
                ) as stream:
                    for text in stream.text_stream:
                        yield text
            return stream_tokens()
        else:
            # Non-streaming mode
            message = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature
            )
            return message.content[0].text
    
    elif llm_provider == "openai":
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not found in environment variables")
        
        client = OpenAI(api_key=api_key)
        
        if stream:
            # Return a generator for streaming
            def stream_tokens():
                stream = client.chat.completions.create(
                    model=model_name,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=temperature,
                    stream=True
                )
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            return stream_tokens()
        else:
            # Non-streaming mode
            response = client.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature
            )
            return response.choices[0].message.content
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}. Use 'claude' or 'openai'.")


def extract_citations_from_response(
    response: str,
    chunks: List[Dict[str, Any]]
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Extract cited chunk IDs from the LLM response and map to source metadata.
    
    This function should:
    1. Parse the response to find all citation references (e.g., [CITE: chunk_id_1])
    2. For each cited chunk_id, retrieve the chunk metadata from the chunks list
    3. Build a list of citations with: source_file, page_number, chunk_id
    4. Return the response with a list of source citations
    
    Args:
        response: Raw LLM response (containing [CITE: ...] references)
        chunks: List of original retrieved chunks (for metadata lookup)
        
    Returns:
        Tuple of:
        - cleaned_response: Response text (optionally with citations formatted)
        - citations: List of dicts with source_file, page_number, chunk_id
        
    Example:
        Input response: "According to the notes [CITE: chunk_1], the answer is X."
        Output: ("According to the notes [1], the answer is X.",
                 [{"source_file": "notes.pdf", "page_number": 3, "chunk_id": "chunk_1", "text": "..."}])
        
    TODO: Implement citation extraction
    - Find all [CITE: <chunk_id>] patterns in response using regex
    - For each chunk_id, find metadata in chunks list
    - Map [CITE: chunk_id] to numbered citations [1], [2], etc.
    - Return cleaned response and citation metadata
    """
    import re
    
    # Find all [CITE: chunk_id] patterns
    cite_pattern = r"\[CITE:\s*([^\]]+)\]"
    citations_found = re.findall(cite_pattern, response)
    
    # Build citation metadata
    citations = []
    chunk_id_to_number = {}
    cleaned_response = response
    
    for i, chunk_id in enumerate(citations_found, 1):
        chunk_id = chunk_id.strip()
        if chunk_id not in chunk_id_to_number:
            chunk_id_to_number[chunk_id] = i
            
            # Find chunk metadata
            chunk_meta = None
            for chunk in chunks:
                if chunk.get("chunk_id") == chunk_id:
                    chunk_meta = chunk.get("metadata", {})
                    break
            
            if chunk_meta:
                citations.append({
                    "chunk_id": chunk_id,
                    "source_file": chunk_meta.get("source_file", "unknown"),
                    "page_number": chunk_meta.get("page_number", None),
                    "number": i
                })
    
    # Replace [CITE: chunk_id] with [number]
    for chunk_id, number in chunk_id_to_number.items():
        cleaned_response = re.sub(
            rf"\[CITE:\s*{re.escape(chunk_id)}\]",
            f"[{number}]",
            cleaned_response
        )
    
    return cleaned_response, citations


def generate_with_citations(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    llm_provider: str = "claude",
    model_name: str = "claude-3-sonnet-20240229",
    max_tokens: int = 1024,
    temperature: float = 0.3
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Main orchestration: generate answer with citations.
    
    This function should:
    1. Build system and user prompts
    2. Call LLM API
    3. Extract citations from response
    4. Return answer and structured citations
    
    Args:
        query: User's question
        retrieved_chunks: List of retrieved chunks
        llm_provider: LLM provider ("claude" or "openai")
        model_name: Model name
        max_tokens: Max tokens to generate
        temperature: LLM temperature
        
    Returns:
        Tuple of:
        - answer: Generated response (with citations formatted)
        - citations: List of dicts with source metadata
        
    TODO: Implement answer generation with citations
    - Call build_system_prompt()
    - Call build_user_prompt()
    - Call call_llm_api() with stream=False
    - Call extract_citations_from_response()
    - Return answer and citations
    """
    logger.info(f"Generating answer for query: {query}")
    
    # Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(query, retrieved_chunks)
    
    # Call LLM
    response = call_llm_api(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm_provider=llm_provider,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False
    )
    
    # Extract citations
    cleaned_response, citations = extract_citations_from_response(response, retrieved_chunks)
    
    logger.info(f"Generated answer with {len(citations)} citations")
    return cleaned_response, citations


def generate_with_citations_streaming(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    llm_provider: str = "claude",
    model_name: str = "claude-3-sonnet-20240229",
    max_tokens: int = 1024,
    temperature: float = 0.3
) -> Iterator[str]:
    """
    Stream answer generation token-by-token.
    
    Yields tokens as they arrive from the LLM. Useful for real-time UI updates.
    
    Args:
        query: User's question
        retrieved_chunks: List of retrieved chunks
        llm_provider: LLM provider
        model_name: Model name
        max_tokens: Max tokens to generate
        temperature: LLM temperature
        
    Yields:
        String tokens as they arrive
        
    TODO: Implement streaming generation
    - Build system and user prompts
    - Call call_llm_api() with stream=True
    - Yield tokens as they arrive
    """
    logger.info(f"Streaming answer for query: {query}")
    
    # Build prompts
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(query, retrieved_chunks)
    
    # Call LLM with streaming
    token_stream = call_llm_api(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm_provider=llm_provider,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    )
    
    # Yield tokens
    for token in token_stream:
        yield token


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Generation module loaded. Awaiting implementation of prompt and LLM logic.")
