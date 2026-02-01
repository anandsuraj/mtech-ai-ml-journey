"""
Document Preprocessing Module for Hybrid RAG System
Handles text chunking with overlap and metadata management
"""

import os
import sys
import json
import pickle
import tiktoken
from typing import List, Dict
import nltk
from nltk.tokenize import sent_tokenize

# Fix imports - add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        encoding_name: Name of the encoding to use
        
    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_text_with_overlap(
    text: str,
    min_tokens: int = config.MIN_CHUNK_TOKENS,
    max_tokens: int = config.MAX_CHUNK_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS
) -> List[str]:
    """
    Chunk text with token-based splitting and overlap.
    Respects sentence boundaries when possible.
    
    Args:
        text: Text to chunk
        min_tokens: Minimum tokens per chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Split into sentences first
    sentences = split_into_sentences(text)
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    encoding = tiktoken.get_encoding("cl100k_base")
    
    for sentence in sentences:
        sentence_tokens = len(encoding.encode(sentence))
        
        # If current chunk + this sentence exceeds max, save current chunk
        if current_tokens + sentence_tokens > max_tokens and current_chunk:
            # Only save if we have at least min_tokens
            if current_tokens >= min_tokens:
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk)
                overlap_chunk_tokens = len(encoding.encode(overlap_text))
                
                # Take sentences from the end for overlap
                overlap_sentences = []
                overlap_token_count = 0
                for s in reversed(current_chunk):
                    s_tokens = len(encoding.encode(s))
                    if overlap_token_count + s_tokens <= overlap_tokens:
                        overlap_sentences.insert(0, s)
                        overlap_token_count += s_tokens
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_tokens = overlap_token_count
            else:
                # Current chunk is too small, keep adding
                pass
        
        current_chunk.append(sentence)
        current_tokens += sentence_tokens
    
    # Add the last chunk if it meets minimum requirements
    if current_chunk and current_tokens >= min_tokens:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using simple heuristics.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    import re
    
    # Simple sentence splitting on periods, question marks, exclamation marks
    # followed by whitespace and a capital letter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    # Filter out empty strings
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def create_chunks_with_metadata(corpus: List[Dict[str, str]]) -> List[Dict]:
    """
    Create chunks from corpus with metadata.
    
    Args:
        corpus: List of article dictionaries with 'title', 'content', 'url'
        
    Returns:
        List of chunk dictionaries with metadata
    """
    all_chunks = []
    chunk_id = 0
    
    for article in corpus:
        title = article['title']
        content = article['content']
        url = article['url']
        
        # Skip articles with no content
        if not content:
            continue
        
        # Chunk the content
        chunks = chunk_text_with_overlap(content)
        
        # Add metadata to each chunk
        for position, chunk_text in enumerate(chunks):
            chunk_data = {
                'chunk_id': f"chunk_{chunk_id}",
                'text': chunk_text,
                'title': title,
                'url': url,
                'position': position,  # Position within the article
                'total_chunks': len(chunks),  # Total chunks for this article
                'token_count': count_tokens(chunk_text)
            }
            all_chunks.append(chunk_data)
            chunk_id += 1
    
    return all_chunks


def save_chunks(chunks: List[Dict], filepath: str = config.CHUNKS_FILE):
    """
    Save chunks to JSON file.
    
    Args:
        chunks: List of chunk dictionaries
        filepath: Path to save the JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(chunks)} chunks to {filepath}")


def load_chunks(filepath: str = config.CHUNKS_FILE) -> List[Dict]:
    """
    Load chunks from JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of chunk dictionaries
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    return chunks


def preprocess_corpus(corpus: List[Dict[str, str]]) -> List[Dict]:
    """
    Full preprocessing pipeline: take corpus and create chunks with metadata.
    
    Args:
        corpus: List of article dictionaries
        
    Returns:
        List of chunk dictionaries
    """
    print(f"Preprocessing {len(corpus)} articles...")
    
    # Create chunks
    chunks = create_chunks_with_metadata(corpus)
    
    # Print statistics
    print(f"\nPreprocessing Statistics:")
    print(f"  Total articles: {len(corpus)}")
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Avg chunks per article: {len(chunks) / len(corpus):.2f}")
    
    token_counts = [c['token_count'] for c in chunks]
    print(f"  Avg tokens per chunk: {sum(token_counts) / len(token_counts):.2f}")
    print(f"  Min tokens per chunk: {min(token_counts)}")
    print(f"  Max tokens per chunk: {max(token_counts)}")
    
    # Save chunks
    save_chunks(chunks)
    
    return chunks


if __name__ == "__main__":
    # Test preprocessing on existing corpus
    import pickle
    
    # Load corpus
    with open(config.CORPUS_FILE, 'rb') as f:
        corpus = pickle.load(f)
    
    # Preprocess
    chunks = preprocess_corpus(corpus)
    
    print(f"\nDone! Chunks saved to {config.CHUNKS_FILE}")
    print("\n" + "="*70)
    print("Next step: python src/hybrid_retrieval.py")
    print("="*70)
