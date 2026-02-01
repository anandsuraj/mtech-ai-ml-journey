"""
Sparse Keyword Retrieval using BM25
"""

import os
import sys
import json
import pickle
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

# Fix imports - add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SparseRetriever:
    """Sparse keyword retrieval using BM25 algorithm."""
    
    def __init__(self):
        """Initialize the sparse retriever."""
        self.bm25 = None
        self.chunks = None
        self.tokenized_chunks = None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization by splitting on whitespace and lowercasing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        return text.lower().split()
    
    def build_index(self, chunks: List[Dict]):
        """
        Build BM25 index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        print(f"Building BM25 index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        
        # Tokenize all chunks
        self.tokenized_chunks = [self.tokenize(chunk['text']) for chunk in chunks]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_chunks)
        
        print(f"BM25 index built with {len(self.tokenized_chunks)} documents")
    
    def search(self, query: str, top_k: int = config.SPARSE_TOP_K) -> List[Tuple[Dict, float]]:
        """
        Search for relevant chunks using BM25.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_dict, bm25_score) tuples
        """
        if self.bm25 is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = scores.argsort()[-top_k:][::-1]
        
        # Prepare results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self.chunks[idx].copy()
            chunk['sparse_score'] = float(scores[idx])
            chunk['rank'] = rank
            results.append((chunk, float(scores[idx])))
        
        return results
    
    def save_index(self, index_path: str = config.BM25_INDEX_FILE):
        """
        Save BM25 index and metadata.
        
        Args:
            index_path: Path to save the index
        """
        data = {
            'bm25': self.bm25,
            'chunks': self.chunks,
            'tokenized_chunks': self.tokenized_chunks
        }
        
        with open(index_path, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"BM25 index saved to {index_path}")
    
    def load_index(self, index_path: str = config.BM25_INDEX_FILE):
        """
        Load BM25 index and metadata.
        
        Args:
            index_path: Path to the index file
        """
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.chunks = data['chunks']
        self.tokenized_chunks = data['tokenized_chunks']
        
        print(f"BM25 index loaded from {index_path}")


if __name__ == "__main__":
    # Test sparse retrieval
    from preprocessing import load_chunks
    
    # Load chunks
    chunks = load_chunks()
    
    # Build index
    retriever = SparseRetriever()
    retriever.build_index(chunks)
    
    # Save index
    retriever.save_index()
    
    # Test search
    query = "What is machine learning?"
    results = retriever.search(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print("\nTop 5 BM25 results:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. BM25 Score: {score:.4f}")
        print(f"   Title: {chunk['title']}")
        print(f"   URL: {chunk['url']}")
        print(f"   Text: {chunk['text'][:200]}...")
