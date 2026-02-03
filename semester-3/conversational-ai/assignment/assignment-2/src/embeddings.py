"""
Dense Vector Retrieval using Sentence Transformers
"""

import os
import sys

# Fix imports - add parent directory to path FIRST
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import config

# CRITICAL: Set HuggingFace cache and threading safety for macOS
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = config.CACHE_DIR
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Prevent OpenMP runtime conflict on macOS
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid deadlock during initialization
# Now safe to import transformer libraries
if 'HF_TOKEN' in os.environ:
    print(f"DEBUG: Found HF_TOKEN in environment: {os.environ['HF_TOKEN'][:5]}...")
else:
    print("DEBUG: No HF_TOKEN found in environment (Good for public access)")
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss


class DenseRetriever:
    """Dense retrieval using sentence embeddings and FAISS."""
    
    def __init__(self, model_name: str = config.EMBEDDING_MODEL):
        """
        Initialize the dense retriever.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        print(f"Loading embedding model: {model_name}")
        # Force token=False to access public model without auth (avoids expired token errors)
        self.model = SentenceTransformer(model_name, token=False)
        self.index = None
        self.chunks = None
        self.embeddings = None
    
    def build_index(self, chunks: List[Dict]):
        """
        Build FAISS index from chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
        """
        print(f"Building dense index for {len(chunks)} chunks...")
        
        self.chunks = chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        print(f"Building FAISS index (dimension: {dimension})...")
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
        self.index.add(self.embeddings)
        
        print(f"Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = config.DENSE_TOP_K) -> List[Tuple[Dict, float]]:
        """
        Search for similar chunks using dense retrieval.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_dict, similarity_score) tuples
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], scores[0]):
            chunk = self.chunks[idx].copy()
            chunk['dense_score'] = float(score)
            chunk['rank'] = len(results) + 1
            results.append((chunk, float(score)))
        
        return results
    
    def save_index(self, index_path: str = config.VECTOR_INDEX_FILE):
        """
        Save FAISS index and metadata.
        
        Args:
            index_path: Path to save the index
        """
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata (chunks and embeddings)
        metadata_path = index_path.replace('.bin', '_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.embeddings
            }, f)
        
        print(f"Dense index saved to {index_path}")
    
    def load_index(self, index_path: str = config.VECTOR_INDEX_FILE):
        """
        Load FAISS index and metadata.
        
        Args:
            index_path: Path to the index file
        """
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        metadata_path = index_path.replace('.bin', '_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            self.chunks = metadata['chunks']
            self.embeddings = metadata['embeddings']
        
        print(f"Dense index loaded from {index_path}")


if __name__ == "__main__":
    # Test dense retrieval
    from preprocessing import load_chunks
    
    # Load chunks
    chunks = load_chunks()
    
    # Build index
    retriever = DenseRetriever()
    retriever.build_index(chunks)
    
    # Save index
    retriever.save_index()
    
    # Test search
    query = "What is machine learning?"
    results = retriever.search(query, top_k=5)
    
    print(f"\nQuery: {query}")
    print("\nTop 5 results:")
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. Score: {score:.4f}")
        print(f"   Title: {chunk['title']}")
        print(f"   URL: {chunk['url']}")
        print(f"   Text: {chunk['text'][:200]}...")
