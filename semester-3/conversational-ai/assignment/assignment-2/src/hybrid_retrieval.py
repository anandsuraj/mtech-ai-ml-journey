# Hybrid retrieval combining dense and sparse methods with RRF
# Reciprocal Rank Fusion as per assignment formula

import os
import sys
from typing import List, Dict, Tuple
import pickle

# Fix imports - add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# macOS Threading Safety
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import config
from src.embeddings import DenseRetriever
from src.sparse_retrieval import SparseRetriever


class HybridRetriever:
    """Hybrid retrieval combining dense and sparse methods using RRF."""
    
    def __init__(self):
        """Initialize hybrid retriever with dense and sparse retrievers."""
        self.dense_retriever = DenseRetriever()
        self.sparse_retriever = SparseRetriever()
    
    def build_indices(self, chunks: List[Dict]):
        """
        Build both dense and sparse indices.
        
        Args:
            chunks: List of chunk dictionaries
        """
        print("Building hybrid retrieval system...")
        self.dense_retriever.build_index(chunks)
        self.sparse_retriever.build_index(chunks)
        print("Hybrid retrieval system ready!")
    
    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Dict, float]],
        sparse_results: List[Tuple[Dict, float]],
        k: int = config.RRF_K
    ) -> List[Tuple[Dict, float]]:
        """
        Combine results using Reciprocal Rank Fusion.
        Formula: RRF_score(d) = Σ 1/(k + rank_i(d)) where k=60 (as per assignment)
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval
            k: RRF constant (default 60 as per assignment)
            
        Returns:
            List of (chunk_dict, rrf_score) tuples sorted by RRF score
        """
        # Create a dictionary to store RRF scores
        rrf_scores = {}
        
        # Process dense results
        for rank, (chunk, score) in enumerate(dense_results, 1):
            chunk_id = chunk['chunk_id']
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    'chunk': chunk,
                    'rrf_score': 0,
                    'dense_rank': None,
                    'sparse_rank': None,
                    'dense_score': None,
                    'sparse_score': None
                }
            
            # Add RRF contribution from dense retrieval
            rrf_scores[chunk_id]['rrf_score'] += 1 / (k + rank)
            rrf_scores[chunk_id]['dense_rank'] = rank
            rrf_scores[chunk_id]['dense_score'] = score
        
        # Process sparse results
        for rank, (chunk, score) in enumerate(sparse_results, 1):
            chunk_id = chunk['chunk_id']
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {
                    'chunk': chunk,
                    'rrf_score': 0,
                    'dense_rank': None,
                    'sparse_rank': None,
                    'dense_score': None,
                    'sparse_score': None
                }
            
            # Add RRF contribution from sparse retrieval
            rrf_scores[chunk_id]['rrf_score'] += 1 / (k + rank)
            rrf_scores[chunk_id]['sparse_rank'] = rank
            rrf_scores[chunk_id]['sparse_score'] = score
        
        # Sort by RRF score
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        # Prepare output format
        results = []
        for item in sorted_results:
            chunk = item['chunk'].copy()
            chunk['rrf_score'] = item['rrf_score']
            chunk['dense_rank'] = item['dense_rank']
            chunk['sparse_rank'] = item['sparse_rank']
            chunk['dense_score'] = item['dense_score']
            chunk['sparse_score'] = item['sparse_score']
            results.append((chunk, item['rrf_score']))
        
        return results
    
    def search(
        self,
        query: str,
        dense_k: int = config.DENSE_TOP_K,
        sparse_k: int = config.SPARSE_TOP_K,
        final_top_n: int = config.FINAL_TOP_N
    ) -> Tuple[List[Tuple[Dict, float]], Dict]:
        """
        Hybrid search using RRF.
        
        Args:
            query: Query text
            dense_k: Number of results from dense retrieval
            sparse_k: Number of results from sparse retrieval
            final_top_n: Final number of results to return after RRF
            
        Returns:
            Tuple of (top_n_results, metadata) where metadata contains individual retrieval results
        """
        # Get dense results
        dense_results = self.dense_retriever.search(query, top_k=dense_k)
        
        # Get sparse results
        sparse_results = self.sparse_retriever.search(query, top_k=sparse_k)
        
        # Apply RRF
        rrf_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Take top N
        top_n_results = rrf_results[:final_top_n]
        
        # Prepare metadata
        metadata = {
            'dense_results': dense_results,
            'sparse_results': sparse_results,
            'rrf_results': rrf_results,
            'dense_k': dense_k,
            'sparse_k': sparse_k,
            'final_top_n': final_top_n
        }
        
        return top_n_results, metadata
    
    def save_indices(
        self,
        vector_path: str = config.VECTOR_INDEX_FILE,
        bm25_path: str = config.BM25_INDEX_FILE
    ):
        """
        Save both dense and sparse indices.
        
        Args:
            vector_path: Path for vector index
            bm25_path: Path for BM25 index
        """
        self.dense_retriever.save_index(vector_path)
        self.sparse_retriever.save_index(bm25_path)
    
    def load_indices(
        self,
        vector_path: str = config.VECTOR_INDEX_FILE,
        bm25_path: str = config.BM25_INDEX_FILE
    ):
        """
        Load both dense and sparse indices.
        
        Args:
            vector_path: Path to vector index
            bm25_path: Path to BM25 index
        """
        self.dense_retriever.load_index(vector_path)
        self.sparse_retriever.load_index(bm25_path)


if __name__ == "__main__":
    # Test hybrid retrieval
    from preprocessing import load_chunks
    
    # Load chunks
    chunks = load_chunks()
    
    # Build hybrid retrieval system
    retriever = HybridRetriever()
    retriever.build_indices(chunks)
    
    # Save indices
    retriever.save_indices()
    
    # Test search
    query = "What is machine learning?"
    results, metadata = retriever.search(query)
    
    print(f"\nQuery: {query}")
    print(f"\nTop {len(results)} RRF results:")
    for i, (chunk, rrf_score) in enumerate(results, 1):
        print(f"\n{i}. RRF Score: {rrf_score:.6f}")
        print(f"   Dense Rank: {chunk.get('dense_rank', 'N/A')}, Score: {chunk.get('dense_score', 'N/A')}")
        print(f"   Sparse Rank: {chunk.get('sparse_rank', 'N/A')}, Score: {chunk.get('sparse_score', 'N/A')}")
        print(f"   Title: {chunk['title']}")
        print(f"   URL: {chunk['url']}")
        print(f"   Text: {chunk['text'][:150]}...")
