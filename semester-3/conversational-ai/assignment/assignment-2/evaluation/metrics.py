"""
Evaluation Metrics for RAG System
Implements MRR (mandatory), NDCG@K, and BERTScore (custom metrics)
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import ndcg_score
from bert_score import score as bert_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class EvaluationMetrics:
    """Calculate evaluation metrics for RAG system."""
    
    def __init__(self):
        """Initialize metrics calculator."""
        pass
    
    def calculate_mrr_url(
        self,
        ground_truth_urls: List[str],
        retrieved_chunks: List[Tuple[Dict, float]],
        cutoff: int = config.MRR_CUTOFF
    ) -> float:
        """
        Calculate Mean Reciprocal Rank at URL level (not chunk level).
        This is the MANDATORY metric as per assignment.
        
        Formula: MRR = 1/rank where rank is the position of the first correct URL
        
        Args:
            ground_truth_urls: List of correct Wikipedia URLs
            retrieved_chunks: List of (chunk_dict, score) tuples from retrieval
            cutoff: Maximum rank to consider
            
        Returns:
            Reciprocal rank (1/rank) or 0 if not found in top cutoff
        
        Interpretation:
            1.0 = correct URL at rank 1 (perfect)
            0.5 = correct URL at rank 2
            0.1 = correct URL at rank 10
            0.0 = correct URL not in top cutoff results
        """
        # Extract URLs from retrieved chunks
        retrieved_urls = [chunk['url'] for chunk, _ in retrieved_chunks[:cutoff]]
        
        # Find rank of first correct URL
        for rank, url in enumerate(retrieved_urls, 1):
            if url in ground_truth_urls:
                return 1.0 / rank
        
        return 0.0
    
    def calculate_ndcg_at_k(
        self,
        ground_truth_urls: List[str],
        retrieved_chunks: List[Tuple[Dict, float]],
        k: int = config.NDCG_K
    ) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.
        This is CUSTOM METRIC #1.
        
        Justification:
            NDCG measures ranking quality by considering both relevance and position.
            Unlike MRR which only considers the first relevant result, NDCG evaluates
            the entire ranking, making it ideal for assessing retrieval quality.
        
        Calculation:
            DCG@K = Σ(i=1 to K) (2^rel_i - 1) / log2(i + 1)
            IDCG@K = DCG for perfect ranking
            NDCG@K = DCG@K / IDCG@K
        
        Interpretation:
            1.0 = perfect ranking (all relevant docs at top in ideal order)
            0.7-0.9 = good ranking (most relevant docs near top)
            0.5-0.7 = fair ranking (some relevant docs scattered)
            <0.5 = poor ranking (relevant docs buried or missing)
        
        Args:
            ground_truth_urls: List of correct Wikipedia URLs
            retrieved_chunks: List of (chunk_dict, score) tuples
            k: Number of top results to consider
            
        Returns:
            NDCG@K score between 0 and 1
        """
        # Create relevance scores (1 if URL matches ground truth, 0 otherwise)
        relevance_scores = []
        for chunk, _ in retrieved_chunks[:k]:
            relevance = 1 if chunk['url'] in ground_truth_urls else 0
            relevance_scores.append(relevance)
        
        # If no relevant documents, return 0
        if sum(relevance_scores) == 0:
            return 0.0
        
        # Calculate NDCG using sklearn
        # Reshape for sklearn format
        y_true = np.array([relevance_scores])
        y_score = np.array([list(range(k, 0, -1))])  # Higher scores for higher ranks
        
        try:
            ndcg = ndcg_score(y_true, y_score)
            return float(ndcg)
        except:
            return 0.0
    
    def calculate_bertscore(
        self,
        reference_answer: str,
        generated_answer: str
    ) -> Dict[str, float]:
        """
        Calculate BERTScore for answer semantic similarity.
        This is CUSTOM METRIC #2.
        
        Justification:
            BERTScore measures semantic similarity beyond exact word matching.
            Unlike BLEU or ROUGE which rely on n-gram overlap, BERTScore uses
            contextual embeddings to capture meaning, making it ideal for
            evaluating generated answers that may use different words but
            convey the same meaning.
        
        Calculation:
            - Compute BERT embeddings for reference and generated answer tokens
            - Calculate token-level cosine similarity
            - Apply greedy matching to find best alignment
            - Compute precision, recall, and F1
        
        Interpretation:
            >0.9 = excellent semantic match (nearly identical meaning)
            0.8-0.9 = good match (similar meaning, different wording)
            0.7-0.8 = fair match (related but some divergence)
            0.6-0.7 = weak match (loosely related)
            <0.6 = poor match (different meaning)
        
        Args:
            reference_answer: Ground truth answer
            generated_answer: System-generated answer
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Calculate BERTScore
        P, R, F1 = bert_score(
            [generated_answer],
            [reference_answer],
            lang='en',
            verbose=False
        )
        
        return {
            'precision': float(P[0]),
            'recall': float(R[0]),
            'f1': float(F1[0])
        }
    
    def evaluate_single_question(
        self,
        question_data: Dict,
        retrieved_chunks: List[Tuple[Dict, float]],
        generated_answer: str
    ) -> Dict:
        """
        Evaluate a single question with all metrics.
        
        Args:
            question_data: Question dictionary with ground truth
            retrieved_chunks: Retrieved chunks from RAG system
            generated_answer: Answer generated by LLM
            
        Returns:
            Dictionary with all metric scores
        """
        # Extract ground truth URLs (handle both single and multiple)
        if isinstance(question_data['source_url'], list):
            ground_truth_urls = question_data['source_url']
        else:
            ground_truth_urls = [question_data['source_url']]
        
        # Calculate MRR (mandatory metric)
        mrr = self.calculate_mrr_url(ground_truth_urls, retrieved_chunks)
        
        # Calculate NDCG@K (custom metric 1)
        ndcg = self.calculate_ndcg_at_k(ground_truth_urls, retrieved_chunks)
        
        # Calculate BERTScore (custom metric 2)
        bert_scores = self.calculate_bertscore(
            question_data['answer'],
            generated_answer
        )
        
        return {
            'mrr': mrr,
            'ndcg_at_k': ndcg,
            'bertscore_precision': bert_scores['precision'],
            'bertscore_recall': bert_scores['recall'],
            'bertscore_f1': bert_scores['f1']
        }


if __name__ == "__main__":
    # Test metrics
    metrics = EvaluationMetrics()
    
    # Mock data for testing
    ground_truth_urls = ["https://en.wikipedia.org/wiki/Machine_learning"]
    
    retrieved_chunks = [
        ({'url': 'https://en.wikipedia.org/wiki/Artificial_intelligence', 'text': 'AI text'}, 0.9),
        ({'url': 'https://en.wikipedia.org/wiki/Machine_learning', 'text': 'ML text'}, 0.8),
        ({'url': 'https://en.wikipedia.org/wiki/Deep_learning', 'text': 'DL text'}, 0.7),
    ]
    
    # Test MRR
    mrr = metrics.calculate_mrr_url(ground_truth_urls, retrieved_chunks)
    print(f"MRR: {mrr:.4f} (correct URL at rank 2, so 1/2 = 0.5)")
    
    # Test NDCG
    ndcg = metrics.calculate_ndcg_at_k(ground_truth_urls, retrieved_chunks, k=3)
    print(f"NDCG@3: {ndcg:.4f}")
    
    # Test BERTScore
    reference = "Machine learning is a subset of artificial intelligence."
    generated = "ML is part of AI and focuses on learning from data."
    bert_scores = metrics.calculate_bertscore(reference, generated)
    print(f"BERTScore F1: {bert_scores['f1']:.4f}")
