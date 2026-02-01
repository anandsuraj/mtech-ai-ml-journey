"""
Innovative Evaluation Features
Includes ablation studies and error analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, List
import config


def run_ablation_study(retriever, generator) -> Dict:
    """
    Run ablation study comparing dense-only, sparse-only, and hybrid retrieval.
    
    Args:
        retriever: HybridRetriever instance
        generator: ResponseGenerator instance
    
    Returns:
        Dictionary with ablation study results
    """
    print("\nRunning ablation study...")
    
    # Load questions
    with open(config.QUESTIONS_FILE, 'r') as f:
        questions_data = json.load(f)
    
    # Sample 20 questions for ablation (to save time)
    import random
    sample_questions = random.sample(questions_data, min(20, len(questions_data)))
    
    results = {
        'dense_only': {'correct': 0, 'total': 0},
        'sparse_only': {'correct': 0, 'total': 0},
        'hybrid': {'correct': 0, 'total': 0}
    }
    
    for i, q in enumerate(sample_questions):
        print(f"  Testing question {i+1}/{len(sample_questions)}...", end='\r')
        
        question = q['question']
        source_url = q['source_url']
        
        # Test dense-only
        try:
            dense_results = retriever.dense_retrieval(question, k=5)
            dense_urls = [r['url'] for r in dense_results]
            if isinstance(source_url, list):
                results['dense_only']['correct'] += any(url in dense_urls for url in source_url)
            else:
                results['dense_only']['correct'] += (source_url in dense_urls)
            results['dense_only']['total'] += 1
        except:
            pass
        
        # Test sparse-only
        try:
            sparse_results = retriever.sparse_retrieval(question, k=5)
            sparse_urls = [r['url'] for r in sparse_results]
            if isinstance(source_url, list):
                results['sparse_only']['correct'] += any(url in sparse_urls for url in source_url)
            else:
                results['sparse_only']['correct'] += (source_url in sparse_urls)
            results['sparse_only']['total'] += 1
        except:
            pass
        
        # Test hybrid
        try:
            hybrid_results = retriever.hybrid_search(question, k=5)
            hybrid_urls = [r['url'] for r in hybrid_results]
            if isinstance(source_url, list):
                results['hybrid']['correct'] += any(url in hybrid_urls for url in source_url)
            else:
                results['hybrid']['correct'] += (source_url in hybrid_urls)
            results['hybrid']['total'] += 1
        except:
            pass
    
    print()  # New line after progress
    
    # Calculate accuracy
    for method in results:
        if results[method]['total'] > 0:
            results[method]['accuracy'] = results[method]['correct'] / results[method]['total']
        else:
            results[method]['accuracy'] = 0.0
    
    print(f"  Dense-only accuracy: {results['dense_only']['accuracy']:.3f}")
    print(f"  Sparse-only accuracy: {results['sparse_only']['accuracy']:.3f}")
    print(f"  Hybrid accuracy: {results['hybrid']['accuracy']:.3f}")
    
    return results


def analyze_errors(results: Dict) -> Dict:
    """
    Analyze errors by question type and failure mode.
    
    Args:
        results: Evaluation results dictionary
    
    Returns:
        Dictionary with error analysis
    """
    print("\nAnalyzing errors by question type...")
    
    error_analysis = {
        'by_type': {
            'factual': {'total': 0, 'failed': 0},
            'comparative': {'total': 0, 'failed': 0},
            'inferential': {'total': 0, 'failed': 0},
            'multi_hop': {'total': 0, 'failed': 0}
        },
        'failure_modes': {
            'retrieval_failure': 0,
            'generation_failure': 0,
            'both_failure': 0
        }
    }
    
    # Analyze per-question results
    if 'per_question_results' in results:
        for q_result in results['per_question_results']:
            q_type = q_result.get('type', 'unknown')
            
            if q_type in error_analysis['by_type']:
                error_analysis['by_type'][q_type]['total'] += 1
                
                # Check if retrieval failed (MRR = 0)
                retrieval_failed = q_result.get('mrr_url', 0) == 0
                
                # Check if generation failed (BERTScore < 0.5)
                generation_failed = q_result.get('bertscore_f1', 0) < 0.5
                
                if retrieval_failed and generation_failed:
                    error_analysis['by_type'][q_type]['failed'] += 1
                    error_analysis['failure_modes']['both_failure'] += 1
                elif retrieval_failed:
                    error_analysis['by_type'][q_type]['failed'] += 1
                    error_analysis['failure_modes']['retrieval_failure'] += 1
                elif generation_failed:
                    error_analysis['by_type'][q_type]['failed'] += 1
                    error_analysis['failure_modes']['generation_failure'] += 1
    
    # Calculate failure rates
    for q_type in error_analysis['by_type']:
        total = error_analysis['by_type'][q_type]['total']
        if total > 0:
            failed = error_analysis['by_type'][q_type]['failed']
            error_analysis['by_type'][q_type]['failure_rate'] = failed / total
        else:
            error_analysis['by_type'][q_type]['failure_rate'] = 0.0
    
    print("  Error rates by question type:")
    for q_type, stats in error_analysis['by_type'].items():
        if stats['total'] > 0:
            print(f"    {q_type}: {stats['failure_rate']:.1%} ({stats['failed']}/{stats['total']})")
    
    return error_analysis


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("Use run_evaluation.py to run the complete evaluation pipeline.")
