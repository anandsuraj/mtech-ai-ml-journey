"""
Automated Evaluation Pipeline
Single-command pipeline to evaluate the RAG system
"""

import json
import time
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_retrieval import HybridRetriever
from src.llm_generation import ResponseGenerator
from evaluation.metrics import EvaluationMetrics
import config


class EvaluationPipeline:
    """Automated pipeline for RAG system evaluation."""
    
    def __init__(self):
        """Initialize evaluation pipeline."""
        print("Initializing evaluation pipeline...")
        
        # Load RAG system
        print("Loading RAG system...")
        self.retriever = HybridRetriever()
        self.retriever.load_indices()
        
        self.generator = ResponseGenerator()
        
        # Initialize metrics
        self.metrics_calculator = EvaluationMetrics()
        
        print("Pipeline initialized!")
    
    def load_questions(self, filepath: str = config.QUESTIONS_FILE) -> list:
        """Load test questions."""
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        return questions
    
    def evaluate_single_question(self, question_data: dict) -> dict:
        """Evaluate RAG system on a single question."""
        query = question_data['question']
        q_id = question_data.get('question_id', 'unknown')
        
        # Retrieval
        retrieval_start = time.time()
        retrieved_chunks, metadata = self.retriever.search(query)
        retrieval_time = time.time() - retrieval_start
        
        # Generation
        generation_start = time.time()
        result = self.generator.generate_answer(query, retrieved_chunks)
        generation_time = time.time() - generation_start
        
        # Calculate metrics
        metrics = self.metrics_calculator.evaluate_single_question(
            question_data,
            retrieved_chunks,
            result['answer']
        )
        
        # Combine results
        eval_result = {
            'question_id': q_id,
            'question': query,
            'question_type': question_data['type'],
            'ground_truth_answer': question_data['answer'],
            'generated_answer': result['answer'],
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time,
            **metrics,
            'retrieved_chunks': [
                {
                    'title': chunk['title'],
                    'url': chunk['url'],
                    'rrf_score': chunk.get('rrf_score', 0),
                    'dense_rank': chunk.get('dense_rank'),
                    'sparse_rank': chunk.get('sparse_rank')
                }
                for chunk, _ in retrieved_chunks
            ]
        }
        
        return eval_result
    
    def run_evaluation(self, questions: list) -> list:
        """Run evaluation on all questions."""
        results = []
        
        print(f"\nEvaluating {len(questions)} questions...")
        
        for i, question in enumerate(tqdm(questions)):
            try:
                result = self.evaluate_single_question(question)
                results.append(result)
            except Exception as e:
                q_id = question.get('question_id', i+1)
                print(f"\nError evaluating question {q_id}: {e}")
                continue
        
        return results
    
    def calculate_overall_metrics(self, results: list) -> dict:
        """Calculate overall performance metrics."""
        if not results:
            return {}
        
        metrics = {
            'total_questions': len(results),
            'avg_mrr': sum(r['mrr'] for r in results) / len(results),
            'avg_ndcg_at_k': sum(r['ndcg_at_k'] for r in results) / len(results),
            'avg_bertscore_f1': sum(r['bertscore_f1'] for r in results) / len(results),
            'avg_bertscore_precision': sum(r['bertscore_precision'] for r in results) / len(results),
            'avg_bertscore_recall': sum(r['bertscore_recall'] for r in results) / len(results),
            'avg_retrieval_time': sum(r['retrieval_time'] for r in results) / len(results),
            'avg_generation_time': sum(r['generation_time'] for r in results) / len(results),
            'avg_total_time': sum(r['total_time'] for r in results) / len(results)
        }
        
        # By question type
        by_type = {}
        for result in results:
            q_type = result['question_type']
            if q_type not in by_type:
                by_type[q_type] = []
            by_type[q_type].append(result)
        
        type_metrics = {}
        for q_type, type_results in by_type.items():
            type_metrics[q_type] = {
                'count': len(type_results),
                'avg_mrr': sum(r['mrr'] for r in type_results) / len(type_results),
                'avg_ndcg_at_k': sum(r['ndcg_at_k'] for r in type_results) / len(type_results),
                'avg_bertscore_f1': sum(r['bertscore_f1'] for r in type_results) / len(type_results)
            }
        
        metrics['by_question_type'] = type_metrics
        
        return metrics
    
    def save_results(self, results: list, overall_metrics: dict):
        """Save evaluation results to JSON and CSV."""
        # Save detailed results to JSON
        output_data = {
            'overall_metrics': overall_metrics,
            'detailed_results': results
        }
        
        with open(config.RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to {config.RESULTS_FILE}")
        
        # Save to CSV for easy viewing
        import pandas as pd
        
        df = pd.DataFrame(results)
        df.to_csv(config.RESULTS_CSV, index=False)
        
        print(f"CSV results saved to {config.RESULTS_CSV}")
    
    def run(self):
        """Run complete evaluation pipeline."""
        print("\n" + "="*60)
        print("HYBRID RAG SYSTEM - AUTOMATED EVALUATION PIPELINE")
        print("="*60)
        
        # Load questions
        print("\n1. Loading test questions...")
        questions = self.load_questions()
        print(f"   Loaded {len(questions)} questions")
        
        # Run evaluation
        print("\n2. Running evaluation...")
        results = self.run_evaluation(questions)
        print(f"   Evaluated {len(results)} questions successfully")
        
        # Calculate overall metrics
        print("\n3. Calculating overall metrics...")
        overall_metrics = self.calculate_overall_metrics(results)
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Questions: {overall_metrics['total_questions']}")
        print(f"\nRetrieval Metrics:")
        print(f"  Average MRR (URL-level): {overall_metrics['avg_mrr']:.4f}")
        print(f"  Average NDCG@{config.NDCG_K}: {overall_metrics['avg_ndcg_at_k']:.4f}")
        print(f"\nAnswer Quality Metrics (BERTScore):")
        print(f"  F1: {overall_metrics['avg_bertscore_f1']:.4f}")
        print(f"  Precision: {overall_metrics['avg_bertscore_precision']:.4f}")
        print(f"  Recall: {overall_metrics['avg_bertscore_recall']:.4f}")
        print(f"\nPerformance:")
        print(f"  Avg Retrieval Time: {overall_metrics['avg_retrieval_time']:.2f}s")
        print(f"  Avg Generation Time: {overall_metrics['avg_generation_time']:.2f}s")
        print(f"  Avg Total Time: {overall_metrics['avg_total_time']:.2f}s")
        
        print(f"\nBy Question Type:")
        for q_type, metrics in overall_metrics['by_question_type'].items():
            print(f"  {q_type.capitalize()} ({metrics['count']} questions):")
            print(f"    MRR: {metrics['avg_mrr']:.4f}, NDCG@K: {metrics['avg_ndcg_at_k']:.4f}, BERTScore F1: {metrics['avg_bertscore_f1']:.4f}")
        
        # Save results
        print("\n4. Saving results...")
        self.save_results(results, overall_metrics)
        
        # Return results for further processing
        return_data = {
            'overall_metrics': overall_metrics,
            'per_question_results': results
        }
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        
        return return_data


if __name__ == "__main__":
    pipeline = EvaluationPipeline()
    pipeline.run()
