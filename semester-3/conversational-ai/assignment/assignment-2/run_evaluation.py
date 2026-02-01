# Master Evaluation Runner
# Automatically checks dependencies and runs complete evaluation pipeline

import sys
import os
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

def check_file_exists(filepath, description):
    """Check if a required file exists"""
    exists = os.path.exists(filepath)
    status = "OK" if exists else "MISS"
    print(f"  [{status}] {description}: {filepath}")
    return exists

def run_script(script_path, description):
    """Run a Python script"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed: {e}")
        return False

def main():
    print("="*60)
    print("HYBRID RAG SYSTEM - COMPLETE EVALUATION PIPELINE")
    print("="*60)
    print()
    print("This script will automatically:")
    print("  1. Check if indices exist (run if needed)")
    print("  2. Generate questions if needed")
    print("  3. Run complete evaluation")
    print("  4. Generate HTML report")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    print()
    
    # STEP 1: Check corpus
    corpus_exists = check_file_exists(config.CORPUS_FILE, "Corpus")
    
    # STEP 2: Check chunks
    chunks_exists = check_file_exists(config.CHUNKS_FILE, "Chunks")
    
    # STEP 3: Check indices
    faiss_exists = check_file_exists(config.VECTOR_INDEX_FILE, "FAISS Index")
    bm25_exists = check_file_exists(config.BM25_INDEX_FILE, "BM25 Index")
    indices_exist = faiss_exists and bm25_exists
    
    # STEP 4: Check questions
    questions_exist = check_file_exists(config.QUESTIONS_FILE, "Questions Dataset")
    
    print()
    
    # Run missing dependencies
    if not corpus_exists:
        print("\n[WARNING] Corpus not found!")
        print("You need to run data collection first:")
        print("  python src/data_collection.py")
        sys.exit(1)
    
    if not chunks_exists:
        print("\n[WARNING] Chunks not found! Running preprocessing...")
        if not run_script("src/preprocessing.py", "Text Preprocessing"):
            print("\n[ERROR] Preprocessing failed. Please check errors above.")
            sys.exit(1)
    
    if not indices_exist:
        print("\n[WARNING] Indices not found! Building retrieval indices...")
        if not run_script("src/hybrid_retrieval.py", "Index Building"):
            print("\n[ERROR] Index building failed. Please check errors above.")
            sys.exit(1)
    
    if not questions_exist:
        print("\n[WARNING] Questions not found! Generating test questions...")
        print("This will take 10-15 minutes...")
        if not run_script("evaluation/question_generation.py", "Question Generation"):
            print("\n[ERROR] Question generation failed. Please check errors above.")
            sys.exit(1)
    
    # All dependencies ready, run evaluation
    print("\n" + "="*60)
    print("ALL DEPENDENCIES READY - STARTING EVALUATION")
    print("="*60)
    print()
    
    try:
        # Import after dependencies are ready
        from evaluation.pipeline import EvaluationPipeline
        from evaluation.innovative_eval import run_ablation_study, analyze_errors
        from evaluation.report_generator import ReportGenerator
        
        # Run evaluation pipeline (it loads its own retriever and generator)
        print("Running evaluation pipeline (this takes 20-30 minutes)...")
        pipeline = EvaluationPipeline()
        results = pipeline.run()
        
        # Get retriever and generator from pipeline for ablation study
        retriever = pipeline.retriever
        generator = pipeline.generator
        
        print("\nSaving results...")
        import json
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        
        with open(config.RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
        print(f" [OK] Saved results to {config.RESULTS_FILE}")
        
        # Also save CSV
        import pandas as pd
        df = pd.DataFrame(results['per_question_results'])
        df.to_csv(config.RESULTS_CSV, index=False)
        print(f" [OK] Saved CSV to {config.RESULTS_CSV}")
        
        # Run ablation study
        print("\nRunning ablation study...")
        ablation_results = run_ablation_study(retriever, generator)
        
        # Error analysis
        print("\nAnalyzing errors...")
        error_analysis = analyze_errors(results)
        
        # Combine all results
        extended_results = {
            **results,
            'ablation_study': ablation_results,
            'error_analysis': error_analysis
        }
        
        import json
        with open(config.EXTENDED_RESULTS, 'w') as f:
            json.dump(extended_results, f, indent=2)
        print(f" [OK] Saved extended results to {config.EXTENDED_RESULTS}")
        
        # Generate visualizations and HTML report
        print("\nGenerating visualizations and HTML report...")
        report_gen = ReportGenerator()
        report_gen.generate_reports(config.EXTENDED_RESULTS)
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print()
        print("Generated files:")
        print(f"  1. {config.RESULTS_FILE}")
        print(f"  2. {config.RESULTS_CSV}")
        print(f"  3. {config.EXTENDED_RESULTS}")
        print(f"  4. {config.HTML_REPORT}")
        print(f"  5. {os.path.join(config.REPORTS_DIR, 'metrics_overview.png')}")
        print(f"  6. {os.path.join(config.REPORTS_DIR, 'question_type_performance.png')}")
        print(f"  7. {os.path.join(config.REPORTS_DIR, 'retrieval_comparison.png')}")
        print(f"  8. {os.path.join(config.REPORTS_DIR, 'generation_metrics.png')}")
        print(f"  9. {os.path.join(config.REPORTS_DIR, 'time_analysis.png')}")
        print(f" 10. {os.path.join(config.REPORTS_DIR, 'ablation_study.png')}")
        print(f" 11. {os.path.join(config.REPORTS_DIR, 'error_analysis.png')}")
        
        print(f"\n [DONE] Open {config.HTML_REPORT} in your browser to view the complete report!")
        
        # Print key metrics
        metrics = results['overall_metrics']
        print("\nKey Metrics:")
        print(f"  MRR (URL-level): {metrics['mrr_url']:.3f}")
        print(f"  NDCG@5: {metrics['ndcg_at_5']:.3f}")
        print(f"  BERTScore F1: {metrics['bertscore_f1']:.3f}")
        print(f"  Total questions: {metrics['total_questions']}")
        
    except FileNotFoundError as e:
        print(f"\n [ERROR] {e}")
        print("\nMake sure you have:")
        print("  1. Built the indices: python src/hybrid_retrieval.py")
        print("  2. Generated questions: python evaluation/question_generation.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n [ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
