"""
Quick Reference - Hybrid RAG System Commands
"""

# =============================================================================
# INITIAL SETUP (Run once)
# =============================================================================

# 1. Install all dependencies
pip install -r requirements.txt

# 2. Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt')"

# =============================================================================
# DATA COLLECTION (Run once to generate fixed URLs)
# =============================================================================

# Generate the fixed 200 Wikipedia URLs (ONLY run this once!)
python src/data_collection.py --new-fixed
# Output: data/fixed_urls.json

# =============================================================================
# FULL PIPELINE (Run for each new corpus)
# =============================================================================

# Step 1: Collect corpus (200 fixed + 300 random = 500 articles)
python src/data_collection.py
# Output: data/corpus.pkl
# Time: ~15-30 minutes (depends on network speed)

# Step 2: Preprocess into chunks
python src/preprocessing.py
# Output: data/chunks.json
# Time: ~1-2 minutes

# Step 3: Build retrieval indices
python src/hybrid_retrieval.py
# Output: data/faiss_index.bin, data/faiss_index_metadata.pkl, data/bm25_index.pkl
# Time: ~5-10 minutes (depends on GPU/CPU)

# =============================================================================
# USAGE - RUN THE RAG SYSTEM
# =============================================================================

# Option 1: Interactive Streamlit UI
cd ui
streamlit run app.py
# Opens browser at http://localhost:8501
# Enter questions and see answers with sources and scores

# Option 2: Programmatic usage
python
>>> from src.hybrid_retrieval import HybridRetriever
>>> from src.llm_generation import ResponseGenerator
>>> retriever = HybridRetriever()
>>> retriever.load_indices()
>>> generator = ResponseGenerator()
>>> chunks, metadata = retriever.search("What is machine learning?")
>>> result = generator.generate_answer("What is machine learning?", chunks)
>>> print(result['answer'])

# =============================================================================
# EVALUATION
# =============================================================================

# Step 1: Generate 100 test questions
python evaluation/question_generation.py
# Output: evaluation/questions_dataset.json
# Time: ~10-15 minutes

# Step 2: Run complete evaluation (ALL-IN-ONE COMMAND!)
python run_evaluation.py
# Outputs:
#   - reports/results.json (detailed results)
#   - reports/results.csv (CSV format)
#   - reports/evaluation_report.html (complete HTML report)
#   - reports/*.png (visualizations)
# Time: ~20-30 minutes for 100 questions

# Step 3: View the report
# Open reports/evaluation_report.html in your browser

# =============================================================================
# TESTING INDIVIDUAL COMPONENTS
# =============================================================================

# Test data collection
python src/data_collection.py

# Test preprocessing
python src/preprocessing.py

# Test dense retrieval
python src/embeddings.py

# Test sparse retrieval
python src/sparse_retrieval.py

# Test hybrid retrieval
python src/hybrid_retrieval.py

# Test LLM generation
python src/llm_generation.py

# Test metrics
python evaluation/metrics.py

# =============================================================================
# FILE LOCATIONS
# =============================================================================

# Data files (generated)
data/fixed_urls.json              # 200 fixed Wikipedia URLs
data/corpus.pkl                   # Collected articles
data/chunks.json                  # Preprocessed chunks
data/faiss_index.bin              # Dense vector index
data/faiss_index_metadata.pkl     # Dense index metadata
data/bm25_index.pkl               # Sparse BM25 index

# Evaluation files (generated)
evaluation/questions_dataset.json # 100 test questions
reports/results.json              # Evaluation results
reports/results.csv               # Results in CSV format
reports/extended_results.json     # Full results with ablation
reports/evaluation_report.html    # Final HTML report
reports/*.png                     # Visualizations

# Source code
src/data_collection.py            # Wikipedia data collection
src/preprocessing.py              # Text chunking
src/embeddings.py                 # Dense retrieval
src/sparse_retrieval.py           # Sparse (BM25) retrieval
src/hybrid_retrieval.py           # RRF combination
src/llm_generation.py             # Answer generation

# Evaluation code
evaluation/question_generation.py # Generate test questions
evaluation/metrics.py             # MRR, NDCG, BERTScore
evaluation/innovative_eval.py     # Ablation & error analysis
evaluation/pipeline.py            # Automated evaluation pipeline
evaluation/report_generator.py    # HTML report generation

# UI
ui/app.py                         # Streamlit application

# =============================================================================
# CONFIGURATION
# =============================================================================

# All parameters are in config.py
# Key settings:
#   - FIXED_URLS_COUNT = 200
#   - RANDOM_URLS_COUNT = 300
#   - MIN_CHUNK_TOKENS = 200
#   - MAX_CHUNK_TOKENS = 400
#   - CHUNK_OVERLAP_TOKENS = 50
#   - RRF_K = 60 (as per assignment)
#   - DENSE_TOP_K = 10
#   - SPARSE_TOP_K = 10
#   - FINAL_TOP_N = 5

# =============================================================================
# SUBMISSION CHECKLIST
# =============================================================================

# Before submitting, make sure you have:
# [ ] Generated fixed_urls.json (200 URLs)
# [ ] Collected corpus and built indices
# [ ] Generated 100 test questions
# [ ] Run complete evaluation
# [ ] Generated HTML report
# [ ] Tested Streamlit UI
# [ ] Verified all visualizations are created
# [ ] Checked README.md is complete
# [ ] Reviewed walkthrough.md

# =============================================================================
# TROUBLESHOOTING
# =============================================================================

# If you get "index not found" errors:
#   Run: python src/hybrid_retrieval.py

# If you get "questions not found" errors:
#   Run: python evaluation/question_generation.py

# If Streamlit UI doesn't load indices:
#   Make sure you ran: python src/hybrid_retrieval.py

# If evaluation fails:
#   Check that indices are built and questions are generated

# For GPU issues:
#   The system auto-detects and falls back to CPU if no GPU

# For memory issues:
#   Reduce DENSE_TOP_K and SPARSE_TOP_K in config.py
#   Or process questions in smaller batches

# =============================================================================
# EXPECTED PERFORMANCE
# =============================================================================

# Retrieval time: ~0.5-1.0 seconds per query
# Generation time: ~1-2 seconds per query
# Total time per query: ~2-3 seconds
# Evaluation (100 questions): ~20-30 minutes

# =============================================================================
# END OF QUICK REFERENCE
# =============================================================================