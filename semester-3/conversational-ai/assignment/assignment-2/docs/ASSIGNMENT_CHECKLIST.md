# Assignment Requirements Checklist

## Part 1: Hybrid RAG System (10 Marks)

### 1.1 Dense Vector Retrieval 
-  Sentence embedding model (all-MiniLM-L6-v2)
-  FAISS vector index
-  Cosine similarity search
-  Top-K retrieval
- **File**: `src/embeddings.py`

### 1.2 Sparse Keyword Retrieval 
-  BM25 algorithm implemented
-  Index built over chunks
-  Top-K results retrieval
- **File**: `src/sparse_retrieval.py`

### 1.3 Reciprocal Rank Fusion (RRF) 
-  RRF formula: score(d) = Σ 1/(k + rank_i(d))
-  k = 60 as specified
-  Combines dense and sparse results
-  Selects top-N chunks by RRF score
- **File**: `src/hybrid_retrieval.py`

### 1.4 Response Generation 
-  Open-source LLM (Flan-T5-base)
-  Context concatenation with query
-  Answer generation within context limits
- **File**: `src/llm_generation.py`

### 1.5 User Interface 
-  Built with Flask
-  Displays user query input
-  Shows generated answer
-  Shows top retrieved chunks with sources
-  Displays dense/sparse/RRF scores
-  Shows response time
- **Files**: `ui/app.py`, `ui/templates/`, `ui/static/`

## Part 2: Automated Evaluation (10 Marks)

### 2.1 Question Generation 
-  100 Q&A pairs generated
-  Diverse question types (factual, comparative, inferential, multi-hop)
-  Stored with ground truth
-  Source IDs included
-  Question categories tracked
- **File**: `evaluation/question_generation.py`

### 2.2 Evaluation Metrics

#### 2.2.1 Mandatory Metric (2 Marks) 
-  MRR calculated at URL level
-  Measures rank of first correct Wikipedia URL
-  Average of 1/rank across all questions
- **File**: `evaluation/metrics.py`

#### 2.2.2 Custom Metrics (4 Marks) 
-  Selected 2 additional metrics:
  - **NDCG@K**: Normalized Discounted Cumulative Gain
  - **BERTScore**: Semantic similarity evaluation
-  Justification provided for each metric
-  Calculation methods documented
-  Interpretation guidelines included
- **File**: `evaluation/metrics.py`, `evaluation/report_generator.py`

### 2.3 Innovative Evaluation (4 Marks) 
-  Ablation Studies: Dense-only, sparse-only, hybrid comparison
-  Error Analysis: Categorize failures by question type
-  Visualizations: Performance charts and heatmaps
- **File**: `evaluation/innovative_eval.py`

### 2.4 Automated Pipeline 
-  Single-command execution
-  Loads questions
-  Runs RAG system
-  Computes all metrics
-  Generates comprehensive reports (HTML)
-  Structured output (CSV/JSON)
- **File**: `run_evaluation.py`, `evaluation/pipeline.py`

### 2.5 Evaluation Report 
-  Overall performance summary
-  MRR and custom metrics averages
-  Detailed justification for custom metrics
-  Results table with all required columns
-  Visualizations (comparisons, distributions, heatmaps)
-  Error analysis with examples
- **File**: `evaluation/report_generator.py`

## Dataset Requirements 

### Wikipedia URL Collection
-  200 fixed URLs (diverse topics, minimum 200 words)
-  Stored in `fixed_urls.json`
-  300 random URLs per run (changes each indexing)
-  Total: 500 URLs (200 fixed + 300 random)
- **Files**: `scripts/generate_fixed_urls.py`, `src/data_collection.py`

### Text Processing 
-  Chunking: 200-400 tokens with 50-token overlap
-  Metadata: URL, title, unique chunk IDs
-  Clean text extraction
- **File**: `src/preprocessing.py`

## Additional Requirements 

### Documentation
-  README.md with setup instructions
-  Code comments (natural, student-style)
-  Configuration centralized (`config.py`)

### Ease of Use 
-  Docker support for team deployment
-  Single script to run entire pipeline (`run.sh`)
-  Clear directory structure
-  Requirements file with all dependencies

## Verification Status

**All assignment requirements met!** 

### Key Files Summary:
1. **Core System**: `src/*.py` (6 files)
2. **Evaluation**: `evaluation/*.py` (5 files)
3. **UI**: `ui/app.py`, `ui/templates/`, `ui/static/`
4. **Configuration**: `config.py`, `requirements.txt`
5. **Execution**: `run.sh`, `run_evaluation.py`
6. **Documentation**: `README.md`, `DOCKER.md`
7. **Docker**: `Dockerfile`, `docker-compose.yml`
8. **Utilities**: `scripts/*.py` (3 files)

### Ready for Submission
- Clean project structure
- All features implemented
- Comprehensive documentation
- Docker support for team collaboration
- Single-command execution
- Automated evaluation pipeline
