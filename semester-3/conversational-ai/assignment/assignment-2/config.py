# Configuration file for Hybrid RAG System
# All system parameters and settings in one place

import os

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SRC_DIR = os.path.join(BASE_DIR, 'src')
EVAL_DIR = os.path.join(BASE_DIR, 'evaluation')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
UI_DIR = os.path.join(BASE_DIR, 'ui')

# HuggingFace and Threading Configuration
# Purge potentially expired tokens that cause 401 errors
for key in ['HF_TOKEN', 'HUGGINGFACE_HUB_TOKEN', 'HUGGING_FACE_HUB_TOKEN']:
    if key in os.environ:
        # Check if it's the known expired token or just generally purge for this submission
        # This ensures we use public access for the assignment
        print(f"DEBUG: Removing {key} from environment to avoid 401 Unauthorized errors.")
        del os.environ[key]

# macOS and Threading Safety (Critical for Stable Execution on Mac)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

HF_TOKEN = None # Force None for this project
CACHE_DIR = os.path.join(BASE_DIR, '.cache', 'huggingface')

# Dataset configuration
FIXED_URLS_COUNT = 200 #200
RANDOM_URLS_COUNT = 300 #300
TOTAL_URLS_COUNT = 500 #500
MIN_WORDS_PER_PAGE = 200 #200

# File paths
FIXED_URLS_FILE = os.path.join(DATA_DIR, 'fixed_urls.json')
CORPUS_FILE = os.path.join(DATA_DIR, 'corpus.pkl')
CHUNKS_FILE = os.path.join(DATA_DIR, 'chunks.json')
VECTOR_INDEX_FILE = os.path.join(DATA_DIR, 'faiss_index.bin')
VECTOR_METADATA_FILE = os.path.join(DATA_DIR, 'faiss_index_metadata.pkl')
BM25_INDEX_FILE = os.path.join(DATA_DIR, 'bm25_index.pkl')

# Chunking settings (from assignment requirements)
MIN_CHUNK_TOKENS = 200
MAX_CHUNK_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50

# Model settings
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
LLM_MODEL = 'google/flan-t5-base'
MAX_GEN_LENGTH = 256
TEMPERATURE = 0.7

# Retrieval settings
DENSE_TOP_K = 10
SPARSE_TOP_K = 10
RRF_K = 60  # From assignment formula: RRF_score(d) = sum(1/(k + rank_i(d)))
FINAL_TOP_N = 5

# Evaluation settings
NUM_QUESTIONS = 100
QUESTIONS_COUNT = 100  # Total number of questions to generate
QUESTIONS_FILE = os.path.join(EVAL_DIR, 'questions_dataset.json')
RESULTS_FILE = os.path.join(REPORTS_DIR, 'results.json')
RESULTS_CSV = os.path.join(REPORTS_DIR, 'results.csv')
EXTENDED_RESULTS = os.path.join(REPORTS_DIR, 'extended_results.json')
HTML_REPORT = os.path.join(REPORTS_DIR, 'evaluation_report.html')

# Question distribution by type
QUESTION_TYPES = {
    'factual': 30,
    'comparative': 20,
    'inferential': 30,
    'multi_hop': 20
}

# Metrics configuration
NDCG_K = 5
MRR_CUTOFF = 10

# UI settings
UI_HOST = '0.0.0.0'
UI_PORT = 5000
UI_DEBUG = True

# Create required directories
for directory in [DATA_DIR, SRC_DIR, EVAL_DIR, REPORTS_DIR, UI_DIR]:
    os.makedirs(directory, exist_ok=True)
