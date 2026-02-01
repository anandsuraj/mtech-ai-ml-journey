# Flask web application for Hybrid RAG System
# Production-ready UI instead of Streamlit

import os
import sys
import time
from typing import Dict, List

# Fix imports - add parent directory to path FIRST
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set HuggingFace cache before importing transformers
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = config.CACHE_DIR
if config.HF_TOKEN:
    os.environ['HF_TOKEN'] = config.HF_TOKEN

from flask import Flask, render_template, request, jsonify
import json

from src.hybrid_retrieval import HybridRetriever
from src.llm_generation import ResponseGenerator

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Global variables for models (loaded once at startup)
retriever = None
generator = None

def initialize_models():
    """Load models at startup to avoid reloading on each request"""
    global retriever, generator
    
    print("Loading RAG system models...")
    
    try:
        retriever = HybridRetriever()
        retriever.load_indices()
        generator = ResponseGenerator()
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search():
    """Handle search requests"""
    try:        
        # Get query from request
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Perform retrieval
        retrieval_start = time.time()
        retrieved_chunks, metadata = retriever.search(query)
        retrieval_time = time.time() - retrieval_start
        
        # Generate answer
        generation_start = time.time()
        result = generator.generate_answer(query, retrieved_chunks)
        generation_time = time.time() - generation_start
        
        # Prepare response
        chunks_data = []
        for chunk, score in retrieved_chunks:
            chunks_data.append({
                'title': chunk['title'],
                'url': chunk['url'],
                'text': chunk['text'],
                'rrf_score': chunk.get('rrf_score', 0),
                'dense_rank': chunk.get('dense_rank'),
                'sparse_rank': chunk.get('sparse_rank')
            })
        
        response = {
            'query': query,
            'answer': result['answer'],
            'retrieval_time': retrieval_time,
            'generation_time': generation_time,
            'total_time': retrieval_time + generation_time,
            'chunks': chunks_data
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Search error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'models_loaded': retriever is not None and generator is not None
    })

if __name__ == '__main__':
    # Initialize models before starting server
    if not initialize_models():
        print("Failed to load models. Make sure indices are built.")
        print("Run: python src/hybrid_retrieval.py")
        sys.exit(1)
    
    # Start Flask server
    print(f"Starting server on {config.UI_HOST}:{config.UI_PORT}")
    app.run(
        host=config.UI_HOST,
        port=config.UI_PORT,
        debug=config.UI_DEBUG
    )
