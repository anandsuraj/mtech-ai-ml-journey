import os
import sys
import time
from typing import List, Tuple

# Fix imports - add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Set HuggingFace cache
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = config.CACHE_DIR
if config.HF_TOKEN:
    os.environ['HF_TOKEN'] = config.HF_TOKEN

import gradio as gr
from src.hybrid_retrieval import HybridRetriever
from src.llm_generation import ResponseGenerator

# Global variables for models
retriever = None
generator = None

def initialize_models():
    """Load models at startup"""
    global retriever, generator
    
    print("Loading RAG system models for Gradio...")
    try:
        retriever = HybridRetriever()
        retriever.load_indices()
        generator = ResponseGenerator()
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def respond(message, history):
    """Handle chat messages"""
    if retriever is None or generator is None:
        return "System is still initializing models. Please wait a moment."
    
    try:
        # Perform retrieval
        retrieved_chunks, _ = retriever.search(message)
        
        # Generate answer
        result = generator.generate_answer(message, retrieved_chunks)
        answer = result['answer']
        
        # Format sources
        sources = "\n\n**Sources:**\n"
        seen_urls = set()
        for chunk, _ in retrieved_chunks:
            url = chunk.get('url', '')
            title = chunk.get('title', 'Unknown Source')
            if url and url not in seen_urls:
                sources += f"- [{title}]({url})\n"
                seen_urls.add(url)
            elif title and title not in seen_urls:
                sources += f"- {title}\n"
                seen_urls.add(title)
        
        return answer + sources
    except Exception as e:
        return f"Error: {str(e)}"

# UI Layout
with gr.Blocks(title="Hybrid RAG Chatbot") as demo:
    gr.Markdown("# Hybrid RAG Chatbot")
    gr.Markdown("Ask questions about the conversational AI course content.")
    
    chatbot = gr.ChatInterface(
        fn=respond,
        examples=["What is Hybrid Retrieval?", "How does RRF work?", "Tell me about BERT score."],
        cache_examples=False,
    )

if __name__ == "__main__":
    if initialize_models():
        # Launch with sharing enabled
        demo.launch(share=True, server_name=config.UI_HOST, server_port=config.UI_PORT + 1)
    else:
        print("Failed to initialize system. Please ensure indices are built (run src/hybrid_retrieval.py).")
