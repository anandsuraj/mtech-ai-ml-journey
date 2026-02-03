# LLM response generation for RAG system
# Generates answers using Flan-T5 based on retrieved context

import os
import sys

# Fix imports - add parent directory to path FIRST
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import config

# CRITICAL: Set HuggingFace cache to local directory BEFORE transformer imports
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = config.CACHE_DIR
if config.HF_TOKEN:
    os.environ['HF_TOKEN'] = config.HF_TOKEN

# Now safe to import transformer libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from typing import List, Dict, Tuple
import time


class ResponseGenerator:
    # Generates answers using LLM with retrieved chunks as context
    
    def __init__(self, model_name: str = config.LLM_MODEL):
        # Load the language model for generation        
        print(f"Loading LLM model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=False)
        
        # Check if GPU is available and use it
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
    
    def format_context(self, chunks: List[Tuple[Dict, float]]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            chunks: List of (chunk_dict, score) tuples
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, (chunk, score) in enumerate(chunks, 1):
            # Include title and text
            context_parts.append(
                f"[{i}] {chunk['title']}\n{chunk['text']}"
            )
        
        return "\n\n".join(context_parts)
    
    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Dict, float]],
        max_length: int = config.MAX_GEN_LENGTH
    ) -> Dict:
        """
        Generate answer based on query and retrieved context.
        
        Args:
            query: User's question
            retrieved_chunks: List of (chunk_dict, score) tuples
            max_length: Maximum length of generated answer
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Format context
        context = self.format_context(retrieved_chunks)
        
        # Create prompt
        prompt = self.create_prompt(query, context)
        
        # Generate answer
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                temperature=config.TEMPERATURE
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        
        return {
            'answer': answer,
            'query': query,
            'context': context,
            'num_chunks_used': len(retrieved_chunks),
            'generation_time': generation_time,
            'model': self.model_name
        }
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for the LLM.
        
        Args:
            query: User's question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Answer the following question based on the provided context. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt


if __name__ == "__main__":
    # Test response generation
    from preprocessing import load_chunks
    from hybrid_retrieval import HybridRetriever
    
    # Load chunks
    chunks = load_chunks()
    
    # Initialize retriever
    retriever = HybridRetriever()
    retriever.load_indices()
    
    # Initialize generator
    generator = ResponseGenerator()
    
    # Test query
    query = "What is machine learning?"
    
    # Retrieve relevant chunks
    print(f"Searching for: {query}")
    retrieved_chunks, metadata = retriever.search(query)
    
    # Generate answer
    print("\nGenerating answer...")
    result = generator.generate_answer(query, retrieved_chunks)
    
    print(f"\nQuery: {result['query']}")
    print(f"Answer: {result['answer']}")
    print(f"Generation time: {result['generation_time']:.2f} seconds")
    print(f"Chunks used: {result['num_chunks_used']}")
