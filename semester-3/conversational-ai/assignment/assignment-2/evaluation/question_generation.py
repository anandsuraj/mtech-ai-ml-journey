"""
Automated Question Generation from Wikipedia Corpus
Rule-based approach for fast, reliable question generation
"""

import os
import sys
import json
import random
import re
from typing import List, Dict

# Fix imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.preprocessing import load_chunks


class QuestionGenerator:
    """Generate questions from Wikipedia corpus using rule-based templates."""
    
    def __init__(self):
        """Initialize question generator with templates."""
        print("Initializing rule-based question generator...")
        
        self.factual_templates = [
            "What is {topic}?",
            "What are {topic}?",
            "Who was {topic}?",
            "When did {topic} occur?",
            "Where is {topic} located?",
            "What happened in {topic}?",
        ]
        
        self.inferential_templates = [
            "Why is {topic} important?",
            "How does {topic} work?",
            "Why did {topic} happen?",
            "How is {topic} related to its context?",
        ]
        
        print("Question generator ready!")
    
    def extract_topic(self, text: str) -> str:
        """Extract a topic from text (first noun phrase or first sentence subject)."""
        # Simple extraction: get first 3-5 words that aren't common words
        words = text.split()[:50]
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'but'}
        meaningful_words = [w for w in words if w.lower() not in common_words and len(w) > 2]
        
        if len(meaningful_words) >= 2:
            return ' '.join(meaningful_words[:3])
        return ' '.join(words[:3])
    
    def extract_answer(self, text: str, question: str) -> str:
        """Extract answer from text based on question type."""
        # For simplicity, return first 2-3 sentences as answer
        sentences = text.split('. ')
        
        if 'Who' in question or 'What' in question:
            # Return first 1-2 sentences
            return '. '.join(sentences[:2]) + '.'
        elif 'When' in question or 'Where' in question:
            # Look for dates or locations in first few sentences
            for sent in sentences[:3]:
                if any(char.isdigit() for char in sent):
                    return sent + '.'
            return sentences[0] + '.'
        else:
            # For why/how, return 2-3 sentences
            return '. '.join(sentences[:3]) + '.'
    
    def generate_factual_question(self, chunk: Dict) -> Dict:
        """Generate a factual question (who, what, when, where)."""
        text = chunk['text']
        title = chunk.get('title', 'the topic')
        
        # Use title as topic or extract from text
        topic = title if title != 'the topic' else self.extract_topic(text)
        
        # Select random template
        template = random.choice(self.factual_templates)
        question = template.format(topic=topic)
        
        # Extract answer
        answer = self.extract_answer(text, question)
        
        return {
            'question': question,
            'answer': answer,
            'type': 'factual',
            'chunk_id': chunk['chunk_id'],
            'source_url': chunk['url']
        }
    
    def generate_comparative_question(self, chunks: List[Dict]) -> Dict:
        """Generate a comparative question from two chunks."""
        chunk1, chunk2 = random.sample(chunks, 2)
        
        title1 = chunk1.get('title', 'the first topic')
        title2 = chunk2.get('title', 'the second topic')
        
        templates = [
            f"What are the differences between {title1} and {title2}?",
            f"How does {title1} compare to {title2}?",
            f"What are the similarities between {title1} and {title2}?",
        ]
        
        question = random.choice(templates)
        
        # Combine info from both chunks
        answer = f"{chunk1['text'][:200]}. Meanwhile, {chunk2['text'][:200]}."
        
        return {
            'question': question,
            'answer': answer,
            'type': 'comparative',
            'chunk_id': f"{chunk1['chunk_id']},{chunk2['chunk_id']}",
            'source_url': f"{chunk1['url']},{chunk2['url']}"
        }
    
    def generate_inferential_question(self, chunk: Dict) -> Dict:
        """Generate an inferential question (why, how)."""
        text = chunk['text']
        title = chunk.get('title', 'this topic')
        
        # Select random template
        template = random.choice(self.inferential_templates)
        question = template.format(topic=title)
        
        # Extract answer (longer for inferential)
        answer = self.extract_answer(text, question)
        
        return {
            'question': question,
            'answer': answer,
            'type': 'inferential',
            'chunk_id': chunk['chunk_id'],
            'source_url': chunk['url']
        }
    
    def generate_multihop_question(self, chunks: List[Dict]) -> Dict:
        """Generate a multi-hop question requiring multiple chunks."""
        selected_chunks = random.sample(chunks, min(3, len(chunks)))
        
        titles = [c.get('title', 'topic') for c in selected_chunks]
        
        templates = [
            f"How are {titles[0]}, {titles[1]}, and {titles[2]} related?",
            f"What connects {titles[0]} and {titles[1]}?",
            f"How does {titles[0]} influence {titles[1]}?",
        ]
        
        question = random.choice(templates)
        
        # Combine info from all chunks
        combined_answer = '. '.join([c['text'][:150] for c in selected_chunks])
        
        return {
            'question': question,
            'answer': combined_answer,
            'type': 'multi_hop',
            'chunk_id': [c['chunk_id'] for c in selected_chunks],
            'source_url': [c['url'] for c in selected_chunks]
        }
    
    def generate_dataset(self, chunks: List[Dict]) -> List[Dict]:
        """Generate complete question dataset."""
        questions = []
        question_id = 1
        
        # Get question counts from config
        total = config.QUESTIONS_COUNT
        types = config.QUESTION_TYPES
        
        print(f"\nGenerating {total} questions...")
        print(f"  - Factual: {types['factual']}")
        print(f"  - Comparative: {types['comparative']}")
        print(f"  - Inferential: {types['inferential']}")
        print(f"  - Multi-hop: {types['multi_hop']}")
        
        # Factual questions
        print("\nGenerating factual questions...")
        for i in range(types['factual']):
            chunk = random.choice(chunks)
            q = self.generate_factual_question(chunk)
            q['question_id'] = question_id
            questions.append(q)
            question_id += 1
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{types['factual']}...")
        
        # Comparative questions
        print("\nGenerating comparative questions...")
        for i in range(types['comparative']):
            q = self.generate_comparative_question(chunks)
            q['question_id'] = question_id
            questions.append(q)
            question_id += 1
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{types['comparative']}...")
        
        # Inferential questions
        print("\nGenerating inferential questions...")
        for i in range(types['inferential']):
            chunk = random.choice(chunks)
            q = self.generate_inferential_question(chunk)
            q['question_id'] = question_id
            questions.append(q)
            question_id += 1
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{types['inferential']}...")
        
        # Multi-hop questions
        print("\nGenerating multi-hop questions...")
        for i in range(types['multi_hop']):
            q = self.generate_multihop_question(chunks)
            q['question_id'] = question_id
            questions.append(q)
            question_id += 1
            if (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{types['multi_hop']}...")
        
        print(f"\nGenerated {len(questions)} questions total!")
        return questions
    
    def save_dataset(self, questions: List[Dict]):
        """Save questions to JSON file."""
        os.makedirs(os.path.dirname(config.QUESTIONS_FILE), exist_ok=True)
        
        with open(config.QUESTIONS_FILE, 'w') as f:
            json.dump(questions, f, indent=2)
        
        print(f"\nSaved questions to: {config.QUESTIONS_FILE}")


def main():
    """Main execution."""
    print("="*60)
    print("QUESTION GENERATION - RULE-BASED APPROACH")
    print("="*60)
    
    # Load chunks
    print("\nLoading corpus chunks...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")
    
    # Generate questions
    generator = QuestionGenerator()
    questions = generator.generate_dataset(chunks)
    
    # Save dataset
    generator.save_dataset(questions)
    
    print("\n" + "="*60)
    print("QUESTION GENERATION COMPLETE!")
    print("="*60)
    print(f"\nGenerated {len(questions)} questions")
    print(f"Saved to: {config.QUESTIONS_FILE}")


if __name__ == "__main__":
    main()
