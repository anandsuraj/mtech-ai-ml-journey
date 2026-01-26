"""
Statistical Machine Translation Application with BLEU Evaluation
NLP Applications Assignment 2

This Flask application provides:
1. Statistical Machine Translation using Google Translate API
2. BLEU Score computation with n-gram precision analysis
3. Support for multiple reference translations
"""

from flask import Flask, render_template, request, jsonify
from googletrans import Translator
import nltk
from collections import Counter
import math
import re

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data for tokenization if not already present
nltk_data_dir = '/tmp/nltk_data'
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.insert(0, nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    except Exception as e:
        print(f"Warning: Could not download punkt: {e}")
# Initialize translator for Google Translate API access
translator = Translator()

def tokenize(text):
    """
    Break text into individual words for n-gram calculation.
    Converts to lowercase and removes punctuation.
    
    Args:
        text (str): Input text to tokenize
    
    Returns:
        list: List of tokens (words)
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return tokens

def get_ngrams(tokens, n):
    """
    Create n-grams (contiguous sequences) from tokens.
    Example: "hello world" -> 1-grams: [hello, world], 2-grams: [hello world]
    
    Args:
        tokens (list): List of tokens
        n (int): Size of n-gram (1 for unigram, 2 for bigram, etc.)
    
    Returns:
        list: List of n-grams as tuples
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def modified_precision(candidate_tokens, reference_tokens_list, n):
    """
    Calculate n-gram precision with clipping to prevent gaming the metric.
    Clips  each n-gram count to max times it appears in any reference.
    
    Args:
        candidate_tokens (list): Tokens from candidate (machine) translation
        reference_tokens_list (list): List of token lists from reference translations
        n (int): N-gram size
    
    Returns:
        float: Modified precision score (0 to 1)
    """
    candidate_ngrams = get_ngrams(candidate_tokens, n)
    
    if not candidate_ngrams:
        return 0.0
    
    candidate_counts = Counter(candidate_ngrams)
    
    # Track max count for each n-gram across all references
    max_ref_counts = Counter()
    for ref_tokens in reference_tokens_list:
        ref_ngrams = get_ngrams(ref_tokens, n)
        ref_counts = Counter(ref_ngrams)
        
        for ngram in ref_counts:
            max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_counts[ngram])
    
    # Clip counts to reference maximum
    clipped_counts = 0
    total_counts = 0
    
    for ngram, count in candidate_counts.items():
        clipped_counts += min(count, max_ref_counts[ngram])
        total_counts += count
    
    precision = clipped_counts / total_counts if total_counts > 0 else 0.0
    
    return precision

def brevity_penalty(candidate_length, reference_lengths):
    """
    Penalize translations that are too short. Without this, systems could
    game precision by only translating a few easy words.
    
    Formula: BP = 1 if c > r else e^(1 - r/c)
    
    Args:
        candidate_length (int): Length of candidate translation
        reference_lengths (list): Lengths of reference translations
    
    Returns:
        float: Brevity penalty (0 to 1)
    """
    # Use closest reference length for fairness
    closest_ref_length = min(reference_lengths, 
                             key=lambda ref_len: abs(ref_len - candidate_length))
    
    if candidate_length >= closest_ref_length:
        return 1.0
    else:
        return math.exp(1 - (closest_ref_length / candidate_length))

def calculate_bleu(candidate, references, max_n=4):
    """
    Calculate BLEU score - standard metric for MT evaluation.
    Compares machine translation against human references using n-gram matching.
    
    Formula: BP * exp(avg of log precisions)
    
    Args:
        candidate (str): Machine translation to evaluate
        references (list): Human reference translations
        max_n (int): Maximum n-gram size (default 4)
    
    Returns:
        dict: BLEU score and detailed metrics
    """
    candidate_tokens = tokenize(candidate)
    reference_tokens_list = [tokenize(ref) for ref in references]
    
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref) for ref in reference_tokens_list]
    
    # Get precision for each n-gram level
    precisions = []
    precision_details = {}
    
    for n in range(1, max_n + 1):
        prec = modified_precision(candidate_tokens, reference_tokens_list, n)
        precisions.append(prec)
        precision_details[f'{n}-gram'] = round(prec, 4)
    
    bp = brevity_penalty(candidate_length, reference_lengths)
    
    # Use geometric mean (gives equal weight to all n-gram sizes)
    if min(precisions) > 0:
        log_precision_sum = sum(math.log(p) for p in precisions)
        bleu_score = bp * math.exp(log_precision_sum / max_n)
    else:
        bleu_score = 0.0
    
    return {
        'bleu_score': round(bleu_score, 4),
        'brevity_penalty': round(bp, 4),
        'precision_details': precision_details,
        'candidate_length': candidate_length,
        'reference_length': reference_lengths[0] if reference_lengths else 0
    }

@app.route('/')
def index():
    """Render the main application page"""
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    """
    Translate text using Google Translate API.
    Production systems would use trained MT models like Moses.
    
    Returns:
        JSON: Translation result or error
    """
    try:
        data = request.get_json()
        source_text = data.get('source_text', '').strip()
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'hi')
        
        if not source_text:
            return jsonify({'error': 'Source text is required'}), 400
        
        # Can specify source or let googletrans auto-detect
        translation = translator.translate(
            source_text, 
            src=source_lang, 
            dest=target_lang
        )
        
        return jsonify({
            'translated_text': translation.text,
            'source_lang': source_lang,
            'target_lang': target_lang
        })
    
    except Exception as e:
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500

@app.route('/evaluate_bleu', methods=['POST'])
def evaluate_bleu():
    """
    Compute BLEU score comparing translation to references.
    Higher score (closer to 1.0) means better quality.
    
    Returns:
        JSON: BLEU score with detailed breakdown
    """
    try:
        data = request.get_json()
        candidate = data.get('candidate', '').strip()
        references = data.get('references', [])
        
        if not candidate:
            return jsonify({'error': 'Candidate translation is required'}), 400
        
        if not references or not any(ref.strip() for ref in references):
            return jsonify({'error': 'At least one reference translation is required'}), 400
        
        references = [ref.strip() for ref in references if ref.strip()]
        
        result = calculate_bleu(candidate, references)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'BLEU evaluation failed: {str(e)}'}), 500

@app.route('/translate_and_evaluate', methods=['POST'])
def translate_and_evaluate():
    """
    Convenience endpoint that translates and evaluates in one call.
    
    Returns:
        JSON: Translation and BLEU evaluation results
    """
    try:
        data = request.get_json()
        source_text = data.get('source_text', '').strip()
        source_lang = data.get('source_lang', 'en')
        target_lang = data.get('target_lang', 'hi')
        references = data.get('references', [])
        
        if not source_text:
            return jsonify({'error': 'Source text is required'}), 400
        
        translation = translator.translate(
            source_text, 
            src=source_lang, 
            dest=target_lang
        )
        
        translated_text = translation.text
        
        # Calculate BLEU if references provided
        bleu_result = None
        if references and any(ref.strip() for ref in references):
            references = [ref.strip() for ref in references if ref.strip()]
            bleu_result = calculate_bleu(translated_text, references)
        
        return jsonify({
            'translated_text': translated_text,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'bleu_evaluation': bleu_result
        })
    
    except Exception as e:
        return jsonify({'error': f'Operation failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
