# Flask backend for Spell Checker Application
# This application uses TextBlob library for spell checking

from flask import Flask, render_template, request, jsonify
from textblob import TextBlob
from textblob import Word
import re
import json
import os
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)

# History file to store all spell checking attempts
HISTORY_FILE = 'spell_check_history.json'


def load_history():
    """
    Load spell checking history from JSON file.
    
    Returns:
        list: List of previous spell checking attempts
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_to_history(original, corrected, misspelled_count, misspelled_words):
    """
    Save spell checking attempt to history file.
    
    Args:
        original (str): Original text
        corrected (str): Corrected text
        misspelled_count (int): Number of misspelled words
        misspelled_words (dict): Dictionary of misspelled words with suggestions
    """
    # Load existing history
    history = load_history()
    
    # Create new history entry
    entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original': original,
        'corrected': corrected,
        'misspelled_count': misspelled_count,
        'misspelled_words': misspelled_words
    }
    
    # Add to beginning of history (most recent first)
    history.insert(0, entry)
    
    # Keep only last 100 entries to prevent file from growing too large
    history = history[:100]
    
    # Save to file
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def extract_words(text):
    """
    Extract words from text, removing punctuation and special characters.
    
    Args:
        text (str): Input text to process
    
    Returns:
        list: List of words extracted from text
    """
    # Use regex to find all words (letters only)
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return words


def check_spelling(text):
    """
    Check spelling of text using TextBlob library.
    Detects misspelled words and provides correction suggestions.
    
    Args:
        text (str): Input text to check
    
    Returns:
        dict: Dictionary containing:
            - original: original text
            - corrected: auto-corrected text
            - misspelled: dict of misspelled words with suggestions
    """
    # Extract words from text
    words = extract_words(text)
    
    # Dictionary to store misspelled words and their suggestions
    misspelled_details = {}
    
    # Check each word individually using TextBlob
    for word in words:
        # Create a Word object for spell checking
        w = Word(word.lower())
        
        # Get spelling suggestions
        suggestions = w.spellcheck()
        
        # If the word is misspelled, the first suggestion will have confidence < 1.0
        # or the corrected word will be different from original
        if suggestions and len(suggestions) > 0:
            # suggestions is a list of tuples: [(word, confidence), ...]
            top_suggestion = suggestions[0][0]
            confidence = suggestions[0][1]
            
            # If confidence is not 1.0 or the suggestion is different, it's misspelled
            if confidence < 1.0 or top_suggestion.lower() != word.lower():
                # Get top 5 unique suggestions
                unique_suggestions = []
                seen = set()
                for suggestion, conf in suggestions:
                    if suggestion.lower() not in seen and suggestion.lower() != word.lower():
                        unique_suggestions.append(suggestion)
                        seen.add(suggestion.lower())
                        if len(unique_suggestions) >= 5:
                            break
                
                if unique_suggestions:
                    misspelled_details[word] = unique_suggestions
    
    # Auto-correct the text using TextBlob
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    
    return {
        'original': text,
        'corrected': corrected_text,
        'misspelled': misspelled_details
    }


# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/check_spelling', methods=['POST'])
def check_spelling_route():
    """
    API endpoint for spell checking.
    
    Receives text from frontend, checks spelling using TextBlob library,
    and returns:
    - Original text
    - Corrected text
    - Misspelled words with suggestions
    
    Also saves the attempt to history file for future reference.
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        text = data.get('text', '')
        
        # Validate input
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Check spelling using the library
        result = check_spelling(text)
        
        # Save to history for future reference
        save_to_history(
            original=result['original'],
            corrected=result['corrected'],
            misspelled_count=len(result['misspelled']),
            misspelled_words=result['misspelled']
        )
        
        # Print results to terminal (as per requirements)
        print("\n" + "=" * 60)
        print("SPELL CHECK RESULTS")
        print("=" * 60)
        print(f"Original Text: {result['original']}")
        print(f"Corrected Text: {result['corrected']}")
        print(f"Misspelled Words: {len(result['misspelled'])}")
        for word, suggestions in result['misspelled'].items():
            print(f"  - '{word}' → {suggestions}")
        print("=" * 60 + "\n")
        
        # Return success response
        return jsonify({
            'success': True,
            'original': result['original'],
            'corrected': result['corrected'],
            'misspelled': result['misspelled']
        })
    
    except Exception as e:
        # Handle errors
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """
    API endpoint to retrieve spell checking history.
    
    Returns:
        JSON array of previous spell checking attempts
    """
    try:
        history = load_history()
        return jsonify({
            'success': True,
            'history': history,
            'total': len(history)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """
    API endpoint to clear all spell checking history.
    """
    try:
        # Delete history file
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        
        return jsonify({
            'success': True,
            'message': 'History cleared successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# RUN APPLICATION
# ============================================
if __name__ == '__main__':
    print("=" * 60)
    print("Spell Checker Application")
    print("Using TextBlob library for spell checking")
    print("=" * 60)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    # Run Flask app in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)
