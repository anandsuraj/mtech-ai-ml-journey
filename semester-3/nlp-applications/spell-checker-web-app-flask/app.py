# Flask backend for Spell Checker Application
# This application uses PySpellChecker library for spell checking

from flask import Flask, render_template, request, jsonify
from spellchecker import SpellChecker
import re
import json
import os
from datetime import datetime

# Initialize Flask application
app = Flask(__name__)

# Initialize spell checker (English by default)
spell = SpellChecker()

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
    
    # Keep only last 25 entries to prevent file from growing too large
    history = history[:25]
    
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
    Check spelling of text using PySpellChecker library.
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
    
    # Find misspelled words
    misspelled = spell.unknown(words)
    
    # Dictionary to store misspelled words and their suggestions
    misspelled_details = {}
    
    # Get suggestions for each misspelled word
    for word in misspelled:
        # Get top 5 correction candidates
        candidates = spell.candidates(word)
        if candidates:
            suggestions = list(candidates)[:5]
            misspelled_details[word] = suggestions
    
    # Auto-correct the text
    corrected_words = []
    for word in words:
        if word.lower() in misspelled:
            # Use the best correction
            correction = spell.correction(word.lower())
            if correction:
                corrected_words.append(correction)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    
    corrected_text = ' '.join(corrected_words)
    
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
    
    Receives text from frontend, checks spelling using PySpellChecker library,
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
    print("Using PySpellChecker library for spell checking")
    print("=" * 60)
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("=" * 60)
    
    # Run Flask app in debug mode
    app.run(debug=True, host='0.0.0.0', port=5000)
