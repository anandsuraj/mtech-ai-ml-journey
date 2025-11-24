# Spell Checker Application

A web-based spell checker using Flask and **PySpellChecker** library with history tracking.

## Features

- ✅ Uses PySpellChecker library for accurate spell checking
- ✅ Detects misspelled words in input text
- ✅ Provides suggested corrections for each misspelling
- ✅ **Stores all spell check history for future reference**
- ✅ View past spell checking attempts with timestamps
- ✅ Beautiful web interface with tabs
- ✅ Real-time statistics

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open: http://localhost:5000

## Usage

### Spell Checker Tab
1. Enter text with spelling errors
2. Click "Check Spelling"
3. View:
   - Original text
   - Corrected text
   - Misspelled words with suggestions
   - Statistics (total words, errors, accuracy)

### History Tab
1. Click on "History" tab
2. View all previous spell checking attempts
3. Each entry shows:
   - Timestamp
   - Original text
   - Corrected text
   - Number of errors
   - Misspelled words with suggestions
4. Clear history if needed

## Example

**Input:**
```
Teh quick brown fox jumps over teh lazy dog
```

**Output:**
- Detected misspelled words: "Teh" (appears twice)
- Suggestions: ["the", "tea", "ten", "tech", "ted"]
- Corrected text: "The quick brown fox jumps over the lazy dog"
- **Saved to history with timestamp**

## History Storage

All spell checking attempts are automatically saved to `spell_check_history.json` file:
- Stores last 100 attempts
- Includes timestamp, original text, corrected text, and suggestions
- Persists across application restarts
- Can be cleared from the History tab

## Library Used

**PySpellChecker** - A pure Python spell checking library that supports multiple languages. It uses frequency-based word lists and provides fast, accurate spell checking with support for English, Spanish, French, German, Portuguese, Russian, and Arabic.
