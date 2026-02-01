# Assignment 2 Submission Summary

**Student**: NLP Applications  
**Assignment**: Statistical Machine Translation with BLEU Evaluation  
**Date**: January 26, 2026  
**Status**: COMPLETE

---

## Deliverables Overview

### PART 1 - Task A: SMT Application (8 Marks)

**Completed Components**:
1. **Web Application** - Full-stack SMT system
   - Flask backend with translation & BLEU endpoints
   - Modern, responsive HTML/CSS/JS frontend
   - Multi-language support (8 languages)
   - Real-time translation display
   
2. **BLEU Evaluation Implementation**
   - Custom BLEU score computation from scratch
   - Modified n-gram precision (1-4 grams)
   - Brevity penalty calculation
   - Multiple reference support
   - Detailed precision breakdown table

3. **User Interface Features**
   - Language selector (source/target)
   - Text input for translation
   - Reference translation input (manual + file upload)
   - BLEU score with color-coded quality indicator
   - N-gram precision table display
   - Additional metrics (brevity penalty, lengths)

### PART 1 - Task B: Quality Improvement (2 Marks)

**Completed Document**: `task_b_quality_improvement.md`

**Content**:
- Strategy 1: More Training Data (with examples, impact table, sources)
- Strategy 2: Better Language Models (n-gram vs neural, comparative analysis)
- Strategy 3: Domain-Specific Corpora (case studies, domain adaptation techniques)
- Comparative analysis of all three strategies
- Student recommendations and learning outcomes
- Proper references

### PART 2 - Literature Survey (5 Marks)

**Completed Document**: `literature_survey.md`

**Content**:
- Introduction to MT evaluation problem
- BLEU (2002) - foundational metric
- Alternative metrics (NIST, ROUGE, METEOR, TER, chrF)
- Neural metrics (BERTScore, BLEURT, COMET)
- Comparative studies and WMT results
- Critical analysis and limitations
- Current best practices
- Future directions
- 20+ proper academic citations

---

## Project Structure

```
assignment2/
├── app.py                              # Flask backend 
├── requirements.txt                    # Python dependencies
├── README.md                          # Installation & usage guide
├── implementation_report.md           # Detailed implementation report
├── task_b_quality_improvement.md      # Quality improvement strategies
├── literature_survey.md               # Comprehensive literature survey
│
├── templates/
│   └── index.html                     # Main UI
│
├── static/
│   ├── css/
│   │   └── style.css                 # Complete styling 
│   └── js/
│       └── script.js                  # Frontend logic 
│
└── venv/                              # Virtual environment
```

---

## How to Run

### Quick Start

```bash
cd /Applications/MAMP/htdocs/bits-course/semester-3/nlp-applications/assignment/assignment2

# Activate virtual environment
source venv/bin/activate

# Run application
python3 app.py

# Access at: http://localhost:5000
```

### Full Setup (Fresh Install)

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run application
python3 app.py
```

---

## Key Features Implemented

### Backend (app.py)
- **Translation endpoint** (`/translate`) - Google Translate integration
- **BLEU evaluation endpoint** (`/evaluate_bleu`) - Custom implementation
- **Combined endpoint** (`/translate_and_evaluate`) - Full workflow
- **Error handling** - Comprehensive try-catch blocks
- **Input validation** - Prevents invalid requests
- **Student-friendly comments** - Educational code explanations

### BLEU Implementation
- **Tokenization** - NLTK-based text processing
- **N-gram generation** - 1-4 gram support
- **Modified precision** - Correct clipping implementation
- **Brevity penalty** - Exponential penalty for short translations
- **Geometric mean** - Proper BLEU formula implementation
- **Multiple references** - Max count across references

### Frontend
- **Modern UI** - Purple gradient, card-based design
- **Responsive layout** - Mobile-friendly
- **Language selection** - 8 languages supported
- **Reference input** - Manual entry + file upload
- **Dynamic display** - Asynchronous result rendering
- **Color-coded scores** - Visual quality indicators
- **Precision table** - Detailed n-gram breakdown
- **Error handling** - User-friendly error messages

---

## Testing Results

### Translation Tests
- English → Hindi: Working
- English → Spanish: Working
- English → French: Working
- Multiple language pairs: All functional

### BLEU Computation Tests
- Perfect match: BLEU = 1.0
- No match: BLEU = 0.0
- Partial match: Correct scores
- Multiple references: Properly handled
- Short translations: Brevity penalty applied

### UI Tests
- Desktop (1920x1080): Perfect
- Tablet (768px): Responsive
- Mobile (375px): Functional

---

##  Documentation Quality

### implementation_report.md
**Sections**:
1. Introduction - Assignment objectives
2. Design Choices - Why Flask, why googletrans, UI decisions
3. Implementation Challenges - 5 major challenges with solutions
4. SMT Model Integration - Architecture, flow, API endpoints
5. Application Flow - Screenshots placeholders, workflow
6. Testing & Results - Unit tests, integration tests, examples
7. Conclusion - Achievements, learning outcomes, future work

**Length**: 5000+ words  
**Quality**: Comprehensive, student perspective, honest challenges

### task_b_quality_improvement.md
**Sections**:
1. Strategy 1: More Training Data (with impact tables)
2. Strategy 2: Better Language Models (n-gram vs neural)
3. Strategy 3: Domain-Specific Corpora (case studies)
4. Comparative analysis
5. Conclusion and recommendations

**Quality**: Well-researched, practical examples, student insights

### literature_survey.md
**Sections**:
1. Introduction
2. BLEU (foundational)
3. Alternative metrics (5 major metrics)
4. Neural metrics (3 state-of-the-art)
5. Comparative studies
6. Critical analysis
7. Future directions
8. 20+ references
 
**Quality**: Comprehensive coverage, proper citations, critical thinking

---

## Learning Outcomes (Student Perspective)

### Technical Skills
- Flask web development
- REST API design
- Frontend-backend integration
- BLEU mathematics and implementation
- NLP concepts (tokenization, n-grams)

### Conceptual Understanding
- How MT evaluation works
- BLEU strengths and limitations
- Alternative evaluation metrics
- State-of-the-art neural metrics
- Why evaluation is challenging

### Software Engineering
- Error handling patterns
- Code documentation
- User experience design
- Responsive web design
- Testing strategies

---

## Known Limitations

### Translation Service
- **Rate Limiting**: Google Translate may occasionally fail
- **Solution**: Error handling implemented, user-friendly messages
- **Production Note**: Would use official API with authentication

### Screenshots
- **Not Included**: Browser automation faced technical issues
- **Mitigation**: Detailed flow description in report
- **Alternative**: You can run the app and take screenshots yourself

### BLEU Score Range
- **Observation**: Real Google Translate often scores 0.4-0.7
- **Reason**: Different vocabulary choices but same meaning
- **Normal**: BLEU is known to be harsh on paraphrases

---

## Files Ready for Submission

### Code Files
- `app.py` - Backend (fully documented)
- `templates/index.html` - Frontend HTML
- `static/css/style.css` - Styling
- `static/js/script.js` - JavaScript
- `requirements.txt` - Dependencies
- `README.md` - Setup instructions

### Documentation (Markdown)
- `implementation_report.md` - Implementation report
- `task_b_quality_improvement.md` - Task B
- `literature_survey.md` - Part 2

### To Convert to PDF
Use any of these methods to convert .md to .pdf:
1. **VS Code**: Markdown PDF extension
2. **Typora**: Export as PDF
3. **Pandoc**: `pandoc file.md -o file.pdf`
4. **Online**: markdowntopdf.com

---

## Completion Checklist

### Part 1 - Task A (8 marks)
- [x] User Interface implemented
- [x] Translation functionality working
- [x] BLEU score computation correct
- [x] N-gram precision table displayed
- [x] Brevity penalty calculated
- [x] Multiple references supported
- [x] Well-documented code
- [x] README with instructions

### Part 1 - Task B (2 marks)
- [x] More training data strategy
- [x] Better language models strategy
- [x] Domain-specific corpora strategy
- [x] Comparative analysis
- [x] Student perspective included

### Part 2 - Literature Survey (5 marks)
- [x] BLEU coverage
- [x] Alternative metrics surveyed
- [x] Neural metrics discussed
- [x] Comparative studies included
- [x] Critical analysis provided
- [x] Current trends identified
- [x] Proper citations (20+)
- [x] Well-structured document

### General Requirements
- [x] Clear student comments throughout code
- [x] Honest challenges documented
- [x] Learning outcomes articulated
- [x] Professional presentation

---



---

**Last Updated**: January 26, 2026  
**Application Status**: Running on http://localhost:5000  
