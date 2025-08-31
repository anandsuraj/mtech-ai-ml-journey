
## Overview
This project performs sentiment analysis on financial text data using various Natural Language Processing (NLP) techniques and machine learning models.

## Key Terms and Concepts Explained

### 1. **Sentiment Analysis**
- **Definition**: Determining the emotional tone or opinion expressed in text
- **Example**: 
  - "The stock market is performing excellently" → **Positive**
  - "Company profits declined significantly" → **Negative**
  - "The quarterly report shows standard results" → **Neutral**

### 2. **Text Preprocessing**

#### **Removing Punctuation, Numbers, and Special Characters**
- **Purpose**: Clean text to focus on meaningful words
- **Example**:
  - Original: "Apple's stock price rose 15% today!"
  - After cleaning: "Apple stock price rose today"

#### **Stop Words Removal**
- **Definition**: Common words that don't carry much meaning (the, is, at, which, on)
- **Example**:
  - Original: "The company is performing well in the market"
  - After removal: "company performing well market"

#### **Stemming vs Lemmatization**
- **Stemming**: Reduces words to their root form (crude chopping)
  - Example: running → run, better → better
- **Lemmatization**: Reduces words to their meaningful base form (intelligent reduction)
  - Example: running → run, better → good
- **We use Lemmatization** because it's more accurate

### 3. **Word Cloud**
- **Definition**: Visual representation where frequently used words appear larger
- **Purpose**: Quick visualization of most common terms in the dataset
- **Example**: In financial data, words like "profit", "loss", "market" might appear prominently

### 4. **Word Embeddings (Word2Vec)**

#### **What are Word Embeddings?**
- **Definition**: Converting words into numerical vectors that capture semantic meaning
- **Example**: 
  - "king" might be represented as [0.2, -0.1, 0.8, 0.3, ...]
  - "queen" might be represented as [0.1, -0.2, 0.7, 0.4, ...]
  - Similar words have similar vector representations

#### **Skip-gram Model**
- **How it works**: Given a target word, predict the surrounding context words
- **Example**: 
  - Sentence: "The company reported strong profits"
  - Target word: "reported"
  - Context words to predict: "company", "strong"
- **Advantages**: 
  - Good with rare words
  - Captures semantic relationships well
  - Better for word similarity tasks

#### **CBOW (Continuous Bag of Words)**
- **How it works**: Given context words, predict the target word
- **Example**: 
  - Context words: "The", "company", "strong", "profits"
  - Target word to predict: "reported"
- **Advantages**: 
  - Faster training
  - Good with frequent words
  - Better for syntactic relationships

### 5. **Machine Learning Components**

#### **Features (X) and Labels (Y)**
- **X (Features)**: The input data (cleaned sentences converted to numbers)
- **Y (Labels)**: The output we want to predict (sentiment: positive/negative/neutral)

#### **Train-Test Split**
- **Purpose**: Divide data to train model and test its performance
- **Example**: 80% for training, 20% for testing
- **Why**: Prevents overfitting and gives realistic performance measure

#### **Decision Tree Classifier**
- **Definition**: Algorithm that makes decisions by asking yes/no questions
- **Example**: 
  ```
  Does sentence contain "profit"?
  ├─ Yes: Does it contain "decline"?
  │  ├─ Yes → Negative
  │  └─ No → Positive
  └─ No: Does it contain "loss"?
     ├─ Yes → Negative
     └─ No → Neutral
  ```

#### **Confusion Matrix**
- **Definition**: Table showing correct vs incorrect predictions
- **Example**:
  ```
  Actual\Predicted  Positive  Negative  Neutral
  Positive             85        3        2
  Negative              5       78        7
  Neutral               8        4       88
  ```
- **Reading**: Diagonal values are correct predictions, off-diagonal are errors

### 6. **HMM POS Tagging**

#### **POS (Parts of Speech) Tagging**
- **Definition**: Identifying grammatical roles of words (noun, verb, adjective, etc.)
- **Example**: 
  - "The company reported strong profits"
  - "The" → Determiner (DT)
  - "company" → Noun (NN)
  - "reported" → Verb (VBD)
  - "strong" → Adjective (JJ)
  - "profits" → Noun (NNS)

#### **HMM (Hidden Markov Model)**
- **Definition**: Statistical model that considers word sequence patterns
- **How it works**: Uses probability of word sequences to determine POS tags
- **Example**: After "The", there's high probability of a noun, so "company" is likely NN

## Project Workflow

### Step 1: Data Loading
```python
# Load dataset
fsa = pd.read_csv('FinancialSentimentAnalysis.csv')
# Expected columns: 'Sentence', 'Sentiment'
```

### Step 2: Text Preprocessing
```python
# Original: "Apple's Q3 profits increased by 15%!"
# After preprocessing: "apple q profits increased"
```

### Step 3: Normalization
```python
# After lemmatization: "apple q profit increase"
```

### Step 4: Feature Extraction
- Convert cleaned text to numerical vectors using Word2Vec
- Two approaches: Skip-gram and CBOW

### Step 5: Model Training
- Split data into training and testing sets
- Train Decision Tree on both Skip-gram and CBOW features
- Compare performance

### Step 6: Evaluation
- Use confusion matrix to see prediction accuracy
- Compare which word embedding method works better

## Expected Results

### Model Comparison
- **Skip-gram**: Usually better for semantic understanding
- **CBOW**: Usually faster but may be less accurate
- **Decision Factor**: Dataset size, word frequency, task requirements

### Performance Metrics
- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: Detailed breakdown of prediction errors

## Files Required
1. `FinancialSentimentAnalysis.csv` - Main dataset
2. `financial_sentiment_analysis.py` - Main code file
3. Required Python packages (listed in code)

## How to Run
1. Download dataset from provided Google Drive link
2. Install required packages: `pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn gensim`
3. Run: `python financial_sentiment_analysis.py`


Dataset Overview Analysis
Dataset Composition:

Total samples: 5,842 financial sentences
Sentiment distribution:

Neutral: 3,130 (most common - 53.6%)
Positive: ~1,356 (23.2%)
Negative: ~1,356 (23.2%)



Sample Sentence Analysis
Let me explain the sentiment classifications for the 5 sample sentences:
1. Positive Sentiment Example
Original: "The GeoSolutions technology will leverage Benefon GPS solution..."
Cleaned: "geosolutions technology leverage benefon gps solution..."
Why Positive: Contains words like "leverage", "solution", "powerful", "commercial model" - indicating business growth and technological advancement.
2. Negative Sentiment Example
Original: "$ESI on lows, down $1.50 to $2.50 BK a real possibility"
Cleaned: "esi low bk real possibility"
Why Negative: Contains words like "lows", "down", indicating declining stock performance and potential bankruptcy (BK).
3. Positive Sentiment Example
Original: "For the last quarter of 2010, Componenta's net sales doubled..."
Cleaned: "last quarter componenta net sale doubled..."
Why Positive: "doubled" and "net sales" indicate strong financial performance.
4. Neutral Sentiment Examples
Original: "According to the Finnish-Russian Chamber of Commerce..."
Why Neutral: Factual statement without emotional or performance indicators.
Model Performance Analysis
Key Findings:

Training Accuracy: 92.85% (both models) - Very high!
Testing Accuracy:

Skip-gram: 44.91%
CBOW: 40.29%


What This Means:

Severe Overfitting: The huge gap between training (92.85%) and testing (~44%) indicates the models memorized training data but can't generalize.
Skip-gram vs CBOW:

Skip-gram performed better (44.91% vs 40.29%)
Skip-gram is better at capturing semantic relationships
Both are still performing poorly on unseen data

Why Low Testing Accuracy?

Financial text complexity: Financial language is highly domain-specific
Neutral dominance: 53.6% neutral samples make classification challenging
Small vector size: 100-dimensional vectors might be insufficient
Limited training data: 5,842 samples might not be enough for robust Word2Vec

POS Tagging Analysis
The HMM POS tagging on the first cleaned sentence shows:

Nouns (NN, NNS): technology, solution, platform, content, model
Adjectives (JJ): gps, relevant, new, powerful, commercial
Verbs (VBG, VBN, VBD): providing, based, benefon

This helps understand the grammatical structure and can improve sentiment analysis.