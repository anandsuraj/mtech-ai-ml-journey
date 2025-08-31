# Financial Sentiment Analysis Results

## Dataset Overview Analysis

### Dataset Composition:
- **Total samples**: 5,842 financial sentences
- **Sentiment distribution**:
  - **Neutral**: 3,130 (most common - 53.6%)
  - **Positive**: ~1,356 (23.2%)
  - **Negative**: ~1,356 (23.2%)

## Sample Sentence Analysis

Let me explain the sentiment classifications for the 5 sample sentences:

### 1. Positive Sentiment Example
- **Original**: "The GeoSolutions technology will leverage Benefon GPS solution..."
- **Cleaned**: "geosolutions technology leverage benefon gps solution..."
- **Why Positive**: Contains words like "leverage", "solution", "powerful", "commercial model" - indicating business growth and technological advancement.

### 2. Negative Sentiment Example
- **Original**: "$ESI on lows, down $1.50 to $2.50 BK a real possibility"
- **Cleaned**: "esi low bk real possibility"
- **Why Negative**: Contains words like "lows", "down", indicating declining stock performance and potential bankruptcy (BK).

### 3. Positive Sentiment Example
- **Original**: "For the last quarter of 2010, Componenta's net sales doubled..."
- **Cleaned**: "last quarter componenta net sale doubled..."
- **Why Positive**: "doubled" and "net sales" indicate strong financial performance.

### 4. Neutral Sentiment Examples
- **Original**: "According to the Finnish-Russian Chamber of Commerce..."
- **Why Neutral**: Factual statement without emotional or performance indicators.

## Model Performance Analysis

### Key Findings:
- **Training Accuracy**: 92.85% (both models) - Very high!
- **Testing Accuracy**:
  - **Skip-gram**: 44.91%
  - **CBOW**: 40.29%

### What This Means:

1. **Severe Overfitting**: The huge gap between training (92.85%) and testing (~44%) indicates the models memorized training data but can't generalize.

2. **Skip-gram vs CBOW**:
   - Skip-gram performed better (44.91% vs 40.29%)
   - Skip-gram is better at capturing semantic relationships
   - Both are still performing poorly on unseen data

### Why Low Testing Accuracy?

1. **Financial text complexity**: Financial language is highly domain-specific
2. **Neutral dominance**: 53.6% neutral samples make classification challenging
3. **Small vector size**: 100-dimensional vectors might be insufficient
4. **Limited training data**: 5,842 samples might not be enough for robust Word2Vec

## POS Tagging Analysis

The HMM POS tagging on the first cleaned sentence shows:

- **Nouns (NN, NNS)**: technology, solution, platform, content, model
- **Adjectives (JJ)**: gps, relevant, new, powerful, commercial
- **Verbs (VBG, VBN, VBD)**: providing, based, benefon

This helps understand the grammatical structure and can improve sentiment analysis.

## Recommendations for Improvement

1. **Increase vector dimensions** (200-300 instead of 100)
2. **Use pre-trained embeddings** (GloVe, FastText financial embeddings)
3. **Try different algorithms** (Random Forest, SVM, Neural Networks)
4. **Feature engineering** (TF-IDF, n-grams)
5. **Cross-validation** to better handle overfitting
6. **Balanced sampling** to handle neutral class dominance

---

**Conclusion**: The results show that while the preprocessing worked well, the classification models need improvement to better generalize to unseen financial text.