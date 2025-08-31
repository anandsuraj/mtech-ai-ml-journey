# Financial Sentiment Analysis Assignment
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')

# 1. Load dataset and basic exploration
def load_and_explore_data():
    # Load the dataset
    fsa = pd.read_csv('/Users/surajanand/Downloads/Financial_Sentiment_Analysis.csv')
    
    print("Dataset Head:")
    print(fsa.head())
    print("\n" + "="*50 + "\n")
    
    print("Dataset Info:")
    print(fsa.info())
    print("\n" + "="*50 + "\n")
    
    print("Dataset Description:")
    print(fsa.describe())
    print("\n" + "="*50 + "\n")
    
    return fsa

# 2. Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuations, numbers, and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# 3. Remove stopwords and normalize using lemmatization
def clean_and_normalize(text):
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

# 4. Create word cloud
def create_wordcloud(text_data):
    # Combine all text
    all_text = ' '.join(text_data)
    
    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    
    # Plot word cloud
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Cleaned Sentences')
    plt.show()

# 5A. Skip-gram model implementation
def create_skipgram_features(sentences):
    # Tokenize sentences for Word2Vec
    tokenized_sentences = [sentence.split() for sentence in sentences]
    
    # Train Skip-gram model
    skipgram_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, sg=1, workers=4)  # sg=1 for skip-gram
    
    # Create feature vectors by averaging word vectors
    def get_sentence_vector(sentence):
        words = sentence.split()
        word_vectors = [skipgram_model.wv[word] for word in words if word in skipgram_model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(100)
    
    feature_vectors = np.array([get_sentence_vector(sentence) for sentence in sentences])
    return feature_vectors

# 5B. CBOW model implementation
def create_cbow_features(sentences):
    # Tokenize sentences for Word2Vec
    tokenized_sentences = [sentence.split() for sentence in sentences]
    
    # Train CBOW model
    cbow_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, sg=0, workers=4)  # sg=0 for CBOW
    
    # Create feature vectors by averaging word vectors
    def get_sentence_vector(sentence):
        words = sentence.split()
        word_vectors = [cbow_model.wv[word] for word in words if word in cbow_model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(100)
    
    feature_vectors = np.array([get_sentence_vector(sentence) for sentence in sentences])
    return feature_vectors

# Function to train decision tree and display results
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name):
    # Train Decision Tree
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train, y_train)
    
    # Predictions
    train_pred = dt_classifier.predict(X_train)
    test_pred = dt_classifier.predict(X_test)
    
    # Confusion matrices
    train_cm = confusion_matrix(y_train, train_pred)
    test_cm = confusion_matrix(y_test, test_pred)
    
    # Display results
    print(f"\n{model_name} Results:")
    print(f"Training Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"Testing Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(train_cm, annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title(f'{model_name} - Training Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(test_cm, annot=True, fmt='d', ax=axes[1], cmap='Blues')
    axes[1].set_title(f'{model_name} - Testing Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.show()
    
    return accuracy_score(y_test, test_pred)

# 6. HMM POS Tagging function
def hmm_pos_tagging(sentence):
    # Tokenize the sentence
    tokens = word_tokenize(sentence)
    
    # POS tagging using NLTK (which uses HMM-based tagger)
    pos_tags = nltk.pos_tag(tokens)
    
    print("HMM POS Tagging for first cleaned sentence:")
    print("Word\t\tPOS Tag")
    print("-" * 30)
    for word, pos in pos_tags:
        print(f"{word}\t\t{pos}")
    
    return pos_tags

# Main execution
def main():
    # 1. Load and explore data
    print("Step 1: Loading and exploring dataset...")
    fsa = load_and_explore_data()
    
    # 2 & 3. Preprocessing and normalization
    print("Step 2 & 3: Preprocessing and normalizing text...")
    fsa['cleaned_sentence'] = fsa['Sentence'].apply(preprocess_text)
    fsa['cleaned_sentence'] = fsa['cleaned_sentence'].apply(clean_and_normalize)
    
    print("Sample cleaned sentences:")
    print(fsa[['Sentence', 'cleaned_sentence']].head())
    print("\n" + "="*50 + "\n")
    
    # 4. Create word cloud
    print("Step 4: Creating word cloud...")
    create_wordcloud(fsa['cleaned_sentence'])
    
    # 5. Create X and Y objects
    print("Step 5: Creating feature vectors and training models...")
    X_text = fsa['cleaned_sentence']
    Y = fsa['Sentiment']
    
    # 5A. Skip-gram model
    print("5A. Skip-gram Model:")
    X_skipgram = create_skipgram_features(X_text)
    X_train_sg, X_test_sg, y_train_sg, y_test_sg = train_test_split(
        X_skipgram, Y, test_size=0.2, random_state=42, stratify=Y)
    
    skipgram_accuracy = train_and_evaluate(X_train_sg, X_test_sg, y_train_sg, y_test_sg, "Skip-gram")
    
    # 5B. CBOW model
    print("5B. CBOW Model:")
    X_cbow = create_cbow_features(X_text)
    X_train_cbow, X_test_cbow, y_train_cbow, y_test_cbow = train_test_split(
        X_cbow, Y, test_size=0.2, random_state=42, stratify=Y)
    
    cbow_accuracy = train_and_evaluate(X_train_cbow, X_test_cbow, y_train_cbow, y_test_cbow, "CBOW")
    
    # 5C. Compare Skip-gram and CBOW
    print("\n" + "="*50)
    print("5C. Comparison between Skip-gram and CBOW:")
    print(f"Skip-gram Test Accuracy: {skipgram_accuracy:.4f}")
    print(f"CBOW Test Accuracy: {cbow_accuracy:.4f}")
    
    if skipgram_accuracy > cbow_accuracy:
        print("\nSkip-gram performs better than CBOW.")
        print("Skip-gram is better at capturing semantic relationships and works well with rare words.")
        print("It predicts context words from target word, making it effective for word similarity tasks.")
    else:
        print("\nCBOW performs better than Skip-gram.")
        print("CBOW is faster to train and works well with frequent words.")
        print("It predicts target word from context, making it effective for syntactic relationships.")
    
    # 6. HMM POS Tagging
    print("\n" + "="*50)
    print("Step 6: HMM POS Tagging on first cleaned sentence:")
    first_cleaned_sentence = fsa['cleaned_sentence'].iloc[0]
    hmm_pos_tagging(first_cleaned_sentence)

# Run the main function
if __name__ == "__main__":
    main()