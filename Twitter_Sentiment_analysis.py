"""
Twitter Airline Sentiment Analysis
Course: Natural Language Processing
Task: Text Classification using Bag-of-Words and Multiple ML Classifiers
"""

# ========================================
# Import Required Libraries
# ========================================
import pandas as pd
import nltk
import re
from nltk import word_tokenize, sent_tokenize
from collections import Counter, defaultdict
from nltk.util import ngrams

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import string
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

import random
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', None)

# ========================================
# Question 2 - Text Classification Model
# ========================================
"""
Dataset: Twitter Airline Sentiment
Source: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

Tasks:
1. Classify sentiment through bag of words using NLTK sentiment analyzer
2. Compare performance metrics with multiple classifiers
"""

# ========================================
# Load and Explore Dataset
# ========================================
df = pd.read_csv("Tweets.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ========================================
# Text Preprocessing Function
# ========================================
def preprocess(text):
    """
    Preprocess text for sentiment analysis
    - Convert to lowercase
    - Remove punctuation
    - Tokenize
    - Remove stopwords
    - Lemmatize
    """
    text = str(text).lower()  # Convert to lowercase
    text = "".join([c for c in text if c not in string.punctuation])  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]  # Remove stopwords & lemmatize
    return tokens

# Apply preprocessing
df['tokens'] = df['text'].apply(preprocess)
df['features'] = df['tokens'].apply(lambda tokens: {word: True for word in tokens})

# ========================================
# NLTK Naive Bayes Classifier
# ========================================
print("\n" + "="*50)
print("NLTK NAIVE BAYES SENTIMENT ANALYSIS")
print("="*50)

# Create dataset for NLTK
nltk_dataset = [(df['features'][i], df['airline_sentiment'][i]) for i in range(len(df))]

# Split dataset into train/test (70/30)
train_set, test_set = train_test_split(nltk_dataset, test_size=0.3, random_state=42, shuffle=True)

# Train NLTK NaiveBayesClassifier
nb_classifier = NaiveBayesClassifier.train(train_set)

# Calculate NLTK accuracy
nltk_accuracy = accuracy(nb_classifier, test_set)
print(f"NLTK SentimentAnalyzer Accuracy: {nltk_accuracy:.4f}")

# Show most informative features
print("\nMost Informative Features:")
nb_classifier.show_most_informative_features(50)

print("\n" + "="*50)
print("KEY OBSERVATIONS FROM INFORMATIVE FEATURES")
print("="*50)
print("""
Positive sentiment indicators: 
- Words like 'favorite', 'awesome', 'beautiful', 'excellent', 'amazing', 'thank', 'kudos' 
- Indicate customer satisfaction with airline service

Negative sentiment indicators: 
- Words like 'terrible', 'hold', 'rebook', 'suck' 
- Point to pain points: delays, rebooking issues, customer service problems

Neutral sentiment indicators: 
- Words like 'street', 'dragon', 'promo', 'policy'
- Convey factual information rather than emotional content
""")

# ========================================
# Prepare Data for Sklearn Classifiers
# ========================================
print("\n" + "="*50)
print("SKLEARN CLASSIFIERS - BAG OF WORDS APPROACH")
print("="*50)

# Convert token lists back to strings for vectorization
df['clean_text'] = df['tokens'].apply(lambda x: " ".join(x))

X = df['clean_text']
y = df['airline_sentiment']

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorize text using Bag of Words
# Using unigrams, bigrams, and trigrams
vectorizer = CountVectorizer(max_features=25000, ngram_range=(1,3))
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)

print(f"Feature matrix shape: {X_train_bow.shape}")

# ========================================
# Define All Classifiers
# ========================================
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SGDClassifier L1": SGDClassifier(penalty="l1", max_iter=1000),
    "SGDClassifier L2": SGDClassifier(penalty="l2", max_iter=1000),
    "Ridge Classifier": RidgeClassifier(max_iter=1000),
    "Perceptron": Perceptron(max_iter=1000),
    "MultinomialNB": MultinomialNB(),
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB()
}

# ========================================
# Train and Evaluate All Models
# ========================================
print("\n" + "="*50)
print("TRAINING AND EVALUATING ALL MODELS")
print("="*50)

for name, clf in classifiers.items():
    print(f"\n{name}:")
    print("-" * 30)
    
    # Train the model
    clf.fit(X_train_bow, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_bow)
    
    # Calculate and display accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Display detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

# ========================================
# Create Summary Table of All Models
# ========================================
print("\n" + "="*50)
print("PERFORMANCE METRICS SUMMARY TABLE")
print("="*50)

results = []

for name, clf in classifiers.items():
    # Retrain and predict (to ensure consistency)
    clf.fit(X_train_bow, y_train)
    y_pred = clf.predict(X_test_bow)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    results.append([name, acc, precision, recall, f1])

# Create and display results dataframe
results_df = pd.DataFrame(results, columns=['Classifier', 'Accuracy', 'Precision', 'Recall', 'F1-score'])
results_df_sorted = results_df.sort_values(by='Accuracy', ascending=False)

print(results_df_sorted.to_string(index=False))

# ========================================
# Business Insights
# ========================================
print("\n" + "="*50)
print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
print("="*50)

print("""
After evaluating various classifiers for airline tweet sentiment classification:

1. TOP PERFORMERS:
   - Logistic Regression: ~80% accuracy with balanced precision/recall
   - ComplementNB: Nearly equal performance, excellent for imbalanced data
   - Both suitable for production deployment

2. CUSTOMER SERVICE OPTIMIZATION:
   - Negative indicators ('terrible', 'hold', 'rebook') enable quick issue identification
   - Positive indicators ('awesome', 'excellent', 'thank') highlight service strengths
   - Real-time monitoring can reduce response time to complaints

3. MODEL SELECTION:
   - MultinomialNB & SGDClassifier (L1/L2): Good performance with word-count features
   - Ridge Classifier & Perceptron: Robust but may miss subtle sentiments
   - BernoulliNB: Lower performance, less suitable for detailed analysis

4. IMPLEMENTATION STRATEGY:
   - Deploy Logistic Regression or ComplementNB for accurate sentiment classification
   - Enable proactive customer service responses
   - Enhance customer experience and brand trustworthiness
   - Increase airline profitability through improved satisfaction
""")

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)