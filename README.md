# Twitter Airline Sentiment Analysis

## Overview
Sentiment classification and modeling of airline tweets into positive, negative, and neutral categories using Bag-of-Words approach and multiple machine learning classifiers.

## Dataset
- **Source**: [Twitter Airline Sentiment Dataset](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
- **Content**: Customer tweets about US airlines
- **Target Variable**: airline_sentiment (positive, negative, neutral)

## Methodology

### Text Preprocessing Pipeline
- Lowercase transformation
- Punctuation removal
- Tokenization
- Stop words removal using NLTK
- Lemmatization using WordNetLemmatizer

### Feature Engineering
- **Bag-of-Words (BoW)** approach using CountVectorizer
- **N-gram range**: (1,3) - includes unigrams, bigrams, and trigrams
- **Max features**: 25,000
- Feature dictionary creation for NLTK Naive Bayes classifier

### Machine Learning Models Implemented
1. **NLTK Naive Bayes Classifier** - for sentiment analysis and feature importance
2. **Logistic Regression** - Best performing model
3. **SGDClassifier** with L1 regularization
4. **SGDClassifier** with L2 regularization  
5. **Ridge Classifier**
6. **Perceptron**
7. **MultinomialNB**
8. **BernoulliNB**
9. **ComplementNB**

## Key Results

### Model Performance
- **Logistic Regression** achieved the highest accuracy of approximately **80%**
- **ComplementNB** performed exceptionally well, especially suitable for imbalanced data
- All models evaluated using accuracy, precision, recall, and F1-score metrics
- Train-test split: 70-30 ratio

### Most Informative Features (from NLTK Naive Bayes)
- **Positive sentiment indicators**: "favorite", "awesome", "beautiful", "excellent", "amazing", "thank", "kudos"
- **Negative sentiment indicators**: "terrible", "hold", "rebook", "suck"
- **Neutral indicators**: "street", "dragon", "promo", "policy"

## Business Insights

### Operational Value
- **Customer Service Enhancement**: Quick identification of dissatisfied customers through negative sentiment detection
- **Pain Point Identification**: Words like "hold", "rebook" highlight specific service areas needing improvement
- **Real-time Monitoring**: Logistic Regression and ComplementNB recommended for production deployment

### Strategic Benefits
- Enables proactive response to customer complaints
- Improves service quality through data-driven insights
- Enhances brand trust through timely customer engagement
- Provides actionable intelligence for operational improvements

## Technical Skills Demonstrated
- Natural Language Processing (NLP) with NLTK
- Text preprocessing and feature engineering
- Multiple classifier implementation and evaluation
- Model comparison and selection
- Data visualization with confusion matrices
- Business insight extraction from ML models

## Libraries Used
- pandas
- nltk
- scikit-learn
- matplotlib
- numpy

## Files
- `sentiment_analysis.ipynb` - Jupyter notebook with complete analysis
- `Tweets.csv` - Dataset (available from Kaggle)

---
*Project completed as part of Natural Language Processing coursework*
