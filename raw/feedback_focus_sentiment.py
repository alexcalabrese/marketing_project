import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from wordcloud import WordCloud

def load_data():
    return pd.read_csv('merged_data.csv')

def preprocess_text(text):
    import re
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def train_sentiment_model(data):
    vectorizer = TfidfVectorizer(max_features=5000, preprocessor=preprocess_text)
    X = vectorizer.fit_transform(data['review_text'])
    y = data['sentiment_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    return model, vectorizer, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def identify_detractors_promoters(data, model, vectorizer):
    X = vectorizer.transform(data['review_text'])
    data['sentiment_score'] = model.predict_proba(X)[:, 1]  # Probability of positive sentiment
    
    data['customer_type'] = pd.cut(data['sentiment_score'], 
                                   bins=[-float('inf'), 0.3, 0.7, float('inf')],
                                   labels=['Detractor', 'Neutral', 'Promoter'])
    
    return data

def analyze_customer_segments(data):
    segment_summary = data.groupby('customer_type').agg({
        'customer_id': 'count',
        'gross_price': 'mean',
        'sentiment_score': 'mean'
    }).rename(columns={'customer_id': 'count'})
    
    print("Customer Segment Summary:")
    print(segment_summary)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='customer_type', y='gross_price', data=data)
    plt.title('Distribution of Gross Price by Customer Type')
    plt.show()

def generate_wordclouds(data):
    def create_wordcloud(text, title):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
    
    detractor_text = ' '.join(data[data['customer_type'] == 'Detractor']['review_text'])
    promoter_text = ' '.join(data[data['customer_type'] == 'Promoter']['review_text'])
    
    create_wordcloud(detractor_text, 'Detractor Word Cloud')
    create_wordcloud(promoter_text, 'Promoter Word Cloud')

def main():
    data = load_data()
    model, vectorizer, X_test, y_test = train_sentiment_model(data)
    evaluate_model(model, X_test, y_test)
    
    data = identify_detractors_promoters(data, model, vectorizer)
    analyze_customer_segments(data)
    generate_wordclouds(data)
    
    print("\nSuggestions for marketing campaigns:")
    print("1. For Detractors: Address common issues found in negative reviews and offer personalized solutions.")
    print("2. For Promoters: Implement a referral program to leverage their positive sentiment.")
    print("3. For both: Create targeted email campaigns with content tailored to each group's sentiment and purchase history.")

if __name__ == "__main__":
    main()