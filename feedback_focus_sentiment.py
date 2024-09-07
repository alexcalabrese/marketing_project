import pandas as pd
import numpy as np
import re
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from wordcloud import WordCloud
import os
import pandas as pd
import yaml
import torch

def load_data(path='merged_data.csv'):
    with open('configs.yaml', 'r') as file:
         config = yaml.safe_load(file)
    file_paths = config['data_paths']
    base_path = file_paths['base_path']
    
    full_path = os.path.join(base_path, path)
    
    merged_data = pd.read_csv(full_path)
    merged_data['customer_id'] = merged_data['customer_id'].astype(str)

    df_customer_reviews = pd.read_csv(os.path.join(base_path, file_paths['cleaned_customer_reviews']), dtype={
        'review_id': str,
        'customer_id': str,
        'review_text': str
    })

    df_labelled_reviews = pd.read_csv(os.path.join(base_path, file_paths['cleaned_labelled_reviews']), dtype={
        'labelled_reviews_index': str,
        'review_text': str,
        'sentiment_label': str
    })
    
    # Merge the loaded data with reviews
    merged_data = pd.merge(merged_data, df_customer_reviews, on='customer_id', how='left')
    
    return merged_data, df_labelled_reviews

def preprocess_text(text):
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

def train_sentiment_model_transformer(data, logging=True):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    # Tokenize the inputs
    inputs = tokenizer(data['review_text'].tolist(), padding=True, truncation=True, return_tensors='pt')
    labels = data['sentiment_label'].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

    # Prepare datasets for Trainer (assuming appropriate padding and tensor transformation)
    train_dataset = torch.utils.data.TensorDataset(X_train['input_ids'], X_train['attention_mask'], y_train)
    eval_dataset = torch.utils.data.TensorDataset(X_test['input_ids'], X_test['attention_mask'], y_test)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch"
    )
    if logging:
        print("[LOG] Training started...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    trainer.train()

    return model, tokenizer, X_test, y_test



def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

def identify_detractors_promoters(data, model, vectorizer):
    # Fill NaN values in 'review_text' with an empty string
    data['review_text'] = data['review_text'].fillna('')
    
    # Transform the review text using the vectorizer
    X = vectorizer.transform(data['review_text'])
    
    # Predict using the model
    predictions = model.predict(X)
    
    # Add predictions to the data
    data['sentiment'] = predictions
    
    # return data
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

def generate_wordclouds(data, config_path='configs.yaml'):
    def create_wordcloud(text, title):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.show()
        # Save the word cloud to a file in the base path of configs.yaml
        with open('configs.yaml', 'r') as file:
            config = yaml.safe_load(file)
        base_path = config['data_paths']['base_path']
        wordcloud_path = os.path.join(base_path, f'{title}.png')
        wordcloud.to_file(wordcloud_path)
        print(f'[LOG] Word cloud saved to {wordcloud_path} / {title}.png')
        
        
    
    detractor_text = ' '.join(data[data['customer_type'] == 'Detractor']['review_text'])
    promoter_text = ' '.join(data[data['customer_type'] == 'Promoter']['review_text'])
    
    create_wordcloud(detractor_text, 'Detractor Word Cloud')
    create_wordcloud(promoter_text, 'Promoter Word Cloud')

def main():
    df_customer_and_review, df_labelled_reviews = load_data()
    # model, vectorizer, X_test, y_test = train_sentiment_model(df_labelled_reviews)
    # evaluate_model(model, X_test, y_test)
    print("\nTransformers model training:")
    model, vectorizer, X_test, y_test = train_sentiment_model_transformer(df_labelled_reviews)
    evaluate_model(model, X_test, y_test)
    
    df_customer_and_review = identify_detractors_promoters(df_customer_and_review, model, vectorizer)
    analyze_customer_segments(df_customer_and_review)
    generate_wordclouds(df_customer_and_review)
    
    print("\nSuggestions for marketing campaigns:")
    print("1. For Detractors: Address common issues found in negative reviews and offer personalized solutions.")
    print("2. For Promoters: Implement a referral program to leverage their positive sentiment.")
    print("3. For both: Create targeted email campaigns with content tailored to each group's sentiment and purchase history.")

if __name__ == "__main__":
    main()