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
import logging
from sklearn.preprocessing import LabelEncoder

def load_data(path='merged_data.csv', sample=1):
    """
    Load and merge data from CSV files, with an option to sample a percentage of the data.

    Parameters
    ----------
    path : str, optional
        Path to the merged data CSV file, by default 'merged_data.csv'.
    sample : float, optional
        Percentage of data to sample (0.0 to 1.0), by default 1 (100%).

    Returns
    -------
    tuple
        A tuple containing two pandas DataFrames:
        - merged_data: The merged customer and review data.
        - df_labelled_reviews: The labelled reviews data.
    """
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
    
    # Sample the data if sample < 1
    if sample < 1:
        merged_data = merged_data.sample(frac=sample, random_state=42)
        df_labelled_reviews = df_labelled_reviews.sample(frac=sample, random_state=42)
    
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

def custom_train_test_split(inputs, labels, test_size=0.2, random_state=42):
    """
    Perform a custom train-test split on the dataset.

    Parameters
    ----------
    inputs : dict
        Dictionary containing 'input_ids' and 'attention_mask' tensors.
    labels : torch.Tensor
        Tensor of encoded labels.
    test_size : float, optional
        Proportion of the dataset to include in the test split, by default 0.2.
    random_state : int, optional
        Random state for reproducibility, by default 42.

    Returns
    -------
    dict
        Dictionary containing train and test splits for inputs and labels.
    """
    dataset_size = len(labels)
    indices = list(range(dataset_size))
    
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    
    # Shuffle indices
    torch.randperm(dataset_size, out=torch.LongTensor(indices))
    
    # Calculate split index
    split_idx = int(np.floor(test_size * dataset_size))
    
    # Split indices
    train_indices, test_indices = indices[split_idx:], indices[:split_idx]
    
    # Create train and test splits
    X_train = {
        'input_ids': inputs['input_ids'][train_indices],
        'attention_mask': inputs['attention_mask'][train_indices]
    }
    X_test = {
        'input_ids': inputs['input_ids'][test_indices],
        'attention_mask': inputs['attention_mask'][test_indices]
    }
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    
    return X_train, X_test, y_train, y_test

def train_sentiment_model_transformer(data, logging_level=logging.INFO):
    """
    Train a sentiment model using DistilBERT transformer.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'review_text' and 'sentiment_label' columns.
    logging_level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG), by default logging.INFO.

    Returns
    -------
    model : DistilBertForSequenceClassification
        Trained DistilBERT model.
    tokenizer : DistilBertTokenizer
        DistilBERT tokenizer.
    X_test : dict
        Test set inputs.
    y_test : np.ndarray
        Test set labels.
    """
    logging.basicConfig(level=logging_level)
    logger = logging.getLogger(__name__)

    logger.info("Initializing DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)

    logger.info(f"Data shape: {data.shape}")
    logger.info(f"Columns: {data.columns}")

    # Ensure 'review_text' and 'sentiment_label' columns exist
    required_columns = ['review_text', 'sentiment_label']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Data must contain columns: {required_columns}")

    # Remove rows with NaN values
    data = data.dropna(subset=required_columns)
    logger.info(f"Shape after dropping NaN values: {data.shape}")

    logger.info("Tokenizing inputs...")
    inputs = tokenizer(data['review_text'].tolist(), padding=True, truncation=True, return_tensors='pt')
    
    logger.info("Encoding labels...")
    label_encoder = LabelEncoder()
    labels = torch.tensor(label_encoder.fit_transform(data['sentiment_label']))

    logger.info(f"Input size: {len(inputs['input_ids'])}, Labels size: {len(labels)}")

    logger.info("Performing train/test split...")
    # X_train, X_test, y_train, y_test = train_test_split(
    #     {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
    #     labels,
    #     test_size=0.2,
    #     random_state=42
    # )

    logger.info("Performing custom train/test split...")

    # Perform custom train/test split
    X_train, X_test, y_train, y_test = custom_train_test_split(
        {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']},
        labels,
        test_size=0.2,
        random_state=42
    )

    logger.info(f"Train set size: {len(y_train)}, Test set size: {len(y_test)}")
    
    logger.info("Preparing datasets for Trainer...")
    class Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = Dataset(
        {'input_ids': X_train['input_ids'], 'attention_mask': X_train['attention_mask']},
        y_train
    )
    eval_dataset = Dataset(
        {'input_ids': X_test['input_ids'], 'attention_mask': X_test['attention_mask']},
        y_test
    )

    logger.info("Setting up training arguments...")
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

    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Training completed.")
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
    df_customer_and_review, df_labelled_reviews = load_data(sample=0.01)
    
    # Train the ML model
    # model, vectorizer, X_test, y_test = train_sentiment_model(df_labelled_reviews)
    # evaluate_model(model, X_test, y_test)
    
    # Train the transformer pretrained model
    print("\nTransformers model training:")
    model, tokenizer, X_test, y_test = train_sentiment_model_transformer(df_labelled_reviews, logging_level=logging.INFO)
    

    evaluate_model(model, X_test, y_test)
    vectorizer = tokenizer
    df_customer_and_review = identify_detractors_promoters(df_customer_and_review, model, vectorizer)
    analyze_customer_segments(df_customer_and_review)
    generate_wordclouds(df_customer_and_review)
    
    print("\nSuggestions for marketing campaigns:")
    print("1. For Detractors: Address common issues found in negative reviews and offer personalized solutions.")
    print("2. For Promoters: Implement a referral program to leverage their positive sentiment.")
    print("3. For both: Create targeted email campaigns with content tailored to each group's sentiment and purchase history.")

if __name__ == "__main__":
    main()