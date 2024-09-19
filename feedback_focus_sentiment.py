import datetime
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from wordcloud import WordCloud
import os
import pandas as pd
import yaml
import torch
import logging
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Add this near the top of the file, after imports
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        # merged_data = merged_data.sample(frac=sample, random_state=42)
        df_labelled_reviews = df_labelled_reviews.sample(frac=sample, random_state=42)
    
    return merged_data, df_labelled_reviews

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def train_sentiment_model(data, model_type='MultinomialNB'):
    """
    Train a sentiment model using the specified model type.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'review_text' and 'sentiment_label' columns.
    model_type : str, optional
        Type of model to use ('MultinomialNB', 'RandomForest', 'LogisticRegression', 'DecisionTree'), by default 'MultinomialNB'.

    Returns
    -------
    tuple
        Trained model, vectorizer, X_test, y_test.
    """
    vectorizer = TfidfVectorizer(max_features=5000, preprocessor=preprocess_text)
    X = vectorizer.fit_transform(data['review_text'])
    y = data['sentiment_label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'RandomForest':
        model = RandomForestClassifier()
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier()
    else:
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

def evaluate_model_transformer(model, dataset, tokenizer, batch_size=32):
    """
    Evaluate the transformer model on the test set.

    Parameters
    ----------
    model : transformers.PreTrainedModel
        The trained transformer model.
    dataset : torch.utils.data.Dataset
        The dataset to evaluate on.
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer used for the model.
    batch_size : int, optional
        Batch size for evaluation, by default 32.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays for sklearn metrics
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    # Generate classification report and confusion matrix
    class_report = classification_report(all_labels, all_predictions, target_names=['negative', 'neutral', 'positive'])
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    # Log results
    logger.info("Model Evaluation Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info("\nClassification Report:\n" + class_report)
    logger.info("\nConfusion Matrix:\n" + str(conf_matrix))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix
    }

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}

def train_sentiment_model_transformer(data, logging_level=logging.INFO):
    """
    Train a sentiment model using BERT transformer.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing 'review_text' and 'sentiment_label' columns.
    logging_level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG), by default logging.INFO.

    Returns
    -------
    model : BertForSequenceClassification
        Trained BERT model.
    tokenizer : BertTokenizer
        BERT tokenizer.
    X_test : dict
        Test set inputs.
    y_test : np.ndarray
        Test set labels.
    """
    logging.basicConfig(level=logging_level)
    logger = logging.getLogger(__name__)

    # Set up parameters
    bert_model_name = 'bert-base-uncased'
    num_classes = 3  # Assuming 3 classes: negative, neutral, positive
    max_length = 128
    batch_size = 32
    num_epochs = 3
    learning_rate = 2e-5

    logger.info(f"Initializing {bert_model_name} tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes)

    # Prepare data
    texts = data['review_text'].tolist()
    labels = pd.get_dummies(data['sentiment_label']).values.argmax(1)  # Convert to numerical labels

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Set up device, optimizer, and scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Training loop
    logger.info("Starting training...")
    global_step = 0
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        
        # Create a progress bar for each epoch
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # Log every 100 steps
            if global_step % 100 == 0:
                logger.info(f"Step {global_step}: loss = {loss.item():.4f}")

        progress_bar.close()

        # Evaluation
        logger.info("Running evaluation...")
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        eval_steps = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
                _, preds = torch.max(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
                eval_steps += 1

        val_loss /= eval_steps
        accuracy = accuracy_score(val_labels, val_preds)
        report = classification_report(val_labels, val_preds)
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{report}")

    logger.info("Training completed.")
    
    # Save the model
    save_path = os.path.join(os.path.dirname(__file__), 'saved_models')
    os.makedirs(save_path, exist_ok=True)
    model_save_path = os.path.join(save_path, 'sentiment_transformer_model.pth')
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model.config,
        'tokenizer': tokenizer
    }, model_save_path)
    
    logger.info(f"Model saved to {model_save_path}")
    
    return model, tokenizer, val_dataset, val_labels

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
def identify_detractors_promoters(data, model, vectorizer=None, tokenizer=None):
    # Fill NaN values in 'review_text' with an empty string
    data['review_text'] = data['review_text'].fillna('')
    # only for test TODO use only 32 examples
    # print(str(data.shape) + "data Trimmed for testing only 32 examples") 
    # data = data.sample(n=32)

    if vectorizer is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LOG] Using device: {device}")
        # Move model to the detected device
        model = model.to(device)

        # Ensure data is on the same device as the model
        def to_device(batch):
            return {k: v.to(device) for k, v in batch.items()}

        X = tokenizer(data['review_text'].tolist(), padding=True, truncation=True, return_tensors='pt')
        X = to_device(X)

        batch_size = 128  # Adjust this value based on your GPU memory
        predictions = []
        sentiment_score = []

        with torch.no_grad():
            for i in tqdm(range(0, len(X['input_ids']), batch_size), desc="Processing batches", unit="batch"):
                batch = {k: v[i:i+batch_size].to(device) for k, v in X.items()}
                outputs = model(**batch)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions.extend(torch.argmax(probabilities, dim=-1).cpu().numpy())
                sentiment_score.extend(probabilities.cpu().numpy())

        predictions = np.array(predictions)
        sentiment_score = np.array(sentiment_score)
        
        # Add predictions to the data
        data['sentiment'] = predictions
        
        # Create a dictionary of sentiment scores for each class, so the result must be a list of dictionaries of the form {'negative': 0.3, 'neutral': 0.5, 'positive': 0.2}
        sentiment_score_dict = [
            {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2])
            }
            for scores in sentiment_score
        ]
        
        # Add the sentiment score dictionary to the DataFrame
        data['sentiment_score'] = sentiment_score_dict
        
        # Determine customer type based on sentiment scores
        def get_customer_type(neg_score, pos_score):
            if neg_score > 0.3:
                return 'Detractor'
            elif pos_score > 0.7:
                return 'Promoter'
            else:
                return 'Neutral'
        
        data['customer_type'] = data.apply(lambda row: get_customer_type(
            row['sentiment_score'].get('negative', 0),
            row['sentiment_score'].get('positive', 0)
        ), axis=1)
        
    else:
        # Transform the review text using the vectorizer
        X = vectorizer.transform(data['review_text'])
            # Predict using the model
        predictions = model.predict(X)
        # Add predictions to the data
        data['sentiment'] = predictions
        data['sentiment_score'] = model.predict_proba(X)[:, 1]
        data['customer_type'] = pd.cut(data['sentiment_score'], 
                                   bins=[-float('inf'), 0.3, 0.7, float('inf')],
                                   labels=['Detractor', 'Neutral', 'Promoter'])

    data_reduced_columns = ['customer_id', 'sentiment_score', 'customer_type']
    data_reduced = data[data_reduced_columns]
    
    return data , data_reduced

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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'gross_price_by_customer_type_{timestamp}.png')

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
    

def main(MODEL_TRANSFORMER=True):
    df_customer_and_review, df_labelled_reviews = load_data(sample=1) #the training data is 10% of the labelled data
    
    if MODEL_TRANSFORMER:
        # Train the transformer pretrained model
        print("\nTransformers model training:")
        model, tokenizer, val_dataset, val_labels = train_sentiment_model_transformer(df_labelled_reviews, logging_level=logging.INFO)
        evaluate_model_transformer(model, val_dataset, tokenizer)
        
        # Load the saved model
        print("\nLoading the saved transformer model...")
        save_path = os.path.join(os.path.dirname(__file__), 'saved_models')
        model_save_path = os.path.join(save_path, 'sentiment_transformer_model.pth')
        
        if not os.path.exists(model_save_path):
            raise FileNotFoundError(f"Model file not found at {model_save_path}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[LOG] Using device: {device}")

        checkpoint = torch.load(model_save_path, map_location=device)
        
        model = AutoModelForSequenceClassification.from_pretrained('google/mobilebert-uncased', num_labels=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        tokenizer = checkpoint['tokenizer']
        
        model.eval()
        
        print(f"[LOG] Model successfully loaded from {model_save_path}")
        
        df_customer_and_review, sentiment_only_scores = identify_detractors_promoters(df_customer_and_review, model, tokenizer=tokenizer)
    else:
        # Train and evaluate multiple ML models
        models = ['MultinomialNB', 'RandomForest', 'LogisticRegression', 'DecisionTree']
        for model_type in models:
            print(f"\nTraining and evaluating {model_type} model:")
            model, vectorizer, X_test, y_test = train_sentiment_model(df_labelled_reviews, model_type=model_type)
            evaluate_model(model, X_test, y_test)
        
        # Use the best performing model for further analysis (for simplicity, using MultinomialNB here)
        model, vectorizer, X_test, y_test = train_sentiment_model(df_labelled_reviews, model_type='MultinomialNB')
        df_customer_and_review, sentiment_only_scores = identify_detractors_promoters(df_customer_and_review, model, vectorizer=vectorizer)

    try:
        analyze_customer_segments(df_customer_and_review)
        generate_wordclouds(df_customer_and_review)
    except Exception as e:
        print(f"[ERROR] Failed to analyze_customer_segments and generate wordclouds: {str(e)}")

    # Save the processed DataFrame to a CSV file
    output_path = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_path, exist_ok=True)
    
    # Determine the model type and adjust file names accordingly
    model_prefix = "DistilBERT_" if MODEL_TRANSFORMER else ""
    output_file = os.path.join(output_path, f'{model_prefix}processed_customer_reviews.csv')
    output_sentiment_only_scores_file = os.path.join(output_path, f'{model_prefix}sentiment_only_scores.csv')
    
    try:
        df_customer_and_review.to_csv(output_file, index=False)
        sentiment_only_scores.to_csv(output_sentiment_only_scores_file, index=False)
        print(f"[LOG] Processed data saved successfully to {output_file}")
        print(f"[LOG] Sentiment scores saved successfully to {output_sentiment_only_scores_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save processed data: {str(e)}")
    
    print("\nSuggestions for marketing campaigns:")
    print("1. For Detractors: Address common issues found in negative reviews and offer personalized solutions.")
    print("2. For Promoters: Implement a referral program to leverage their positive sentiment.")
    print("3. For both: Create targeted email campaigns with content tailored to each group's sentiment and purchase history.")

if __name__ == "__main__":
    main(MODEL_TRANSFORMER=False)