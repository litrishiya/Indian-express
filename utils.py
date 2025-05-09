"""
Utility functions for the topic modeling project
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import pickle
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
def download_nltk_resources():
    """Download required NLTK resources if not already present"""
    try:
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
                logger.info(f"NLTK resource '{resource}' already downloaded")
            except LookupError:
                nltk.download(resource)
                logger.info(f"Downloaded NLTK resource: {resource}")
    except Exception as e:
        logger.error(f"Error downloading NLTK resources: {e}")

# Load spaCy model
def load_spacy_model():
    """Load spaCy model for text processing"""
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        logger.info("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")
    except Exception as e:
        logger.error(f"Error loading spaCy model: {e}")
        return None

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Custom stopwords for news articles
CUSTOM_STOPWORDS = {
    'said', 'would', 'also', 'one', 'two', 'many', 'like', 'even', 'get', 'get', 'going',
    'could', 'may', 'according', 'says', 'will', 'since', 'still', 'however', 'made',
    'make', 'much', 'use', 'used', 'using', 'lot', 'year', 'years', 'day', 'days',
    'month', 'months', 'say', 'says', 'said', 'time', 'times', 'jan', 'feb', 'mar',
    'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'advertisement', 
    'people', 'percent', 'first', 'last', 'later', 'earlier', 'earlier', 'new', 'news',
    'india', 'indian', 'express'
}

def get_all_stopwords():
    """Get combined list of stopwords (NLTK + custom)"""
    try:
        nltk_stopwords = set(stopwords.words('english'))
        all_stopwords = nltk_stopwords.union(CUSTOM_STOPWORDS)
        return all_stopwords
    except Exception as e:
        logger.error(f"Error getting stopwords: {e}")
        return CUSTOM_STOPWORDS

def clean_text(text):
    """Clean text by removing special characters, URLs, and normalizing"""
    if not isinstance(text, str) or pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S*@\S*\s?', '', text)
    
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_and_lemmatize(text, all_stopwords=None):
    """Tokenize and lemmatize text, removing stopwords"""
    if all_stopwords is None:
        all_stopwords = get_all_stopwords()
    
    if not isinstance(text, str) or not text:
        return []
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens 
              if token not in all_stopwords and len(token) > 2]
    
    return tokens

def load_and_combine_datasets(data_path='archive'):
    """Load and combine all datasets from the archive folder"""
    logger.info("Loading datasets...")
    
    try:
        data_files = [
            'sports_data.csv',
            'business_data.csv',
            'education_data.csv',
            'technology_data.csv',
            'entertainment_data.csv'
        ]

        all_data = []
        for file in data_files:
            file_path = os.path.join(data_path, file)
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Ensure the category column exists or add it based on filename
                if 'category' not in df.columns:
                    category = file.split('_')[0]
                    df['category'] = category
                all_data.append(df)
            else:
                logger.warning(f"Data file not found: {file_path}")
        
        if not all_data:
            logger.error("No data files found!")
            return None
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Successfully loaded {len(combined_df)} articles from {len(all_data)} categories")
        
        return combined_df
    except Exception as e:
        logger.error(f"Error loading datasets: {e}")
        return None

def save_data(data, filename):
    """Save data to pickle file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Data saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving data to {filename}: {e}")
        return False

def load_data(filename):
    """Load data from pickle file"""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Data loaded from {filename}")
        return data
    except FileNotFoundError:
        logger.warning(f"File not found: {filename}")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {filename}: {e}")
        return None
