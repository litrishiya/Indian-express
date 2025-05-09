"""
Data preprocessing module for topic modeling on news articles
Loads, cleans, and prepares text data for topic modeling
"""

import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import logging
import pickle
import time
from utils import (
    download_nltk_resources, 
    load_spacy_model,
    get_all_stopwords, 
    clean_text, 
    tokenize_and_lemmatize,
    load_and_combine_datasets,
    save_data
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NewsDataPreprocessor:
    """Class to preprocess news article data for topic modeling"""
    
    def __init__(self, data_path='archive'):
        """Initialize with path to data directory"""
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.corpus = None
        self.dictionary = None
        self.vectorizer = None
        self.feature_names = None
        self.doc_term_matrix = None
        
        # Ensure NLTK resources are downloaded
        download_nltk_resources()
        
        # Load spaCy model
        self.nlp = load_spacy_model()
        
        # Get stopwords
        self.stopwords = get_all_stopwords()
    
    def load_data(self):
        """Load and combine all news datasets"""
        self.raw_data = load_and_combine_datasets(self.data_path)
        if self.raw_data is not None:
            logger.info(f"Loaded {len(self.raw_data)} news articles")
            # Check for required columns
            required_cols = ['headlines', 'content', 'category']
            missing_cols = [col for col in required_cols if col not in self.raw_data.columns]
            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}")
                
                # Try to handle missing columns
                if 'headlines' not in self.raw_data.columns and 'title' in self.raw_data.columns:
                    self.raw_data.rename(columns={'title': 'headlines'}, inplace=True)
                
                if 'content' not in self.raw_data.columns and 'text' in self.raw_data.columns:
                    self.raw_data.rename(columns={'text': 'content'}, inplace=True)
            
            return True
        return False
    
    def preprocess_data(self):
        """Clean and preprocess the news articles"""
        if self.raw_data is None:
            logger.error("No data loaded. Call load_data() first")
            return False
        
        logger.info("Preprocessing data...")
        start_time = time.time()
        
        # Create a copy of the raw data
        self.processed_data = self.raw_data.copy()
        
        # Handle missing values
        for col in ['headlines', 'description', 'content']:
            if col in self.processed_data.columns:
                self.processed_data[col] = self.processed_data[col].fillna('')
        
        # Combine headlines, description, and content for better topic modeling
        if 'description' in self.processed_data.columns:
            self.processed_data['text'] = (
                self.processed_data['headlines'] + ' ' + 
                self.processed_data['description'] + ' ' + 
                self.processed_data['content']
            )
        else:
            self.processed_data['text'] = (
                self.processed_data['headlines'] + ' ' + 
                self.processed_data['content']
            )
        
        # Clean text
        logger.info("Cleaning text...")
        self.processed_data['cleaned_text'] = self.processed_data['text'].apply(clean_text)
        
        # Tokenize and lemmatize
        logger.info("Tokenizing and lemmatizing...")
        self.processed_data['tokens'] = self.processed_data['cleaned_text'].apply(
            lambda x: tokenize_and_lemmatize(x, self.stopwords)
        )
        
        # Convert tokens back to text for vectorization
        self.processed_data['processed_text'] = self.processed_data['tokens'].apply(lambda x: ' '.join(x))
        
        # Remove articles with very short processed text (less than 10 characters)
        self.processed_data = self.processed_data[
            self.processed_data['processed_text'].str.len() > 10
        ].reset_index(drop=True)
        
        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Processed {len(self.processed_data)} articles")
        
        return True
    
    def vectorize_text(self, method='tfidf', max_df=0.95, min_df=2, ngram_range=(1, 1), max_features=5000):
        """
        Vectorize the processed text using either CountVectorizer or TfidfVectorizer
        
        Args:
            method: Either 'count' or 'tfidf'
            max_df: Ignore terms that appear in more than max_df percent of documents
            min_df: Ignore terms that appear in fewer than min_df documents
            ngram_range: The lower and upper boundary of the range of n-values for n-grams
            max_features: Maximum number of features to keep
        """
        if self.processed_data is None:
            logger.error("No processed data. Call preprocess_data() first")
            return False
        
        logger.info(f"Vectorizing text using {method.upper()} method...")
        
        if method.lower() == 'count':
            self.vectorizer = CountVectorizer(
                max_df=max_df,
                min_df=min_df,
                ngram_range=ngram_range,
                max_features=max_features
            )
        else:  # Default to TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_df=max_df,
                min_df=min_df,
                ngram_range=ngram_range,
                max_features=max_features
            )
        
        # Create document-term matrix
        self.doc_term_matrix = self.vectorizer.fit_transform(self.processed_data['processed_text'])
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        logger.info(f"Created document-term matrix with shape: {self.doc_term_matrix.shape}")
        
        # Create corpus and dictionary for gensim
        self.corpus = [doc.split() for doc in self.processed_data['processed_text']]
        
        return True
    
    def save_processed_data(self, output_dir='.'):
        """Save processed data and vectorization results"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save processed dataframe
        self.processed_data.to_csv(os.path.join(output_dir, 'processed_news_data.csv'), index=False)
        
        # Save vectorizer and document-term matrix
        save_data(self.vectorizer, os.path.join(output_dir, 'vectorizer.pkl'))
        save_data(self.doc_term_matrix, os.path.join(output_dir, 'doc_term_matrix.pkl'))
        save_data(self.feature_names, os.path.join(output_dir, 'feature_names.pkl'))
        save_data(self.corpus, os.path.join(output_dir, 'corpus.pkl'))
        
        logger.info(f"Saved preprocessed data and vectorization results to {output_dir}")
        return True

def main():
    """Main function to run the preprocessing pipeline"""
    preprocessor = NewsDataPreprocessor()
    
    # Load data
    if not preprocessor.load_data():
        logger.error("Failed to load data. Exiting.")
        return
    
    # Preprocess data
    if not preprocessor.preprocess_data():
        logger.error("Failed to preprocess data. Exiting.")
        return
    
    # Vectorize text
    if not preprocessor.vectorize_text(method='tfidf'):
        logger.error("Failed to vectorize text. Exiting.")
        return
    
    # Save processed data
    preprocessor.save_processed_data()
    
    logger.info("Preprocessing complete!")

if __name__ == "__main__":
    main()
