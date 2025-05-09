"""
Topic modeling module for the news articles
Implements Latent Dirichlet Allocation (LDA) to discover topics
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models
import logging
import pickle
import time
from sklearn.decomposition import LatentDirichletAllocation
from utils import load_data, save_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TopicModeler:
    """Class to perform topic modeling on preprocessed news articles"""
    
    def __init__(self, preprocessed_data=None, corpus=None, doc_term_matrix=None, vectorizer=None):
        """Initialize with preprocessed data or load from files"""
        self.preprocessed_data = preprocessed_data
        self.corpus = corpus
        self.doc_term_matrix = doc_term_matrix
        self.vectorizer = vectorizer
        self.feature_names = None
        
        # Models
        self.lda_model = None
        self.gensim_dictionary = None
        self.gensim_corpus = None
        self.gensim_lda_model = None
        
        # Results
        self.topic_keywords = None
        self.doc_topics = None
        
        if vectorizer is not None:
            self.feature_names = vectorizer.get_feature_names_out()
    
    def load_preprocessed_data(self, data_dir='.'):
        """Load preprocessed data from files"""
        logger.info("Loading preprocessed data...")
        
        try:
            # Load processed dataframe
            csv_path = os.path.join(data_dir, 'processed_news_data.csv')
            if os.path.exists(csv_path):
                self.preprocessed_data = pd.read_csv(csv_path)
                logger.info(f"Loaded processed data: {len(self.preprocessed_data)} articles")
            else:
                logger.warning(f"Processed data file not found: {csv_path}")
            
            # Load vectorization results
            self.vectorizer = load_data(os.path.join(data_dir, 'vectorizer.pkl'))
            self.doc_term_matrix = load_data(os.path.join(data_dir, 'doc_term_matrix.pkl'))
            self.feature_names = load_data(os.path.join(data_dir, 'feature_names.pkl'))
            self.corpus = load_data(os.path.join(data_dir, 'corpus.pkl'))
            
            if self.vectorizer is not None:
                logger.info("Loaded vectorizer")
            
            if self.doc_term_matrix is not None:
                logger.info(f"Loaded document-term matrix with shape: {self.doc_term_matrix.shape}")
            
            if all([self.preprocessed_data is not None, 
                    self.doc_term_matrix is not None, 
                    self.vectorizer is not None]):
                return True
            else:
                return False
        
        except Exception as e:
            logger.error(f"Error loading preprocessed data: {e}")
            return False
    
    def prepare_gensim_data(self):
        """Prepare data in gensim format"""
        if self.corpus is None:
            logger.error("No corpus available")
            return False
        
        logger.info("Preparing data for gensim...")
        
        try:
            # Create gensim dictionary
            self.gensim_dictionary = corpora.Dictionary(self.corpus)
            
            # Create gensim corpus
            self.gensim_corpus = [
                self.gensim_dictionary.doc2bow(doc) for doc in self.corpus
            ]
            
            logger.info(f"Created gensim dictionary with {len(self.gensim_dictionary)} terms")
            return True
        
        except Exception as e:
            logger.error(f"Error preparing gensim data: {e}")
            return False
    
    def find_optimal_topics(self, start=2, limit=15, step=1):
        """Find optimal number of topics by coherence score"""
        if self.gensim_corpus is None or self.gensim_dictionary is None:
            logger.error("Gensim data not prepared. Call prepare_gensim_data() first")
            return None
        
        logger.info(f"Finding optimal number of topics (range: {start}-{limit})...")
        
        coherence_values = []
        model_list = []
        
        for num_topics in range(start, limit+1, step):
            logger.info(f"Training LDA model with {num_topics} topics...")
            
            # Train LDA model
            model = LdaModel(
                corpus=self.gensim_corpus,
                id2word=self.gensim_dictionary,
                num_topics=num_topics,
                random_state=42,
                update_every=1,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )
            
            model_list.append(model)
            
            # Calculate coherence score
            coherence_model = CoherenceModel(
                model=model,
                texts=self.corpus,
                dictionary=self.gensim_dictionary,
                coherence='c_v'
            )
            coherence_values.append(coherence_model.get_coherence())
            
            logger.info(f"Topics: {num_topics}, Coherence Score: {coherence_values[-1]:.4f}")
        
        # Find optimal number of topics
        optimal_topics = start + (np.argmax(coherence_values) * step)
        logger.info(f"Optimal number of topics: {optimal_topics} (coherence: {max(coherence_values):.4f})")
        
        # Plot coherence scores
        plt.figure(figsize=(12, 6))
        plt.plot(range(start, limit+1, step), coherence_values)
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.title("Optimal Number of Topics by Coherence Score")
        plt.axvline(x=optimal_topics, color='r', linestyle='--')
        plt.savefig('topic_coherence.png')
        
        # Save coherence data
        coherence_data = {
            'num_topics_range': list(range(start, limit+1, step)),
            'coherence_values': coherence_values,
            'optimal_topics': optimal_topics
        }
        save_data(coherence_data, 'coherence_data.pkl')
        
        return optimal_topics, coherence_values, model_list
    
    def train_lda(self, num_topics=10, random_state=42):
        """Train the LDA model using sklearn"""
        if self.doc_term_matrix is None:
            logger.error("Document-term matrix not available")
            return False
        
        logger.info(f"Training LDA model with {num_topics} topics (sklearn)...")
        
        # Create LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=random_state,
            max_iter=20,
            learning_method='online',
            learning_offset=50.0,
            doc_topic_prior=None,
            topic_word_prior=None
        )
        
        # Fit model
        start_time = time.time()
        self.lda_model.fit(self.doc_term_matrix)
        
        logger.info(f"LDA model training completed in {time.time() - start_time:.2f} seconds")
        
        # Extract topic keywords
        self.extract_topic_keywords()
        
        # Transform documents to topic space
        self.doc_topics = self.lda_model.transform(self.doc_term_matrix)
        
        logger.info("Document-topic matrix created")
        
        return True
    
    def train_gensim_lda(self, num_topics=10, passes=20, random_state=42):
        """Train the LDA model using gensim"""
        if self.gensim_corpus is None or self.gensim_dictionary is None:
            logger.error("Gensim data not prepared. Call prepare_gensim_data() first")
            return False
        
        logger.info(f"Training LDA model with {num_topics} topics (gensim)...")
        
        # Create LDA model
        self.gensim_lda_model = LdaModel(
            corpus=self.gensim_corpus,
            id2word=self.gensim_dictionary,
            num_topics=num_topics,
            random_state=random_state,
            update_every=1,
            passes=passes,
            alpha='auto',
            per_word_topics=True
        )
        
        logger.info("Gensim LDA model training completed")
        
        return True
    
    def extract_topic_keywords(self, num_words=20):
        """Extract top words for each topic from the LDA model"""
        if self.lda_model is None or self.feature_names is None:
            logger.error("LDA model or feature names not available")
            return None
        
        # Get top words for each topic
        topic_keywords = []
        topic_word_distributions = self.lda_model.components_
        
        for topic_idx, topic in enumerate(topic_word_distributions):
            # Sort words by probability and get top words
            top_word_indices = topic.argsort()[:-num_words-1:-1]
            top_words = [self.feature_names[i] for i in top_word_indices]
            word_probs = [topic[i] for i in top_word_indices]
            
            # Normalize probabilities
            word_probs = word_probs / np.sum(word_probs)
            
            topic_keywords.append({
                'topic_id': topic_idx,
                'words': top_words,
                'probs': word_probs.tolist()
            })
        
        self.topic_keywords = topic_keywords
        return topic_keywords
    
    def get_document_topics(self):
        """Get topic distribution for each document"""
        if self.doc_topics is None or self.preprocessed_data is None:
            logger.error("Document-topic matrix or preprocessed data not available")
            return None
        
        # Get dominant topic for each document
        dominant_topics = np.argmax(self.doc_topics, axis=1)
        
        # Create dataframe with document-topic information
        doc_topic_df = self.preprocessed_data.copy()
        doc_topic_df['dominant_topic'] = dominant_topics
        
        # Add topic distribution
        for topic_idx in range(self.doc_topics.shape[1]):
            doc_topic_df[f'topic_{topic_idx}_prob'] = self.doc_topics[:, topic_idx]
        
        return doc_topic_df
    
    def create_pyldavis(self, output_file='lda_visualization.html'):
        """Create interactive visualization of topics using pyLDAvis"""
        if self.gensim_lda_model is None:
            logger.error("Gensim LDA model not available. Call train_gensim_lda() first")
            return False
        
        logger.info("Creating pyLDAvis visualization...")
        
        try:
            # Prepare visualization
            vis_data = pyLDAvis.gensim_models.prepare(
                self.gensim_lda_model, 
                self.gensim_corpus, 
                self.gensim_dictionary
            )
            
            # Save visualization to HTML file
            pyLDAvis.save_html(vis_data, output_file)
            
            logger.info(f"Saved pyLDAvis visualization to {output_file}")
            return True
        
        except Exception as e:
            logger.error(f"Error creating pyLDAvis visualization: {e}")
            return False
    
    def save_model(self, output_dir='.'):
        """Save LDA model and results"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save sklearn LDA model
        if self.lda_model is not None:
            save_data(self.lda_model, os.path.join(output_dir, 'lda_model.pkl'))
        
        # Save gensim LDA model
        if self.gensim_lda_model is not None:
            self.gensim_lda_model.save(os.path.join(output_dir, 'gensim_lda_model'))
        
        # Save topic keywords
        if self.topic_keywords is not None:
            save_data(self.topic_keywords, os.path.join(output_dir, 'topic_keywords.pkl'))
        
        # Save document-topic matrix
        if self.doc_topics is not None:
            save_data(self.doc_topics, os.path.join(output_dir, 'doc_topics.pkl'))
        
        # Save document-topic dataframe
        doc_topic_df = self.get_document_topics()
        if doc_topic_df is not None:
            doc_topic_df.to_csv(os.path.join(output_dir, 'doc_topics.csv'), index=False)
        
        logger.info(f"Saved LDA model and results to {output_dir}")
        return True

def main():
    """Main function to run the topic modeling pipeline"""
    topic_modeler = TopicModeler()
    
    # Load preprocessed data
    if not topic_modeler.load_preprocessed_data():
        logger.error("Failed to load preprocessed data. Exiting.")
        return
    
    # Prepare gensim data
    if not topic_modeler.prepare_gensim_data():
        logger.error("Failed to prepare gensim data. Exiting.")
        return
    
    # Find optimal number of topics
    try:
        optimal_topics, _, _ = topic_modeler.find_optimal_topics(start=5, limit=20, step=1)
    except Exception as e:
        logger.error(f"Error finding optimal topics: {e}")
        optimal_topics = 10  # Default if error occurs
    
    # Train sklearn LDA model
    if not topic_modeler.train_lda(num_topics=optimal_topics):
        logger.error("Failed to train sklearn LDA model. Exiting.")
        return
    
    # Train gensim LDA model
    if not topic_modeler.train_gensim_lda(num_topics=optimal_topics):
        logger.error("Failed to train gensim LDA model. Exiting.")
        return
    
    # Create pyLDAvis visualization
    topic_modeler.create_pyldavis()
    
    # Save model and results
    topic_modeler.save_model()
    
    logger.info("Topic modeling complete!")

if __name__ == "__main__":
    main()
