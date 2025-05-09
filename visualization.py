"""
Visualization module for topic modeling results
Generates various visualizations for topics and their distributions
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from utils import load_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TopicVisualizer:
    """Class to visualize topic modeling results"""
    
    def __init__(self, topic_keywords=None, doc_topics_df=None):
        """Initialize with topic modeling results or load from files"""
        self.topic_keywords = topic_keywords
        self.doc_topics_df = doc_topics_df
        self.topic_names = None
    
    def load_topic_data(self, data_dir='.'):
        """Load topic modeling results from files"""
        logger.info("Loading topic modeling results...")
        
        try:
            # Load topic keywords
            self.topic_keywords = load_data(os.path.join(data_dir, 'topic_keywords.pkl'))
            
            # Load document-topic dataframe
            doc_topics_path = os.path.join(data_dir, 'doc_topics.csv')
            if os.path.exists(doc_topics_path):
                self.doc_topics_df = pd.read_csv(doc_topics_path)
                logger.info(f"Loaded document-topic data: {len(self.doc_topics_df)} documents")
            else:
                logger.warning(f"Document-topic file not found: {doc_topics_path}")
            
            if self.topic_keywords is not None:
                logger.info(f"Loaded topic keywords for {len(self.topic_keywords)} topics")
                return True
            else:
                return False
        
        except Exception as e:
            logger.error(f"Error loading topic data: {e}")
            return False
    
    def assign_topic_names(self, names=None):
        """Assign descriptive names to topics based on keywords or provided names"""
        if self.topic_keywords is None:
            logger.error("Topic keywords not available")
            return False
        
        num_topics = len(self.topic_keywords)
        
        if names is not None and len(names) == num_topics:
            # Use provided names
            self.topic_names = names
        else:
            # Generate names based on top keywords
            self.topic_names = {}
            for topic in self.topic_keywords:
                topic_id = topic['topic_id']
                # Use top 3 keywords as the topic name
                name = ' & '.join(topic['words'][:3])
                self.topic_names[topic_id] = name
        
        logger.info(f"Assigned names to {len(self.topic_names)} topics")
        return True
    
    def generate_wordclouds(self, output_dir='topic_wordclouds'):
        """Generate wordcloud visualizations for each topic"""
        if self.topic_keywords is None:
            logger.error("Topic keywords not available")
            return False
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Generating wordclouds for {len(self.topic_keywords)} topics...")
        
        # Create a wordcloud for each topic
        for topic in self.topic_keywords:
            topic_id = topic['topic_id']
            words = topic['words']
            probs = topic['probs']
            
            # Create dictionary of word frequencies
            word_freq = {words[i]: probs[i] for i in range(len(words))}
            
            # Generate wordcloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color='white',
                max_words=100,
                colormap='viridis',
                prefer_horizontal=1.0
            ).generate_from_frequencies(word_freq)
            
            # Plot and save
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            
            # Use topic name if available
            if self.topic_names and topic_id in self.topic_names:
                plt.title(f"Topic {topic_id}: {self.topic_names[topic_id]}", fontsize=16)
            else:
                plt.title(f"Topic {topic_id}", fontsize=16)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'topic_{topic_id}_wordcloud.png'), dpi=300)
            plt.close()
        
        logger.info(f"Saved wordclouds to {output_dir}")
        return True
    
    def plot_topic_distribution(self, output_file='topic_distribution.png'):
        """Plot distribution of dominant topics across documents"""
        if self.doc_topics_df is None or 'dominant_topic' not in self.doc_topics_df.columns:
            logger.error("Document-topic data not available or missing dominant_topic column")
            return False
        
        # Count documents by dominant topic
        topic_counts = self.doc_topics_df['dominant_topic'].value_counts().sort_index()
        
        # Get topic labels
        if self.topic_names:
            labels = [f"Topic {idx}: {self.topic_names[idx]}" for idx in topic_counts.index]
        else:
            labels = [f"Topic {idx}" for idx in topic_counts.index]
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=topic_counts.index, y=topic_counts.values, palette='viridis')
        plt.title('Distribution of Documents by Dominant Topic', fontsize=16)
        plt.xlabel('Topic', fontsize=14)
        plt.ylabel('Number of Documents', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        # Set tick labels
        ax.set_xticklabels(labels)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        logger.info(f"Saved topic distribution plot to {output_file}")
        return True
    
    def plot_category_topic_heatmap(self, output_file='category_topic_heatmap.png'):
        """Plot heatmap of topic distribution across categories"""
        if self.doc_topics_df is None or 'dominant_topic' not in self.doc_topics_df.columns:
            logger.error("Document-topic data not available or missing dominant_topic column")
            return False
        
        if 'category' not in self.doc_topics_df.columns:
            logger.error("Category column not found in document-topic data")
            return False
        
        # Create cross-tabulation of categories and topics
        category_topic = pd.crosstab(
            self.doc_topics_df['category'],
            self.doc_topics_df['dominant_topic'],
            normalize='index'
        )
        
        # Generate column labels
        if self.topic_names:
            col_labels = [f"Topic {idx}: {self.topic_names[idx]}" for idx in category_topic.columns]
        else:
            col_labels = [f"Topic {idx}" for idx in category_topic.columns]
        
        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            category_topic,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            cbar_kws={'label': 'Proportion of Documents'},
            xticklabels=col_labels
        )
        plt.title('Topic Distribution Across News Categories', fontsize=16)
        plt.xlabel('Topic', fontsize=14)
        plt.ylabel('Category', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        logger.info(f"Saved category-topic heatmap to {output_file}")
        return True
    
    def plot_topic_keywords(self, num_words=10, output_dir='topic_keywords'):
        """Plot horizontal bar charts of top keywords for each topic"""
        if self.topic_keywords is None:
            logger.error("Topic keywords not available")
            return False
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        logger.info(f"Generating keyword bar charts for {len(self.topic_keywords)} topics...")
        
        # Create a horizontal bar chart for each topic
        for topic in self.topic_keywords:
            topic_id = topic['topic_id']
            words = topic['words'][:num_words]
            probs = topic['probs'][:num_words]
            
            # Create horizontal bar chart
            plt.figure(figsize=(10, 6))
            bars = plt.barh(range(len(words)), probs, align='center', color='skyblue')
            plt.yticks(range(len(words)), words)
            
            # Add probability values to the end of each bar
            for i, (prob, word) in enumerate(zip(probs, words)):
                plt.text(prob + 0.01, i, f"{prob:.4f}", va='center')
            
            # Use topic name if available
            if self.topic_names and topic_id in self.topic_names:
                plt.title(f"Topic {topic_id}: {self.topic_names[topic_id]}", fontsize=16)
            else:
                plt.title(f"Topic {topic_id} - Top Keywords", fontsize=16)
            
            plt.xlabel('Probability', fontsize=14)
            plt.ylabel('Keyword', fontsize=14)
            plt.xlim(0, max(probs) * 1.2)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'topic_{topic_id}_keywords.png'), dpi=300)
            plt.close()
        
        logger.info(f"Saved keyword bar charts to {output_dir}")
        return True
    
    def create_topic_similarity_network(self, output_file='topic_similarity_network.html'):
        """Create interactive topic similarity network visualization using Plotly"""
        if self.doc_topics_df is None:
            logger.error("Document-topic data not available")
            return False
        
        # Filter columns that contain topic probabilities
        topic_cols = [col for col in self.doc_topics_df.columns if col.startswith('topic_') and col.endswith('_prob')]
        
        if not topic_cols:
            logger.error("No topic probability columns found in document-topic data")
            return False
        
        # Calculate correlation matrix between topics
        topic_corr = self.doc_topics_df[topic_cols].corr()
        
        # Extract topic IDs from column names
        topic_ids = [int(col.split('_')[1]) for col in topic_cols]
        
        # Create node labels
        if self.topic_names:
            node_labels = [f"Topic {id}: {self.topic_names[id]}" for id in topic_ids]
        else:
            node_labels = [f"Topic {id}" for id in topic_ids]
        
        # Create network edges
        edges = []
        edge_weights = []
        
        for i in range(len(topic_ids)):
            for j in range(i+1, len(topic_ids)):
                # Only add edges with correlation above threshold
                if abs(topic_corr.iloc[i, j]) > 0.1:
                    edges.append((i, j))
                    edge_weights.append(abs(topic_corr.iloc[i, j]))
        
        # Create network layout using Plotly
        fig = go.Figure()
        
        # Add edges
        for (i, j), weight in zip(edges, edge_weights):
            fig.add_trace(
                go.Scatter(
                    x=[i, j, None],
                    y=[0, 0, None],
                    mode='lines',
                    line=dict(width=weight*5, color='rgba(0,0,0,0.3)'),
                    hoverinfo='none'
                )
            )
        
        # Add nodes
        fig.add_trace(
            go.Scatter(
                x=list(range(len(topic_ids))),
                y=[0] * len(topic_ids),
                mode='markers+text',
                marker=dict(size=20, color=list(range(len(topic_ids))), colorscale='Viridis'),
                text=node_labels,
                textposition='top center',
                hoverinfo='text'
            )
        )
        
        # Update layout
        fig.update_layout(
            title='Topic Similarity Network',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            showlegend=False,
            width=800,
            height=600
        )
        
        # Save to HTML file
        fig.write_html(output_file)
        
        logger.info(f"Saved topic similarity network to {output_file}")
        return True
    
    def generate_all_visualizations(self, output_dir='.'):
        """Generate all visualizations"""
        # Create output directories
        wordclouds_dir = os.path.join(output_dir, 'wordclouds')
        keywords_dir = os.path.join(output_dir, 'keywords')
        
        # Generate wordclouds
        self.generate_wordclouds(wordclouds_dir)
        
        # Plot topic distribution
        self.plot_topic_distribution(os.path.join(output_dir, 'topic_distribution.png'))
        
        # Plot category-topic heatmap
        self.plot_category_topic_heatmap(os.path.join(output_dir, 'category_topic_heatmap.png'))
        
        # Plot topic keywords
        self.plot_topic_keywords(output_dir=keywords_dir)
        
        # Create topic similarity network
        self.create_topic_similarity_network(os.path.join(output_dir, 'topic_similarity_network.html'))
        
        logger.info(f"Generated all visualizations in {output_dir}")
        return True


def main():
    """Main function to run the visualization pipeline"""
    visualizer = TopicVisualizer()
    
    # Load topic data
    if not visualizer.load_topic_data():
        logger.error("Failed to load topic data. Exiting.")
        return
    
    # Assign topic names
    visualizer.assign_topic_names()
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()
    
    logger.info("Visualization complete!")

if __name__ == "__main__":
    main()
