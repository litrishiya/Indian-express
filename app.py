"""
Streamlit web application for topic modeling on news articles
Provides interactive exploration of topics and visualizations
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import time
import random
from PIL import Image
import base64
from io import BytesIO

# Import project modules
from utils import load_data
from topic_modeling import TopicModeler
from visualization import TopicVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set page title and config
st.set_page_config(
    page_title="News Topic Explorer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📰 News Topic Explorer")
st.markdown("""
This application uses topic modeling to discover hidden themes in news articles from The Indian Express.
Explore topics across different categories like Sports, Business, Education, Technology, and Entertainment.
""")

# Function to load data
@st.cache_data
def load_processed_data():
    """Load preprocessed data and topic modeling results"""
    try:
        # Load document-topic dataframe
        doc_topics_path = 'doc_topics.csv'
        if os.path.exists(doc_topics_path):
            doc_topics_df = pd.read_csv(doc_topics_path)
            
            # Load topic keywords
            topic_keywords = load_data('topic_keywords.pkl')
            
            return doc_topics_df, topic_keywords
        else:
            return None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Function to run the pipeline if data doesn't exist
def run_pipeline():
    """Run the complete topic modeling pipeline"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Data Preprocessing
    status_text.text("Step 1/3: Preprocessing data...")
    try:
        from data_preprocessing import NewsDataPreprocessor
        preprocessor = NewsDataPreprocessor()
        preprocessor.load_data()
        preprocessor.preprocess_data()
        preprocessor.vectorize_text(method='tfidf')
        preprocessor.save_processed_data()
        progress_bar.progress(33)
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return False
    
    # Step 2: Topic Modeling
    status_text.text("Step 2/3: Running topic modeling...")
    try:
        from topic_modeling import TopicModeler
        topic_modeler = TopicModeler()
        topic_modeler.load_preprocessed_data()
        topic_modeler.prepare_gensim_data()
        
        # Use a predefined number of topics to save time
        num_topics = 10
        status_text.text(f"Training LDA model with {num_topics} topics...")
        
        topic_modeler.train_lda(num_topics=num_topics)
        topic_modeler.train_gensim_lda(num_topics=num_topics)
        topic_modeler.save_model()
        progress_bar.progress(67)
    except Exception as e:
        st.error(f"Error in topic modeling: {e}")
        return False
    
    # Step 3: Visualization
    status_text.text("Step 3/3: Generating visualizations...")
    try:
        from visualization import TopicVisualizer
        visualizer = TopicVisualizer()
        visualizer.load_topic_data()
        visualizer.assign_topic_names()
        visualizer.generate_all_visualizations()
        progress_bar.progress(100)
        status_text.text("Pipeline completed successfully!")
    except Exception as e:
        st.error(f"Error in visualization: {e}")
        return False
    
    return True

# Check if data exists or needs to be processed
doc_topics_df, topic_keywords = load_processed_data()

if doc_topics_df is None or topic_keywords is None:
    st.warning("Topic modeling results not found. Running pipeline...")
    if run_pipeline():
        st.success("Pipeline completed. Reloading data...")
        doc_topics_df, topic_keywords = load_processed_data()
    else:
        st.error("Failed to run pipeline. Please check logs.")
        st.stop()

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Topic Overview", "Topic Details", "Category Analysis", "Article Explorer", "About"]
)

# Assign topic names
topic_names = {}
for topic in topic_keywords:
    topic_id = topic['topic_id']
    name = ' & '.join(topic['words'][:3])
    topic_names[topic_id] = name

# Function to plot wordcloud
def plot_wordcloud(topic_id):
    topic = next(t for t in topic_keywords if t['topic_id'] == topic_id)
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
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    return fig

# Helper function to convert matplotlib figure to image
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

# Helper function to plot topic keyword barchart
def plot_topic_keywords(topic_id, num_words=15):
    topic = next(t for t in topic_keywords if t['topic_id'] == topic_id)
    words = topic['words'][:num_words]
    probs = topic['probs'][:num_words]
    
    fig = go.Figure(go.Bar(
        x=probs,
        y=words,
        orientation='h',
        marker_color='skyblue'
    ))
    
    fig.update_layout(
        title=f"Top Keywords for Topic {topic_id}: {topic_names[topic_id]}",
        xaxis_title="Probability",
        yaxis_title="Keyword",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# Page 1: Topic Overview
if page == "Topic Overview":
    st.header("Topic Overview")
    
    # Display number of topics and documents
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Topics", len(topic_keywords))
    with col2:
        st.metric("Number of Articles", len(doc_topics_df))
    
    # Topic distribution
    st.subheader("Distribution of Articles by Topic")
    topic_counts = doc_topics_df['dominant_topic'].value_counts().sort_index()
    
    # Create bar chart with Plotly
    fig = px.bar(
        x=[f"Topic {i}: {topic_names[i]}" for i in topic_counts.index],
        y=topic_counts.values,
        labels={'x': 'Topic', 'y': 'Number of Articles'},
        color=topic_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=120)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Category-topic heatmap
    st.subheader("Topic Distribution Across News Categories")
    
    # Create cross-tabulation of categories and topics
    category_topic = pd.crosstab(
        doc_topics_df['category'],
        doc_topics_df['dominant_topic'],
        normalize='index'
    )
    
    # Generate column labels
    col_labels = [f"Topic {i}" for i in category_topic.columns]
    
    # Create heatmap with Plotly
    fig = px.imshow(
        category_topic,
        labels=dict(x="Topic", y="Category", color="Proportion"),
        x=col_labels,
        y=category_topic.index,
        color_continuous_scale="YlGnBu",
        aspect="auto"
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Topic similarity network
    st.subheader("Topic Similarity Network")
    if os.path.exists('topic_similarity_network.html'):
        with open('topic_similarity_network.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=600)
    else:
        st.info("Topic similarity network visualization not found.")

# Page 2: Topic Details
elif page == "Topic Details":
    st.header("Topic Details")
    
    # Topic selector
    topic_options = [f"Topic {i}: {topic_names[i]}" for i in range(len(topic_keywords))]
    selected_topic_str = st.selectbox("Select a topic to explore", topic_options)
    selected_topic_id = int(selected_topic_str.split(':')[0].replace('Topic ', ''))
    
    # Display topic details
    st.subheader(f"Topic {selected_topic_id}: {topic_names[selected_topic_id]}")
    
    # Display wordcloud and keywords side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Word Cloud")
        wordcloud_fig = plot_wordcloud(selected_topic_id)
        st.pyplot(wordcloud_fig)
    
    with col2:
        st.subheader("Top Keywords")
        keywords_fig = plot_topic_keywords(selected_topic_id)
        st.plotly_chart(keywords_fig, use_container_width=True)
    
    # Show articles for this topic
    st.subheader(f"Sample Articles for Topic {selected_topic_id}")
    
    # Filter articles with this dominant topic
    topic_articles = doc_topics_df[doc_topics_df['dominant_topic'] == selected_topic_id]
    
    # Sort by topic probability (descending)
    topic_prob_col = f'topic_{selected_topic_id}_prob'
    if topic_prob_col in topic_articles.columns:
        topic_articles = topic_articles.sort_values(by=topic_prob_col, ascending=False)
    
    # Display top 5 articles
    for i, (_, article) in enumerate(topic_articles.head(5).iterrows()):
        with st.expander(f"{i+1}. {article['headlines']} ({article['category']})"):
            st.markdown(f"**Category:** {article['category']}")
            
            if 'description' in article and not pd.isna(article['description']):
                st.markdown(f"**Description:** {article['description']}")
            
            # Show beginning of content
            content = article['content']
            if len(content) > 300:
                st.markdown(f"**Content:** {content[:300]}...")
            else:
                st.markdown(f"**Content:** {content}")
            
            # Show URL if available
            if 'url' in article and not pd.isna(article['url']):
                st.markdown(f"[Read full article]({article['url']})")

# Page 3: Category Analysis
elif page == "Category Analysis":
    st.header("Category Analysis")
    
    # Category selector
    categories = sorted(doc_topics_df['category'].unique())
    selected_category = st.selectbox("Select a news category", categories)
    
    # Filter data for selected category
    category_data = doc_topics_df[doc_topics_df['category'] == selected_category]
    
    # Display category stats
    st.subheader(f"Statistics for {selected_category} Category")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Articles", len(category_data))
    
    # Most common topics in this category
    topic_counts = category_data['dominant_topic'].value_counts().sort_values(ascending=False)
    with col2:
        most_common_topic = topic_counts.index[0]
        st.metric("Most Common Topic", f"Topic {most_common_topic}: {topic_names[most_common_topic]}")
    
    # Topic distribution for this category
    st.subheader(f"Topic Distribution in {selected_category} Category")
    
    # Create bar chart with Plotly
    fig = px.bar(
        x=[f"Topic {i}: {topic_names[i]}" for i in topic_counts.index],
        y=topic_counts.values,
        labels={'x': 'Topic', 'y': 'Number of Articles'},
        color=topic_counts.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=500,
        xaxis_tickangle=-45,
        margin=dict(l=20, r=20, t=40, b=120)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display top articles for most common topic in this category
    st.subheader(f"Sample Articles from Most Common Topic")
    
    # Filter articles with most common topic in this category
    topic_articles = category_data[category_data['dominant_topic'] == most_common_topic]
    
    # Sort by topic probability (descending)
    topic_prob_col = f'topic_{most_common_topic}_prob'
    if topic_prob_col in topic_articles.columns:
        topic_articles = topic_articles.sort_values(by=topic_prob_col, ascending=False)
    
    # Display top 3 articles
    for i, (_, article) in enumerate(topic_articles.head(3).iterrows()):
        with st.expander(f"{i+1}. {article['headlines']}"):
            if 'description' in article and not pd.isna(article['description']):
                st.markdown(f"**Description:** {article['description']}")
            
            # Show beginning of content
            content = article['content']
            if len(content) > 300:
                st.markdown(f"**Content:** {content[:300]}...")
            else:
                st.markdown(f"**Content:** {content}")
            
            # Show URL if available
            if 'url' in article and not pd.isna(article['url']):
                st.markdown(f"[Read full article]({article['url']})")

# Page 4: Article Explorer
elif page == "Article Explorer":
    st.header("Article Explorer")
    
    # Search functionality
    search_query = st.text_input("Search articles by keyword or phrase")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        category_filter = st.multiselect(
            "Filter by category",
            options=sorted(doc_topics_df['category'].unique()),
            default=[]
        )
    
    with col2:
        topic_filter = st.multiselect(
            "Filter by topic",
            options=[f"Topic {i}: {topic_names[i]}" for i in range(len(topic_keywords))],
            default=[]
        )
    
    # Apply filters
    filtered_data = doc_topics_df.copy()
    
    # Apply category filter
    if category_filter:
        filtered_data = filtered_data[filtered_data['category'].isin(category_filter)]
    
    # Apply topic filter
    if topic_filter:
        topic_ids = [int(topic.split(':')[0].replace('Topic ', '')) for topic in topic_filter]
        filtered_data = filtered_data[filtered_data['dominant_topic'].isin(topic_ids)]
    
    # Apply search query
    if search_query:
        search_terms = search_query.lower().split()
        mask = filtered_data['headlines'].str.lower().apply(
            lambda x: all(term in x for term in search_terms)
        )
        filtered_data = filtered_data[mask]
    
    # Display results
    st.subheader(f"Found {len(filtered_data)} articles")
    
    # Sort options
    sort_options = {
        "Newest first": False,  # Assuming we don't have date info
        "Oldest first": False,  # Assuming we don't have date info
        "Alphabetical (A-Z)": True
    }
    
    sort_by = st.radio("Sort by", options=list(sort_options.keys()), horizontal=True)
    
    # Apply sorting
    if sort_by == "Alphabetical (A-Z)":
        filtered_data = filtered_data.sort_values(by='headlines')
    
    # Display articles
    articles_per_page = 10
    num_pages = (len(filtered_data) - 1) // articles_per_page + 1
    
    if num_pages > 0:
        page_num = st.slider("Page", 1, max(1, num_pages), 1)
        start_idx = (page_num - 1) * articles_per_page
        end_idx = min(start_idx + articles_per_page, len(filtered_data))
        
        page_data = filtered_data.iloc[start_idx:end_idx]
        
        for i, (_, article) in enumerate(page_data.iterrows()):
            topic_id = article['dominant_topic']
            with st.expander(f"{start_idx + i + 1}. {article['headlines']} ({article['category']}, Topic {topic_id})"):
                st.markdown(f"**Category:** {article['category']}")
                st.markdown(f"**Topic:** Topic {topic_id}: {topic_names[topic_id]}")
                
                if 'description' in article and not pd.isna(article['description']):
                    st.markdown(f"**Description:** {article['description']}")
                
                # Show beginning of content
                content = article['content']
                if len(content) > 300:
                    st.markdown(f"**Content:** {content[:300]}...")
                else:
                    st.markdown(f"**Content:** {content}")
                
                # Show topic distribution for this article
                topic_probs = {}
                for j in range(len(topic_keywords)):
                    prob_col = f'topic_{j}_prob'
                    if prob_col in article:
                        topic_probs[f"Topic {j}"] = article[prob_col]
                
                if topic_probs:
                    st.subheader("Topic Distribution")
                    fig = px.bar(
                        x=list(topic_probs.keys()),
                        y=list(topic_probs.values()),
                        labels={'x': 'Topic', 'y': 'Probability'},
                        color=list(topic_probs.values()),
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show URL if available
                if 'url' in article and not pd.isna(article['url']):
                    st.markdown(f"[Read full article]({article['url']})")
    else:
        st.info("No articles found matching your filters.")

# Page 5: About
elif page == "About":
    st.header("About")
    
    st.markdown("""
    ## Topic Modeling on The Indian Express News Articles
    
    This application uses **Latent Dirichlet Allocation (LDA)** to discover hidden topics in news articles 
    from The Indian Express across various categories (Sports, Business, Education, Technology, and Entertainment).
    
    ### What is Topic Modeling?
    
    Topic modeling is an unsupervised machine learning technique used to identify hidden semantic structures in text data. 
    LDA works by:
    
    - Assuming each document is a mixture of topics
    - Each topic is a mixture of words
    - Words are generated by sampling topics according to document-topic distribution
    - Then sampling words according to topic-word distribution
    
    ### The Dataset
    
    The dataset contains news articles from The Indian Express, organized into 5 categories:
    - Sports
    - Business
    - Education
    - Technology
    - Entertainment
    
    Each article includes:
    - Headline
    - Description
    - Content
    - URL
    - Category
    
    ### How to Use This App
    
    - **Topic Overview**: Get a bird's-eye view of all discovered topics
    - **Topic Details**: Explore specific topics and their keywords
    - **Category Analysis**: Analyze topic distribution within news categories
    - **Article Explorer**: Search and browse articles by various filters
    
    ### Implementation Details
    
    The topic modeling pipeline includes:
    
    1. **Data Preprocessing**: Cleaning text, removing stopwords, lemmatization
    2. **Feature Extraction**: Converting text to TF-IDF vectors
    3. **LDA Model Training**: Discovering latent topics
    4. **Visualization**: Generating interactive visualizations
    
    This project was implemented using Python with libraries including:
    - scikit-learn
    - gensim
    - NLTK
    - pandas
    - matplotlib
    - wordcloud
    - Streamlit
    """)
    
    # Project credits
    st.subheader("Project Credits")
    st.markdown("""
    This project was created as part of the Topic Modeling on The Indian Express News Article assignment.
    
    - **Data Source**: The Indian Express
    - **Libraries Used**: scikit-learn, gensim, NLTK, pandas, matplotlib, wordcloud, plotly, Streamlit
    """)

# Footer
st.markdown("---")
st.markdown("© 2024 | Topic Modeling on The Indian Express News Articles")
