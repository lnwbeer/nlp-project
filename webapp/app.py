"""
Streamlit web application for Word Sense Disambiguation using GlossBERT.
"""

import os
import sys
import streamlit as st
import nltk
# Use a simpler tokenization approach to avoid NLTK issues
def simple_tokenize(text):
    """Simple tokenization function that splits on whitespace and punctuation."""
    # Remove punctuation or replace with space
    for punct in '.,;:!?()[]{}-\"\\"':
        text = text.replace(punct, ' ')
    # Split on whitespace and filter out empty strings
    return [token for token in text.split() if token]
import pandas as pd
import time
import torch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from glossbert.models.infer import WSDPredictor
from glossbert.utils.wordnet_utils import download_nltk_resources

# Download NLTK resources
@st.cache_resource
def load_nltk_resources():
    download_nltk_resources()
    nltk.download('punkt')
    # Make sure we have all required NLTK resources
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Load WSD predictor
@st.cache_resource
def load_predictor(model_path):
    return WSDPredictor(model_path)

def main():
    # Page configuration
    st.set_page_config(
        page_title="Word Sense Disambiguation",
        page_icon="🔤",
        layout="wide"
    )
    
    # Load resources
    load_nltk_resources()
    
    # Header
    st.title("Word Sense Disambiguation with GlossBERT")
    st.markdown("""
    This application uses a fine-tuned BERT model (GlossBERT) to disambiguate word senses based on context.
    Enter a sentence containing an ambiguous word, select the word, and the model will predict the most likely sense from WordNet.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    **Word Sense Disambiguation (WSD)** is the task of identifying which sense of a word is used in a sentence when the word has multiple meanings.
    
    This application uses **GlossBERT**, which fine-tunes BERT to match context sentences with the appropriate WordNet gloss definitions.
    
    The model was trained on the SemCor dataset, a corpus with words manually tagged with their WordNet senses.
    """)
    
    # Model path selection
    model_path = st.sidebar.selectbox(
        "Select model",
        ["../models/glossbert/best_model", "../models/glossbert/final_model"],
        index=0
    )
    
    # Load predictor
    try:
        predictor = load_predictor(model_path)
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you have trained the model first by running `python glossbert/models/train.py`")
        model_loaded = False
    
    # Input form
    with st.form("wsd_form"):
        # Example sentences
        examples = {
            "Bank example": "I went to the bank to deposit some money.",
            "Bass example": "The bass player performed an amazing solo.",
            "Spring example": "After a long winter, spring finally arrived.",
            "Custom": ""
        }
        
        example_choice = st.selectbox("Choose an example or enter your own:", list(examples.keys()))
        
        if example_choice == "Custom":
            sentence = st.text_area("Enter a sentence containing an ambiguous word:", height=100)
        else:
            sentence = st.text_area("Enter a sentence containing an ambiguous word:", 
                                   value=examples[example_choice], height=100)
        
        # Tokenize sentence for word selection
        if sentence:
            words = simple_tokenize(sentence)
            target_word = st.selectbox("Select the ambiguous word:", words)
        else:
            target_word = ""
        
        # Submit button
        submit_button = st.form_submit_button("Disambiguate")
    
    # Process input and display results
    if submit_button and model_loaded and sentence and target_word:
        with st.spinner("Disambiguating..."):
            # Get predictions
            predictions = predictor.predict(sentence, target_word)
            
            if not predictions:
                st.warning(f"No WordNet senses found for '{target_word}'")
            else:
                # Display results
                st.subheader(f"Senses for '{target_word}' in context")
                
                # Create a dataframe for the results
                results_data = []
                for i, pred in enumerate(predictions):
                    results_data.append({
                        "Rank": i + 1,
                        "Synset": pred['synset_id'],
                        "Definition": pred['gloss'],
                        "Score": f"{pred['score']:.4f}",
                        "Examples": ", ".join(pred['examples']) if pred['examples'] else "N/A",
                        "Lemmas": ", ".join(pred['lemmas'])
                    })
                
                results_df = pd.DataFrame(results_data)
                
                # Display top result
                top_result = results_data[0]
                st.success(f"**Most likely sense:** {top_result['Synset']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Definition:** {top_result['Definition']}")
                    st.markdown(f"**Score:** {top_result['Score']}")
                
                with col2:
                    st.markdown(f"**Lemmas:** {top_result['Lemmas']}")
                    if top_result['Examples'] != "N/A":
                        st.markdown(f"**Example usage:** {top_result['Examples']}")
                
                # Display all results in a table
                st.subheader("All possible senses (ranked by probability)")
                st.dataframe(results_df[["Rank", "Synset", "Definition", "Score"]])
                
                # Visualize scores
                st.subheader("Sense Probability Distribution")
                chart_data = pd.DataFrame({
                    "Synset": [pred['synset_id'] for pred in predictions],
                    "Score": [pred['score'] for pred in predictions]
                })
                st.bar_chart(chart_data.set_index("Synset"))

if __name__ == "__main__":
    main()
