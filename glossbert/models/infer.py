"""
Inference script for GlossBERT Word Sense Disambiguation model.
"""

import os
import sys
import argparse
import logging
import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import BertTokenizer

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import GlossBERT
from utils.wordnet_utils import (
    download_nltk_resources,
    get_glosses_for_word,
    get_examples_for_synset,
    get_lemmas_for_synset
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class WSDPredictor:
    """Word Sense Disambiguation predictor using GlossBERT."""
    
    def __init__(self, model_path: str, max_seq_length: int = 128):
        """
        Initialize the WSD predictor.
        
        Args:
            model_path: Path to the fine-tuned GlossBERT model
            max_seq_length: Maximum sequence length for BERT
        """
        self.max_seq_length = max_seq_length
        
        # Download NLTK resources
        download_nltk_resources()
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = GlossBERT.from_pretrained(model_path)
        
        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded and ready for inference on {self.device}")
    
    def predict(self, sentence: str, target_word: str) -> List[Dict]:
        """
        Predict the most likely sense for a target word in a sentence.
        
        Args:
            sentence: The context sentence
            target_word: The ambiguous target word
            
        Returns:
            List of dictionaries containing sense predictions with scores
        """
        # Get all possible glosses for the target word
        glosses = get_glosses_for_word(target_word)
        
        if not glosses:
            logger.warning(f"No WordNet senses found for '{target_word}'")
            return []
        
        # Prepare input pairs
        pairs = []
        for synset_id, gloss in glosses:
            pairs.append((sentence, gloss, synset_id))
        
        # Get predictions
        predictions = []
        
        for sentence, gloss, synset_id in pairs:
            # Tokenize the sentence-gloss pair
            encoding = self.tokenizer.encode_plus(
                sentence,
                gloss,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
            
            # Move tensors to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            token_type_ids = encoding['token_type_ids'].to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get the probability of the positive class (correct sense)
            positive_prob = probs[1]
            
            # Get examples and lemmas for the synset
            examples = get_examples_for_synset(synset_id)
            lemmas = get_lemmas_for_synset(synset_id)
            
            predictions.append({
                'synset_id': synset_id,
                'gloss': gloss,
                'score': float(positive_prob),
                'examples': examples,
                'lemmas': lemmas
            })
        
        # Sort predictions by score in descending order
        predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        return predictions

def main():
    """Main function for WSD inference."""
    parser = argparse.ArgumentParser(description="Word Sense Disambiguation Inference")
    
    parser.add_argument("--model_path", type=str, default="../models/glossbert/best_model",
                        help="Path to the fine-tuned GlossBERT model")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length for BERT")
    parser.add_argument("--sentence", type=str, required=True,
                        help="Input sentence containing the ambiguous word")
    parser.add_argument("--target_word", type=str, required=True,
                        help="The ambiguous target word")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = WSDPredictor(args.model_path, args.max_seq_length)
    
    # Get predictions
    predictions = predictor.predict(args.sentence, args.target_word)
    
    if not predictions:
        print(f"No senses found for '{args.target_word}'")
        return
    
    # Print results
    print(f"\nWord Sense Disambiguation for '{args.target_word}' in sentence:")
    print(f"  \"{args.sentence}\"")
    print("\nPredicted senses (ranked by probability):")
    
    for i, pred in enumerate(predictions):
        print(f"\n{i+1}. {pred['synset_id']} (Score: {pred['score']:.4f})")
        print(f"   Definition: {pred['gloss']}")
        
        if pred['lemmas']:
            print(f"   Lemmas: {', '.join(pred['lemmas'])}")
        
        if pred['examples']:
            print(f"   Examples:")
            for example in pred['examples']:
                print(f"     - {example}")

if __name__ == "__main__":
    main()
