"""
Dataset preparation for GlossBERT Word Sense Disambiguation.
Handles loading and preprocessing SemCor data to create (context, gloss) pairs.
"""

import os
import json
import random
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.corpus import semcor
from nltk.corpus import wordnet as wn
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import sys
import logging
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.wordnet_utils import download_nltk_resources, get_glosses_for_word

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class SemCorProcessor:
    """Process SemCor corpus to create WSD training data."""
    
    def __init__(self, max_seq_length: int = 128, negative_samples: int = 3):
        """
        Initialize the SemCor processor.
        
        Args:
            max_seq_length: Maximum sequence length for BERT
            negative_samples: Number of negative samples per positive example
        """
        self.max_seq_length = max_seq_length
        self.negative_samples = negative_samples
        download_nltk_resources()
        
        # Check if SemCor is available
        try:
            semcor.sents()
        except LookupError:
            nltk.download('semcor')
    
    def _create_sample_dataset(self) -> List[Dict]:
        """
        Create a small sample dataset for WSD when SemCor data is not available.
        
        Returns:
            List of dictionaries containing sample sentences and tagged words
        """
        # Common ambiguous words with their synsets
        sample_data = [
            {
                'sentence': 'I went to the bank to deposit money.',
                'word': 'bank',
                'synset_id': 'bank.n.01',  # Financial institution
                'pos': 'n'
            },
            {
                'sentence': 'The river bank was covered with wildflowers.',
                'word': 'bank',
                'synset_id': 'bank.n.02',  # Sloping land beside a body of water
                'pos': 'n'
            },
            {
                'sentence': 'The bass player performed an amazing solo.',
                'word': 'bass',
                'synset_id': 'bass.n.01',  # Musical instrument
                'pos': 'n'
            },
            {
                'sentence': 'We caught several bass while fishing yesterday.',
                'word': 'bass',
                'synset_id': 'bass.n.02',  # Type of fish
                'pos': 'n'
            },
            {
                'sentence': 'After a long winter, spring finally arrived.',
                'word': 'spring',
                'synset_id': 'spring.n.01',  # Season
                'pos': 'n'
            },
            {
                'sentence': 'The spring in my watch broke and needs to be replaced.',
                'word': 'spring',
                'synset_id': 'spring.n.02',  # Elastic device
                'pos': 'n'
            },
            {
                'sentence': 'She can run very fast.',
                'word': 'run',
                'synset_id': 'run.v.01',  # Move at a fast pace
                'pos': 'v'
            },
            {
                'sentence': 'They will run the company together.',
                'word': 'run',
                'synset_id': 'run.v.04',  # Manage or operate
                'pos': 'v'
            },
            {
                'sentence': 'The light was too bright for my eyes.',
                'word': 'bright',
                'synset_id': 'bright.a.01',  # Emitting or reflecting light
                'pos': 'a'
            },
            {
                'sentence': 'She is a bright student who learns quickly.',
                'word': 'bright',
                'synset_id': 'bright.a.02',  # Intelligent
                'pos': 'a'
            }
        ]
        
        tagged_sentences = []
        
        for item in sample_data:
            # Get the actual synset object
            try:
                synset = wn.synset(item['synset_id'])
                
                tagged_words = [{
                    'word': item['word'],
                    'synset_id': synset.name(),
                    'gloss': synset.definition(),
                    'pos': synset.pos()
                }]
                
                tagged_sentences.append({
                    'sentence': item['sentence'],
                    'tagged_words': tagged_words
                })
            except:
                # Skip if synset not found
                continue
        
        return tagged_sentences
    
    def extract_tagged_sentences(self) -> List[Dict]:
        """
        Extract sentences with sense-tagged words from SemCor.
        
        Returns:
            List of dictionaries containing sentences and tagged words
        """
        logger.info("Extracting tagged sentences from SemCor...")
        
        tagged_sentences = []
        
        try:
            # Process each file in SemCor
            for fileid in tqdm(semcor.fileids()):
                for sent in semcor.tagged_sents(fileid, tag='sem'):
                    tagged_words = []
                    
                    # Extract words with WordNet tags
                    for word in sent:
                        if hasattr(word, 'label') and word.label():
                            # Get the synset from the tagged word
                            synset = word.label()
                            
                            # Handle different types of labels in SemCor
                            if isinstance(synset, nltk.corpus.reader.wordnet.Synset):
                                synset_id = synset.name()
                                gloss = synset.definition()
                                pos = synset.pos()
                            elif isinstance(synset, str):
                                # Try to convert string label to synset
                                try:
                                    synset_obj = wn.synset(synset)
                                    synset_id = synset
                                    gloss = synset_obj.definition()
                                    pos = synset_obj.pos()
                                except:
                                    # If conversion fails, skip this word
                                    continue
                            else:
                                # Skip if synset type is not recognized
                                continue
                            
                            tagged_words.append({
                                'word': word[0],
                                'synset_id': synset_id,
                                'gloss': gloss,
                                'pos': pos
                            })
                    
                    if tagged_words:
                        # Reconstruct the sentence
                        sentence = ' '.join(word[0] if isinstance(word, tuple) else word for word in sent)
                        
                        tagged_sentences.append({
                            'sentence': sentence,
                            'tagged_words': tagged_words
                        })
            
            logger.info(f"Extracted {len(tagged_sentences)} tagged sentences")
            
            # If no sentences were extracted, create a small sample dataset
            if len(tagged_sentences) == 0:
                logger.warning("No tagged sentences found in SemCor. Creating sample dataset...")
                sample_sentences = self._create_sample_dataset()
                tagged_sentences.extend(sample_sentences)
                logger.info(f"Created {len(sample_sentences)} sample sentences")
                
            return tagged_sentences
            
        except Exception as e:
            logger.error(f"Error extracting tagged sentences: {str(e)}")
            # Create a small sample dataset as fallback
            sample_sentences = self._create_sample_dataset()
            logger.info(f"Created {len(sample_sentences)} sample sentences")
            return sample_sentences
    
    def create_context_gloss_pairs(self, tagged_sentences: List[Dict]) -> List[Dict]:
        """
        Create (context, gloss) pairs with binary labels.
        
        Args:
            tagged_sentences: List of sentences with tagged words
            
        Returns:
            List of (context, gloss) pairs with labels
        """
        logger.info("Creating context-gloss pairs...")
        
        context_gloss_pairs = []
        
        for item in tqdm(tagged_sentences):
            sentence = item['sentence']
            
            for tagged_word in item['tagged_words']:
                word = tagged_word['word']
                correct_synset_id = tagged_word['synset_id']
                correct_gloss = tagged_word['gloss']
                pos = tagged_word['pos']
                
                # Create positive example (correct sense)
                context_gloss_pairs.append({
                    'sentence': sentence,
                    'target_word': word,
                    'gloss': correct_gloss,
                    'synset_id': correct_synset_id,
                    'label': 1  # Positive example
                })
                
                # Get all possible glosses for the word
                all_glosses = get_glosses_for_word(word, pos)
                
                # Filter out the correct gloss
                negative_glosses = [(synset_id, gloss) for synset_id, gloss in all_glosses 
                                   if synset_id != correct_synset_id]
                
                # If there are no negative glosses, skip
                if not negative_glosses:
                    continue
                
                # Sample negative examples
                num_samples = min(self.negative_samples, len(negative_glosses))
                sampled_negative_glosses = random.sample(negative_glosses, num_samples)
                
                # Create negative examples
                for synset_id, gloss in sampled_negative_glosses:
                    context_gloss_pairs.append({
                        'sentence': sentence,
                        'target_word': word,
                        'gloss': gloss,
                        'synset_id': synset_id,
                        'label': 0  # Negative example
                    })
        
        logger.info(f"Created {len(context_gloss_pairs)} context-gloss pairs")
        return context_gloss_pairs
    
    def save_to_csv(self, context_gloss_pairs: List[Dict], output_path: str) -> None:
        """
        Save context-gloss pairs to a CSV file.
        
        Args:
            context_gloss_pairs: List of context-gloss pairs
            output_path: Path to save the CSV file
        """
        df = pd.DataFrame(context_gloss_pairs)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} context-gloss pairs to {output_path}")
    
    def process_and_save(self, output_path: str) -> None:
        """
        Process SemCor and save context-gloss pairs.
        
        Args:
            output_path: Path to save the CSV file
        """
        tagged_sentences = self.extract_tagged_sentences()
        context_gloss_pairs = self.create_context_gloss_pairs(tagged_sentences)
        self.save_to_csv(context_gloss_pairs, output_path)

class GlossBERTDataset(Dataset):
    """Dataset for GlossBERT fine-tuning."""
    
    def __init__(self, data_path: str, tokenizer: BertTokenizer, max_seq_length: int = 128):
        """
        Initialize the GlossBERT dataset.
        
        Args:
            data_path: Path to the CSV file with context-gloss pairs
            tokenizer: BERT tokenizer
            max_seq_length: Maximum sequence length for BERT
        """
        self.data = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a tokenized example.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids, and labels
        """
        row = self.data.iloc[idx]
        
        # Tokenize the sentence-gloss pair
        encoding = self.tokenizer.encode_plus(
            row['sentence'],
            row['gloss'],
            add_special_tokens=True,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        # Convert label to tensor
        label = torch.tensor(row['label'], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'labels': label
        }

def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: BertTokenizer,
    max_seq_length: int = 128,
    batch_size: int = 16
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        train_path: Path to the training CSV file
        val_path: Path to the validation CSV file
        tokenizer: BERT tokenizer
        max_seq_length: Maximum sequence length for BERT
        batch_size: Batch size for training
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Create datasets
    train_dataset = GlossBERTDataset(train_path, tokenizer, max_seq_length)
    val_dataset = GlossBERTDataset(val_path, tokenizer, max_seq_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader

def prepare_semcor_data(output_dir: str, val_split: float = 0.1) -> Tuple[str, str]:
    """
    Prepare SemCor data for GlossBERT training.
    
    Args:
        output_dir: Directory to save the processed data
        val_split: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_path, val_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    all_data_path = os.path.join(output_dir, 'semcor_glossbert_all.csv')
    train_path = os.path.join(output_dir, 'semcor_glossbert_train.csv')
    val_path = os.path.join(output_dir, 'semcor_glossbert_val.csv')
    
    # Check if processed data already exists
    if os.path.exists(train_path) and os.path.exists(val_path):
        logger.info(f"Processed data already exists at {train_path} and {val_path}")
        return train_path, val_path
    
    # Process SemCor data
    processor = SemCorProcessor()
    processor.process_and_save(all_data_path)
    
    # Split into train and validation sets
    df = pd.read_csv(all_data_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    split_idx = int(len(df) * (1 - val_split))
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    logger.info(f"Split data into {len(train_df)} training and {len(val_df)} validation examples")
    
    return train_path, val_path

if __name__ == "__main__":
    # Example usage
    output_dir = "processed_data"
    train_path, val_path = prepare_semcor_data(output_dir)
    
    # Load a tokenizer to test the dataset
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Create a small dataset to test
    dataset = GlossBERTDataset(train_path, tokenizer)
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print("Sample input_ids shape:", sample['input_ids'].shape)
    print("Sample attention_mask shape:", sample['attention_mask'].shape)
    print("Sample token_type_ids shape:", sample['token_type_ids'].shape)
    print("Sample label:", sample['labels'])
