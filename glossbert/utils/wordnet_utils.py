"""
Utility functions for working with WordNet and handling glosses for Word Sense Disambiguation.
"""

import json
import os
import nltk
from nltk.corpus import wordnet as wn
from typing import Dict, List, Tuple, Optional

# Download required NLTK resources if not already available
def download_nltk_resources():
    """Download required NLTK resources if not already available."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    
    try:
        nltk.data.find('corpora/semcor')
    except LookupError:
        nltk.download('semcor')

def get_wordnet_pos(treebank_tag: str) -> Optional[str]:
    """
    Convert Penn Treebank POS tags to WordNet POS tags.
    
    Args:
        treebank_tag: Penn Treebank POS tag
        
    Returns:
        WordNet POS tag or None if no match
    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def get_synsets_for_word(word: str, pos: Optional[str] = None) -> List[wn.synset]:
    """
    Get all possible WordNet synsets for a given word.
    
    Args:
        word: The target word
        pos: Part of speech (optional)
        
    Returns:
        List of synsets
    """
    if pos:
        return wn.synsets(word, pos=pos)
    else:
        return wn.synsets(word)

def get_glosses_for_word(word: str, pos: Optional[str] = None) -> List[Tuple[str, str]]:
    """
    Get all possible glosses for a given word.
    
    Args:
        word: The target word
        pos: Part of speech (optional)
        
    Returns:
        List of tuples (synset_id, gloss)
    """
    synsets = get_synsets_for_word(word, pos)
    return [(synset.name(), synset.definition()) for synset in synsets]

def create_synset_gloss_mapping() -> Dict[str, str]:
    """
    Create a mapping of all WordNet synset IDs to their glosses.
    
    Returns:
        Dictionary mapping synset IDs to glosses
    """
    synset_gloss_map = {}
    
    for synset in list(wn.all_synsets()):
        synset_id = synset.name()
        gloss = synset.definition()
        synset_gloss_map[synset_id] = gloss
    
    return synset_gloss_map

def save_synset_gloss_mapping(output_path: str) -> None:
    """
    Save the synset-to-gloss mapping as a JSON file.
    
    Args:
        output_path: Path to save the JSON file
    """
    synset_gloss_map = create_synset_gloss_mapping()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(synset_gloss_map, f, indent=2)
    
    print(f"Saved synset-gloss mapping to {output_path}")

def load_synset_gloss_mapping(input_path: str) -> Dict[str, str]:
    """
    Load the synset-to-gloss mapping from a JSON file.
    
    Args:
        input_path: Path to the JSON file
        
    Returns:
        Dictionary mapping synset IDs to glosses
    """
    with open(input_path, 'r') as f:
        synset_gloss_map = json.load(f)
    
    return synset_gloss_map

def get_examples_for_synset(synset_id: str) -> List[str]:
    """
    Get usage examples for a given synset.
    
    Args:
        synset_id: The synset ID
        
    Returns:
        List of example sentences
    """
    synset = wn.synset(synset_id)
    examples = synset.examples()
    return examples

def get_lemmas_for_synset(synset_id: str) -> List[str]:
    """
    Get all lemmas (word forms) for a given synset.
    
    Args:
        synset_id: The synset ID
        
    Returns:
        List of lemmas
    """
    synset = wn.synset(synset_id)
    lemmas = [lemma.name().replace('_', ' ') for lemma in synset.lemmas()]
    return lemmas

if __name__ == "__main__":
    # Example usage
    download_nltk_resources()
    
    # Get glosses for an ambiguous word
    word = "bank"
    glosses = get_glosses_for_word(word)
    print(f"Glosses for '{word}':")
    for synset_id, gloss in glosses:
        print(f"  - {synset_id}: {gloss}")
    
    # Create and save synset-gloss mapping
    output_path = "../data/synset_gloss_map.json"
    save_synset_gloss_mapping(output_path)
