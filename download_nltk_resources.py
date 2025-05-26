"""
Download all required NLTK resources for the Word Sense Disambiguation project.
"""

import nltk
import sys

def download_resources():
    """Download all required NLTK resources."""
    resources = [
        'punkt',
        'wordnet',
        'semcor',
        'omw-1.4'  # Open Multilingual WordNet
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)
    
    print("All resources downloaded successfully!")

if __name__ == "__main__":
    download_resources()
