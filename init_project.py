"""
Initialization script for the GlossBERT Word Sense Disambiguation project.
This script sets up the necessary NLTK resources and creates the synset-gloss mapping.
"""

import os
import sys
import argparse
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def main():
    """Initialize the GlossBERT WSD project."""
    parser = argparse.ArgumentParser(description="Initialize GlossBERT WSD project")
    
    parser.add_argument("--data_dir", type=str, default="glossbert/data",
                        help="Directory to store data files")
    
    args = parser.parse_args()
    
    # Create directories
    logger.info("Creating project directories...")
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "processed"), exist_ok=True)
    os.makedirs("glossbert/models/glossbert", exist_ok=True)
    
    # Download NLTK resources
    logger.info("Downloading NLTK resources...")
    from glossbert.utils.wordnet_utils import download_nltk_resources
    download_nltk_resources()
    
    # Create synset-gloss mapping
    logger.info("Creating synset-gloss mapping...")
    from glossbert.utils.wordnet_utils import save_synset_gloss_mapping
    save_synset_gloss_mapping(os.path.join(args.data_dir, "synset_gloss_map.json"))
    
    logger.info("Project initialization complete!")
    logger.info("\nNext steps:")
    logger.info("1. Train the model: python glossbert/models/train.py")
    logger.info("2. Run inference: python glossbert/models/infer.py --sentence \"I went to the bank\" --target_word \"bank\"")
    logger.info("3. Start the web app: streamlit run webapp/app.py")

if __name__ == "__main__":
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    main()
