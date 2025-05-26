# GlossBERT Word Sense Disambiguation

This project implements a Word Sense Disambiguation (WSD) system using a supervised approach based on GlossBERT. The model is fine-tuned to predict the correct sense of an ambiguous word in context by matching it with the appropriate WordNet gloss definition.

## Project Structure

```
.
├── glossbert/
│   ├── data/        # Data processing and dataset creation
│   ├── models/      # Model definition and training code
│   └── utils/       # Utility functions for WordNet and evaluation
├── webapp/          # Streamlit web application
├── requirements.txt # Project dependencies
├── init_project.py  # Initialization script
├── download_nltk_resources.py # Script to download NLTK resources
└── README.md        # Project documentation
```

## Overview

Word Sense Disambiguation is the task of identifying which sense of a word is used in a sentence when the word has multiple meanings. This project uses a fine-tuned BERT model to match context sentences with the appropriate WordNet gloss definitions.

## Features

- Fine-tuning of pretrained BERT model on WordNet glosses
- Supervised approach using SemCor dataset or sample data
- Command-line interface for word sense prediction
- Interactive web application built with Streamlit
- Comprehensive evaluation metrics

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/lnwbeer/nlp-project.git
   cd nlp-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download NLTK resources:
   ```
   python download_nltk_resources.py
   ```

4. Initialize the project:
   ```
   python init_project.py
   ```

## Usage

### Training the Model

```
python glossbert/models/train.py
```

### Running Inference

```
python glossbert/models/infer.py --sentence "I went to the bank to deposit money" --target_word "bank"
```

### Web Application

```
streamlit run webapp/app.py
```

Open your browser and navigate to http://localhost:8501

## Model Architecture

The GlossBERT model treats WSD as a sentence-pair classification task:
1. Each context sentence is paired with potential word sense definitions (glosses)
2. BERT processes these pairs and predicts whether the gloss matches the context
3. The sense with the highest matching score is selected as the correct meaning

## Examples

The system can disambiguate words like:
- "bank" (financial institution vs. sloping land)
- "bass" (musical instrument vs. type of fish)
- "spring" (season vs. elastic device)

## License

MIT

## Acknowledgements

- This project is inspired by the GlossBERT paper: [GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge](https://arxiv.org/abs/1908.07245)
- Uses WordNet from the NLTK library
- Built with PyTorch and Transformers library

## Approach

The project uses GlossBERT, which formulates WSD as a sentence-pair classification task:
1. For each ambiguous word, retrieve all possible senses (glosses) from WordNet
2. Create pairs of (context sentence, gloss) with binary labels (1 for correct sense, 0 for others)
3. Fine-tune BERT to classify whether a gloss matches the context
4. During inference, select the gloss with the highest score

## References

- GlossBERT: BERT for Word Sense Disambiguation with Gloss Knowledge (Huang et al., 2019)
- WordNet: A Lexical Database for English (Miller, 1995)
