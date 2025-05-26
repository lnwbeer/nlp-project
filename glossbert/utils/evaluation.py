"""
Evaluation utilities for Word Sense Disambiguation.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.infer import WSDPredictor

def evaluate_on_test_set(predictor: WSDPredictor, test_data_path: str) -> Dict:
    """
    Evaluate the WSD model on a test dataset.
    
    Args:
        predictor: WSD predictor
        test_data_path: Path to the test data CSV
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load test data
    test_df = pd.read_csv(test_data_path)
    
    # Filter to only positive examples (correct senses)
    test_df = test_df[test_df['label'] == 1]
    
    correct = 0
    total = 0
    
    # Store predictions for analysis
    predictions = []
    
    # Evaluate each example
    for _, row in test_df.iterrows():
        sentence = row['sentence']
        target_word = row['target_word']
        correct_synset_id = row['synset_id']
        
        # Get model predictions
        preds = predictor.predict(sentence, target_word)
        
        if not preds:
            continue
        
        # Get top prediction
        top_pred = preds[0]['synset_id']
        
        # Check if prediction is correct
        is_correct = top_pred == correct_synset_id
        
        if is_correct:
            correct += 1
        
        total += 1
        
        # Store prediction data
        predictions.append({
            'sentence': sentence,
            'target_word': target_word,
            'correct_synset_id': correct_synset_id,
            'predicted_synset_id': top_pred,
            'correct': is_correct,
            'score': preds[0]['score']
        })
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Create predictions DataFrame
    preds_df = pd.DataFrame(predictions)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy,
        'total_examples': total,
        'correct_predictions': correct
    }
    
    return metrics, preds_df

def plot_confusion_matrix(preds_df: pd.DataFrame, output_path: str) -> None:
    """
    Plot confusion matrix for WSD predictions.
    
    Args:
        preds_df: DataFrame of predictions
        output_path: Path to save the plot
    """
    # Create a confusion matrix of correct vs. incorrect predictions
    cm = confusion_matrix(preds_df['correct'], preds_df['correct'], labels=[True, False])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Correct', 'Incorrect'],
                yticklabels=['Correct', 'Incorrect'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_score_distribution(preds_df: pd.DataFrame, output_path: str) -> None:
    """
    Plot distribution of prediction scores.
    
    Args:
        preds_df: DataFrame of predictions
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Plot score distributions for correct and incorrect predictions
    sns.histplot(data=preds_df, x='score', hue='correct', bins=20, 
                 element='step', common_norm=False, stat='density')
    
    plt.title('Distribution of Prediction Scores')
    plt.xlabel('Score')
    plt.ylabel('Density')
    plt.legend(['Incorrect', 'Correct'])
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_evaluation_results(metrics: Dict, preds_df: pd.DataFrame, output_dir: str) -> None:
    """
    Save evaluation results to files.
    
    Args:
        metrics: Dictionary of evaluation metrics
        preds_df: DataFrame of predictions
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    preds_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    # Plot confusion matrix
    plot_confusion_matrix(preds_df, os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot score distribution
    plot_score_distribution(preds_df, os.path.join(output_dir, 'score_distribution.png'))
    
    # Save error analysis
    error_df = preds_df[~preds_df['correct']]
    error_df.to_csv(os.path.join(output_dir, 'errors.csv'), index=False)

def main():
    """Main function for model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate WSD model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned GlossBERT model")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to the test data CSV")
    parser.add_argument("--output_dir", type=str, default="../evaluation",
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # Load predictor
    predictor = WSDPredictor(args.model_path)
    
    # Evaluate model
    metrics, preds_df = evaluate_on_test_set(predictor, args.test_data)
    
    # Save results
    save_evaluation_results(metrics, preds_df, args.output_dir)
    
    # Print summary
    print(f"Evaluation complete. Results saved to {args.output_dir}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Correct predictions: {metrics['correct_predictions']}/{metrics['total_examples']}")

if __name__ == "__main__":
    main()
