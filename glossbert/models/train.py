"""
Training script for GlossBERT Word Sense Disambiguation model.
"""

import os
import sys
import argparse
import logging
import json
import torch
import numpy as np
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertConfig,
    get_linear_schedule_with_warmup,
    set_seed
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model import GlossBERT
from data.dataset import prepare_semcor_data, create_dataloaders
from utils.wordnet_utils import save_synset_gloss_mapping

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def train(args):
    """
    Train the GlossBERT model.
    
    Args:
        args: Training arguments
    """
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save synset-gloss mapping
    synset_gloss_map_path = os.path.join(args.output_dir, 'synset_gloss_map.json')
    save_synset_gloss_mapping(synset_gloss_map_path)
    
    # Prepare data
    logger.info("Preparing SemCor data...")
    train_path, val_path = prepare_semcor_data(
        os.path.join(args.data_dir, 'processed'),
        val_split=args.val_split
    )
    
    # Load tokenizer and model
    logger.info(f"Loading BERT model: {args.model_name_or_path}")
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    config = BertConfig.from_pretrained(args.model_name_or_path)
    model = GlossBERT.from_pretrained(args.model_name_or_path, config=config)
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_path,
        val_path,
        tokenizer,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size
    )
    
    # Prepare optimizer and scheduler
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay,
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    # Calculate total training steps
    total_steps = len(train_dataloader) * args.num_train_epochs
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    logger.info(f"Training on {device}")
    
    # Track training metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    logger.info("Starting training...")
    global_step = 0
    best_val_accuracy = 0.0
    
    for epoch in range(args.num_train_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_train_epochs}")
        
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            # Track loss
            epoch_train_loss += loss.item()
            global_step += 1
            
            # Log progress
            if step % args.logging_steps == 0:
                logger.info(f"Step {step}/{len(train_dataloader)}, Loss: {loss.item():.4f}")
        
        # Calculate average training loss for the epoch
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        predictions = []
        true_labels = []
        
        for batch in tqdm(val_dataloader, desc="Validation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model(**batch)
                loss = outputs['loss']
                logits = outputs['logits']
            
            val_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            predictions.extend(preds)
            true_labels.extend(labels)
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='binary'
        )
        
        val_accuracies.append(accuracy)
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Validation Accuracy: {accuracy:.4f}")
        logger.info(f"Validation Precision: {precision:.4f}")
        logger.info(f"Validation Recall: {recall:.4f}")
        logger.info(f"Validation F1: {f1:.4f}")
        
        # Save the best model
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            logger.info(f"New best validation accuracy: {best_val_accuracy:.4f}")
            
            # Save model
            model_save_path = os.path.join(args.output_dir, 'best_model')
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            
            # Save training arguments
            with open(os.path.join(model_save_path, 'training_args.json'), 'w') as f:
                json.dump(vars(args), f, indent=2)
    
    # Save the final model
    final_model_path = os.path.join(args.output_dir, 'final_model')
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    
    logger.info(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    logger.info(f"Models saved to {args.output_dir}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GlossBERT for WSD")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="Directory containing the data")
    parser.add_argument("--output_dir", type=str, default="../models/glossbert",
                        help="Directory to save the model")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased",
                        help="Path to pre-trained model or model identifier from huggingface.co/models")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Log every X steps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
