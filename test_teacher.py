#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test/Evaluate Teacher Model on Test Set

Usage:
    python test_teacher.py --config config/teacher_config.yaml --checkpoint checkpoints/teacher/best_model.pt
"""

import argparse
import logging
import torch
import torch.nn as nn
from transformers import BertTokenizer
from sklearn.metrics import (
    f1_score, 
    accuracy_score, 
    precision_score, 
    recall_score,
    classification_report,
    confusion_matrix
)
from tqdm import tqdm
import numpy as np
import json

from src import (
    MultiTurnDialogueClassifier,
    create_dataloaders,
    load_config,
    setup_logging,
    get_device,
    count_parameters
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test Teacher Model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/teacher_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/teacher/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_results.json',
        help='Path to save test results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size for testing (default: use config value)'
    )
    return parser.parse_args()


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model checkpoint
    
    Args:
        model: Model instance
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model and checkpoint info
    """
    logging.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    epoch = checkpoint.get('epoch', -1)
    best_f1 = checkpoint.get('best_f1', -1)
    
    logging.info(f"Checkpoint loaded successfully")
    logging.info(f"  Epoch: {epoch + 1}")
    logging.info(f"  Best F1: {best_f1:.4f}")
    
    return model, checkpoint


def evaluate_model(model, data_loader, device, criterion):
    """
    Evaluate model on given dataset
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run on
        criterion: Loss function
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs['logits']
            loss = criterion(logits, labels)
            
            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            # Collect results
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, digits=4)
    
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class.tolist(),
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'confusion_matrix': conf_matrix.tolist(),
        'predictions': np.array(all_preds).tolist(),
        'labels': np.array(all_labels).tolist(),
        'probabilities': np.array(all_probs).tolist()
    }
    
    return results, report


def print_results(results, report):
    """
    Print evaluation results in a nice format
    
    Args:
        results: Dictionary of evaluation metrics
        report: Classification report string
    """
    logging.info("\n" + "="*80)
    logging.info("Test Results")
    logging.info("="*80)
    
    logging.info(f"\nOverall Metrics:")
    logging.info(f"  Loss:            {results['loss']:.4f}")
    logging.info(f"  Accuracy:        {results['accuracy']:.4f}")
    logging.info(f"  Precision (Macro): {results['precision_macro']:.4f}")
    logging.info(f"  Recall (Macro):    {results['recall_macro']:.4f}")
    logging.info(f"  F1 (Macro):        {results['f1_macro']:.4f}")
    logging.info(f"  F1 (Weighted):     {results['f1_weighted']:.4f}")
    
    logging.info(f"\nPer-Class F1 Scores:")
    for i, f1 in enumerate(results['f1_per_class']):
        logging.info(f"  Class {i:2d}: {f1:.4f}")
    
    logging.info("\nClassification Report:")
    logging.info("\n" + report)
    
    logging.info("\nConfusion Matrix:")
    conf_matrix = np.array(results['confusion_matrix'])
    for i, row in enumerate(conf_matrix):
        logging.info(f"  Class {i:2d}: {row}")
    
    logging.info("\n" + "="*80)


def save_results(results, output_path):
    """
    Save results to JSON file
    
    Args:
        results: Dictionary of evaluation metrics
        output_path: Path to save results
    """
    # Create a copy without predictions/labels/probs for summary
    summary = {
        'loss': results['loss'],
        'accuracy': results['accuracy'],
        'precision_macro': results['precision_macro'],
        'recall_macro': results['recall_macro'],
        'f1_macro': results['f1_macro'],
        'f1_weighted': results['f1_weighted'],
        'f1_per_class': results['f1_per_class'],
        'precision_per_class': results['precision_per_class'],
        'recall_per_class': results['recall_per_class'],
        'confusion_matrix': results['confusion_matrix']
    }
    
    # Save summary
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Results saved to: {output_path}")
    
    # Save detailed results (with predictions) separately
    detailed_path = output_path.replace('.json', '_detailed.json')
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Detailed results saved to: {detailed_path}")


def main():
    """Main testing function"""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    setup_logging()
    
    logging.info("="*80)
    logging.info("Teacher Model Testing")
    logging.info("="*80)
    logging.info(f"Configuration file: {args.config}")
    logging.info(f"Checkpoint file: {args.checkpoint}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Get device
    device = get_device(config['device'])
    
    # Override batch size if specified
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
        logging.info(f"Using batch size: {args.batch_size}")
    
    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("Step 1: Loading Test Data")
    logging.info("="*80)
    
    tokenizer = BertTokenizer.from_pretrained(config['model']['encoder_name'])
    logging.info(f"Tokenizer: {config['model']['encoder_name']}")
    
    _, _, test_loader = create_dataloaders(
        train_path=config['data']['train_path'],
        dev_path=config['data']['dev_path'],
        test_path=config['data']['test_path'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_seq_length=config['data']['max_seq_length'],
        num_workers=config['data'].get('num_workers', 4)
    )
    
    logging.info(f"Test samples: {len(test_loader.dataset)}")
    logging.info(f"Test batches: {len(test_loader)}")
    
    # =========================================================================
    # Step 2: Load Model
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("Step 2: Loading Model")
    logging.info("="*80)
    
    model = MultiTurnDialogueClassifier(
        encoder_name=config['model']['encoder_name'],
        num_labels=config['model']['num_labels'],
        max_seq_length=config['data']['max_seq_length'],
        dropout=config['model']['dropout']
    )
    
    num_params = count_parameters(model)
    logging.info(f"Model: {config['model']['encoder_name']}")
    logging.info(f"Parameters: {num_params:,} (~{num_params / 1e6:.1f}M)")
    
    # Load checkpoint
    model, checkpoint = load_checkpoint(model, args.checkpoint, device)
    model = model.to(device)
    
    # =========================================================================
    # Step 3: Evaluate on Test Set
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("Step 3: Evaluating on Test Set")
    logging.info("="*80)
    
    criterion = nn.CrossEntropyLoss()
    results, report = evaluate_model(model, test_loader, device, criterion)
    
    # Print results
    print_results(results, report)
    
    # Save results
    save_results(results, args.output)
    
    # =========================================================================
    # Complete
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("Testing Complete!")
    logging.info("="*80)
    logging.info(f"Test F1 (Macro): {results['f1_macro']:.4f}")
    logging.info(f"Test Accuracy: {results['accuracy']:.4f}")
    logging.info(f"Results saved to: {args.output}")
    logging.info("="*80)


if __name__ == '__main__':
    main()

