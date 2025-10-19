#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Teacher Model (BERT-base) for Multi-turn Dialogue Intent Classification

Usage:
    python train_teacher.py --config config/teacher_config.yaml
"""

import argparse
import logging
from transformers import BertTokenizer

from src import (
    MultiTurnDialogueClassifier,
    create_dataloaders,
    TeacherTrainer,
    set_seed,
    load_config,
    count_parameters,
    setup_logging,
    create_save_dir,
    get_device
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Teacher Model')
    parser.add_argument(
        '--config',
        type=str,
        default='config/teacher_config.yaml',
        help='Path to configuration file'
    )
    return parser.parse_args()


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging()
    
    logging.info("="*80)
    logging.info("Teacher Model Training")
    logging.info("="*80)
    logging.info(f"Configuration file: {args.config}")
    
    # Set random seed
    set_seed(config['seed'])
    logging.info(f"Random seed: {config['seed']}")
    
    # Get device
    device = get_device(config['device'])
    
    # Create save directory
    create_save_dir(config['logging']['save_dir'])
    
    # =========================================================================
    # Step 1: Prepare Data
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("Step 1: Loading Data")
    logging.info("="*80)
    
    tokenizer = BertTokenizer.from_pretrained(config['model']['encoder_name'])
    logging.info(f"Tokenizer: {config['model']['encoder_name']}")
    logging.info(f"Vocab size: {len(tokenizer)}")
    
    train_loader, dev_loader, test_loader = create_dataloaders(
        train_path=config['data']['train_path'],
        dev_path=config['data']['dev_path'],
        test_path=config['data']['test_path'],
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_turns=config['data']['max_turns'],
        max_seq_length=config['data']['max_seq_length'],
        num_workers=config['data'].get('num_workers', 4)
    )
    
    logging.info(f"Training samples: {len(train_loader.dataset)}")
    logging.info(f"Dev samples: {len(dev_loader.dataset)}")
    logging.info(f"Test samples: {len(test_loader.dataset)}")
    
    # =========================================================================
    # Step 2: Create Model
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("Step 2: Creating Teacher Model")
    logging.info("="*80)
    
    model = MultiTurnDialogueClassifier(
        encoder_name=config['model']['encoder_name'],
        num_labels=config['model']['num_labels'],
        num_turns=config['data']['max_turns'],
        max_seq_length=config['data']['max_seq_length'],
        dropout=config['model']['dropout']
    )
    
    num_params = count_parameters(model)
    logging.info(f"Model: {config['model']['encoder_name']}")
    logging.info(f"Hidden size: {model.hidden_size}")
    logging.info(f"Number of labels: {config['model']['num_labels']}")
    logging.info(f"Trainable parameters: {num_params:,}")
    logging.info(f"Model size: ~{num_params / 1e6:.1f}M parameters")
    
    # =========================================================================
    # Step 3: Create Trainer
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("Step 3: Initializing Trainer")
    logging.info("="*80)
    
    trainer = TeacherTrainer(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        encoder_lr=config['training']['encoder_lr'],
        task_lr=config['training']['task_lr'],
        weight_decay=config['training']['weight_decay'],
        num_epochs=config['training']['num_epochs'],
        warmup_ratio=config['training']['warmup_ratio'],
        gradient_clip=config['training']['gradient_clip'],
        save_dir=config['logging']['save_dir'],
        tensorboard_dir=config['logging']['tensorboard_dir'],
        log_every=config['logging']['log_every'],
        patience=config['training']['patience']
    )
    
    logging.info("Trainer initialized successfully")
    
    # =========================================================================
    # Step 4: Start Training
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("Step 4: Training")
    logging.info("="*80)
    
    trainer.train()
    
    # =========================================================================
    # Training Complete
    # =========================================================================
    logging.info("\n" + "="*80)
    logging.info("All Done!")
    logging.info("="*80)
    logging.info(f"Best model saved to: {config['logging']['save_dir']}/best_model.pt")
    logging.info(f"TensorBoard logs: {config['logging']['tensorboard_dir']}")
    logging.info("\nTo view training logs, run:")
    logging.info(f"  tensorboard --logdir {config['logging']['tensorboard_dir']}")
    logging.info("\nTo continue with distillation:")
    logging.info(f"  python train_distill.py --teacher {config['logging']['save_dir']}/best_model.pt")
    logging.info("="*80)


if __name__ == '__main__':
    main()

