#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify all modules can be imported correctly
"""

import sys
import logging

def test_imports():
    """Test if all modules can be imported"""
    print("="*70)
    print("Testing Module Imports")
    print("="*70)
    
    try:
        print("\n1. Testing torch and transformers...")
        import torch
        from transformers import BertTokenizer
        print(f"   ✓ PyTorch version: {torch.__version__}")
        print(f"   ✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA device: {torch.cuda.get_device_name(0)}")
        
        print("\n2. Testing src modules...")
        from src import (
            MultiTurnDialogueClassifier,
            MultiTurnDialogueDataset,
            create_dataloaders,
            TeacherTrainer,
            set_seed,
            load_config,
            count_parameters
        )
        print("   ✓ All src modules imported successfully")
        
        print("\n3. Testing configuration loading...")
        config = load_config('config/teacher_config.yaml')
        print(f"   ✓ Config loaded: {len(config)} sections")
        print(f"   ✓ Encoder: {config['model']['encoder_name']}")
        print(f"   ✓ Num labels: {config['model']['num_labels']}")
        
        print("\n4. Testing tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(config['model']['encoder_name'])
        print(f"   ✓ Tokenizer loaded: vocab size = {len(tokenizer)}")
        
        print("\n5. Testing model initialization...")
        model = MultiTurnDialogueClassifier(
            encoder_name=config['model']['encoder_name'],
            num_labels=config['model']['num_labels'],
            num_turns=config['data']['max_turns'],
            max_seq_length=config['data']['max_seq_length'],
            dropout=config['model']['dropout']
        )
        num_params = count_parameters(model)
        print(f"   ✓ Model created: {num_params:,} parameters")
        print(f"   ✓ Hidden size: {model.hidden_size}")
        
        print("\n6. Testing dummy forward pass...")
        batch_size = 2
        num_turns = 4
        seq_len = 80
        
        dummy_input = {
            'input_ids': torch.randint(0, 21128, (batch_size, num_turns, seq_len)),
            'attention_mask': torch.ones((batch_size, num_turns, seq_len), dtype=torch.long),
            'speaker_ids': torch.zeros((batch_size, num_turns), dtype=torch.long),
            'turn_ids': torch.arange(num_turns).unsqueeze(0).expand(batch_size, -1)
        }
        
        model.eval()
        with torch.no_grad():
            outputs = model(**dummy_input)
        
        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Output logits shape: {outputs['logits'].shape}")
        print(f"   ✓ Turn hidden states shape: {outputs['turn_hidden_states'].shape}")
        
        print("\n7. Checking data files...")
        import os
        data_files = ['data/train.json', 'data/dev.json', 'data/test.json']
        for file_path in data_files:
            if os.path.exists(file_path):
                size_mb = os.path.getsize(file_path) / 1024 / 1024
                print(f"   ✓ {file_path} exists ({size_mb:.1f} MB)")
            else:
                print(f"   ✗ {file_path} NOT FOUND")
        
        print("\n" + "="*70)
        print("✓ All Tests Passed!")
        print("="*70)
        print("\nYou are ready to start training:")
        print("  python train_teacher.py --config config/teacher_config.yaml")
        print("\n" + "="*70)
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_imports()
    sys.exit(0 if success else 1)

