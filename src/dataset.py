"""
Dataset and DataLoader for multi-turn dialogue intent classification
Using concatenation approach - all turns are concatenated into a single sequence
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Dict


class MultiTurnDialogueDataset(Dataset):
    """
    Multi-turn dialogue dataset with concatenation approach
    Handles JSON format dialogue data with variable number of turns
    Concatenates all turns into a single sequence with speaker markers
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        max_seq_length: int = 512,  # Total length for concatenated sequence
        is_training: bool = True
    ):
        """
        Args:
            data_path: Path to JSON data file
            tokenizer: BERT tokenizer
            max_seq_length: Maximum token length for concatenated dialogue
            is_training: Whether in training mode
        """
        super().__init__()
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_training = is_training
        
        print(f"Loaded {len(self.data)} dialogues from {data_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single dialogue sample
        
        Returns:
            Dictionary containing:
                - input_ids: (seq_len,)
                - attention_mask: (seq_len,)
                - token_type_ids: (seq_len,)
                - labels: scalar
                - dialogue_id: string
                - num_turns: scalar (actual number of turns)
        """
        item = self.data[idx]
        
        # Extract dialogue turns
        turns = item['turns']
        label = item['label']
        dialogue_id = item.get('dialogue_id', f'dia_{idx}')
        num_turns = len(turns)
        
        # Build concatenated text with speaker markers
        # Format: [CLS] [USER] turn1_text [SEP] [SYSTEM] turn2_text [SEP] ...
        dialogue_parts = []
        for turn in turns:
            speaker = turn['speaker'].upper()  # USER or SYSTEM
            text = turn['text']
            # Add speaker marker and text
            dialogue_parts.append(f"[{speaker}] {text}")
        
        # Join all turns with space
        full_dialogue = " ".join(dialogue_parts)
        
        # Tokenize the concatenated dialogue
        encoded = self.tokenizer(
            full_dialogue,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),  # (seq_len,)
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded['token_type_ids'].squeeze(0) if 'token_type_ids' in encoded else torch.zeros_like(encoded['input_ids'].squeeze(0)),
            'labels': torch.tensor(label, dtype=torch.long),
            'dialogue_id': dialogue_id,
            'num_turns': num_turns
        }


def create_dataloaders(
    train_path: str,
    dev_path: str,
    test_path: str,
    tokenizer: BertTokenizer,
    batch_size: int = 32,
    max_seq_length: int = 512,  # Total length for concatenated dialogue
    num_workers: int = 4
) -> tuple:
    """
    Create train, dev, and test dataloaders
    
    Args:
        train_path: Path to training data
        dev_path: Path to development data
        test_path: Path to test data
        tokenizer: BERT tokenizer
        batch_size: Batch size
        max_seq_length: Maximum sequence length for concatenated dialogue
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, dev_loader, test_loader)
    """
    # Create datasets
    train_dataset = MultiTurnDialogueDataset(
        train_path, tokenizer, max_seq_length, is_training=True
    )
    dev_dataset = MultiTurnDialogueDataset(
        dev_path, tokenizer, max_seq_length, is_training=False
    )
    test_dataset = MultiTurnDialogueDataset(
        test_path, tokenizer, max_seq_length, is_training=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, dev_loader, test_loader
