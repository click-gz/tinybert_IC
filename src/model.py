"""
Multi-turn Dialogue Intent Classification Model (Concatenation-based)
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Dict, Optional


class MultiTurnDialogueClassifier(nn.Module):
    """
    Multi-turn dialogue intent classifier using concatenation approach
    Concatenates all turns into a single sequence with speaker markers
    Supports both Teacher (BERT-base) and Student (TinyBERT)
    """
    
    def __init__(
        self,
        encoder_name: str,
        num_labels: int = 12,
        max_seq_length: int = 320,  # Total length for all turns (4 turns * 80)
        dropout: float = 0.1,
        # Keep these for backward compatibility but not used
        num_turns: int = 4,
    ):
        """
        Args:
            encoder_name: Name of pre-trained encoder (e.g., 'bert-base-chinese')
            num_labels: Number of intent classes
            max_seq_length: Maximum sequence length for concatenated dialogue
            dropout: Dropout probability
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        
        # 1. BERT Encoder (processes concatenated dialogue)
        self.encoder = BertModel.from_pretrained(encoder_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # 2. Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize classifier
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: (batch_size, seq_len) - concatenated dialogue sequence
            attention_mask: (batch_size, seq_len)
            token_type_ids: (batch_size, seq_len) - optional, for segment embeddings
            output_hidden_states: Whether to output hidden states (for distillation)
            output_attentions: Whether to output attentions (for distillation)
        
        Returns:
            Dictionary containing:
                - logits: (batch_size, num_labels)
                - hidden_state: (batch_size, hidden_size) - CLS representation
                - attentions: List of attention tensors (if requested)
                - all_hidden_states: All hidden states (if requested)
        """
        # Encode the concatenated dialogue sequence
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=True
        )
        
        # Take [CLS] token representation
        cls_output = encoder_outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Classification
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)  # (batch_size, num_labels)
        
        # Prepare output
        result = {
            'logits': logits,
            'hidden_state': cls_output,  # For distillation
        }
        
        if output_attentions:
            result['attentions'] = encoder_outputs.attentions
        
        if output_hidden_states:
            result['all_hidden_states'] = encoder_outputs.hidden_states
        
        return result
    
    def get_encoder_parameters(self):
        """Get encoder parameters (for differential learning rate)"""
        return self.encoder.parameters()
    
    def get_task_parameters(self):
        """Get task-specific parameters (only classifier now)"""
        return self.classifier.parameters()
