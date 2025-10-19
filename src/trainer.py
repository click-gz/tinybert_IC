"""
Teacher Model Trainer
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import logging


class TeacherTrainer:
    """
    Trainer for Teacher model (BERT-base)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        dev_loader,
        device: str = 'cuda',
        encoder_lr: float = 2e-5,
        task_lr: float = 4e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 5,
        warmup_ratio: float = 0.1,
        gradient_clip: float = 1.0,
        save_dir: str = 'checkpoints/teacher',
        tensorboard_dir: str = 'runs/teacher',
        log_every: int = 100,
        patience: int = 3
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            dev_loader: Development data loader
            device: Device to train on
            encoder_lr: Learning rate for encoder
            task_lr: Learning rate for task-specific layers
            weight_decay: Weight decay for regularization
            num_epochs: Number of training epochs
            warmup_ratio: Ratio of warmup steps
            gradient_clip: Gradient clipping value
            save_dir: Directory to save checkpoints
            tensorboard_dir: Directory for tensorboard logs
            log_every: Log every N steps
            patience: Early stopping patience
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_every = log_every
        self.gradient_clip = gradient_clip
        self.patience = patience
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer with differential learning rates
        encoder_params = list(model.get_encoder_parameters())
        task_params = list(model.get_task_parameters())
        
        self.optimizer = AdamW([
            {'params': encoder_params, 'lr': encoder_lr},
            {'params': task_params, 'lr': task_lr}
        ], weight_decay=weight_decay)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # TensorBoard writer
        self.writer = SummaryWriter(tensorboard_dir)
        
        # Best model tracking
        self.best_f1 = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.global_step = 0
        
        logging.info(f"Total training steps: {total_steps}")
        logging.info(f"Warmup steps: {warmup_steps}")
        logging.info(f"Encoder LR: {encoder_lr}, Task LR: {task_lr}")
    
    def train_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Tuple of (average_loss, macro_f1)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        
        for step, batch in enumerate(pbar):
            # Move data to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device) if 'token_type_ids' in batch else None
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs['logits']
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # TensorBoard logging
            if step % self.log_every == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
            
            self.global_step += 1
        
        # Calculate metrics
        avg_loss = total_loss / len(self.train_loader)
        train_f1 = f1_score(all_labels, all_preds, average='macro')
        
        return avg_loss, train_f1
    
    def evaluate(self) -> tuple:
        """
        Evaluate on development set
        
        Returns:
            Tuple of (average_loss, macro_f1, classification_report)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.dev_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device) if 'token_type_ids' in batch else None
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                logits = outputs['logits']
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.dev_loader)
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        
        # Generate classification report
        report = classification_report(all_labels, all_preds, digits=4)
        
        return avg_loss, macro_f1, report
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_f1': self.best_f1,
            'global_step': self.global_step
        }
        
        # Save last checkpoint
        last_path = os.path.join(self.save_dir, 'last_model.pt')
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logging.info(f"✓ Saved best model with F1: {self.best_f1:.4f}")
    
    def train(self):
        """
        Complete training loop with early stopping
        """
        logging.info("="*70)
        logging.info("Starting Teacher Model Training")
        logging.info("="*70)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            logging.info(f"\n{'='*70}")
            logging.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logging.info(f"{'='*70}")
            
            # Train
            train_loss, train_f1 = self.train_epoch(epoch)
            logging.info(f"Train Loss: {train_loss:.4f}, Train Macro-F1: {train_f1:.4f}")
            
            # Evaluate
            dev_loss, dev_f1, report = self.evaluate()
            logging.info(f"Dev Loss: {dev_loss:.4f}, Dev Macro-F1: {dev_f1:.4f}")
            
            # Print classification report
            logging.info("\nClassification Report:")
            logging.info("\n" + report)
            
            # TensorBoard logging
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/train_f1', train_f1, epoch)
            self.writer.add_scalar('epoch/dev_loss', dev_loss, epoch)
            self.writer.add_scalar('epoch/dev_f1', dev_f1, epoch)
            
            # Check for improvement
            is_best = False
            if dev_f1 > self.best_f1:
                self.best_f1 = dev_f1
                self.best_epoch = epoch
                is_best = True
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logging.info(f"\nEarly stopping triggered after {self.patience} epochs without improvement")
                break
        
        # Training complete
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        logging.info(f"\n{'='*70}")
        logging.info("Training Completed!")
        logging.info(f"{'='*70}")
        logging.info(f"Total training time: {hours}h {minutes}m")
        logging.info(f"Best Dev F1: {self.best_f1:.4f} (Epoch {self.best_epoch + 1})")
        logging.info(f"Model saved to: {self.save_dir}")
        logging.info(f"TensorBoard logs: {self.writer.log_dir}")
        logging.info("="*70)
        
        self.writer.close()

