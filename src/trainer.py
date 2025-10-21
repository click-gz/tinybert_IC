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
import json
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend


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
        patience: int = 3,
        keep_checkpoint_max: int = 2
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
            keep_checkpoint_max: Maximum number of recent checkpoints to keep
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
        
        # Keep track of recent checkpoints (for cleanup)
        self.recent_checkpoints = []
        self.max_checkpoints_to_keep = keep_checkpoint_max
        
        # Training history tracking
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_f1': [],
            'dev_loss': [],
            'dev_f1': [],
            'learning_rate': []
        }
        
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
    
    def save_training_history(self):
        """Save training history to CSV and JSON files"""
        # Save as CSV
        csv_path = os.path.join(self.save_dir, 'training_history.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.history.keys())
            writer.writeheader()
            # Write rows
            num_epochs = len(self.history['epoch'])
            for i in range(num_epochs):
                row = {key: self.history[key][i] for key in self.history.keys()}
                writer.writerow(row)
        
        logging.info(f"Training history saved to: {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(self.save_dir, 'training_history.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Training history saved to: {json_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        if len(self.history['epoch']) == 0:
            return
        
        epochs = self.history['epoch']
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss curves
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['dev_loss'], 'r-', label='Dev Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1 curves
        ax2.plot(epochs, self.history['train_f1'], 'b-', label='Train F1', linewidth=2)
        ax2.plot(epochs, self.history['dev_f1'], 'r-', label='Dev F1', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Macro F1 Score', fontsize=12)
        ax2.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Mark best epoch
        if self.best_epoch < len(epochs):
            best_f1 = self.history['dev_f1'][self.best_epoch]
            ax2.axvline(x=epochs[self.best_epoch], color='g', linestyle='--', 
                       alpha=0.5, label=f'Best Epoch ({epochs[self.best_epoch]})')
            ax2.plot(epochs[self.best_epoch], best_f1, 'g*', markersize=15)
            ax2.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"Training curves saved to: {plot_path}")
        
        # Also save as PDF
        pdf_path = os.path.join(self.save_dir, 'training_curves.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        
        plt.close()
    
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
        
        # Save epoch checkpoint (keep last 2)
        epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(checkpoint, epoch_path)
        self.recent_checkpoints.append(epoch_path)
        
        # Remove old checkpoints if we have more than max_checkpoints_to_keep
        if len(self.recent_checkpoints) > self.max_checkpoints_to_keep:
            old_checkpoint = self.recent_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                logging.info(f"Removed old checkpoint: {os.path.basename(old_checkpoint)}")
        
        # Save last checkpoint (always keep)
        last_path = os.path.join(self.save_dir, 'last_model.pt')
        torch.save(checkpoint, last_path)
        
        # Save best checkpoint (always keep)
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
            
            # Record training history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(train_loss)
            self.history['train_f1'].append(train_f1)
            self.history['dev_loss'].append(dev_loss)
            self.history['dev_f1'].append(dev_f1)
            self.history['learning_rate'].append(self.scheduler.get_last_lr()[0])
            
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
        
        # Save training history and plot curves
        logging.info(f"\n{'='*70}")
        logging.info("Saving Training History and Plots")
        logging.info(f"{'='*70}")
        self.save_training_history()
        self.plot_training_curves()
        
        logging.info(f"\n{'='*70}")
        logging.info("Training Completed!")
        logging.info(f"{'='*70}")
        logging.info(f"Total training time: {hours}h {minutes}m")
        logging.info(f"Best Dev F1: {self.best_f1:.4f} (Epoch {self.best_epoch + 1})")
        logging.info(f"Model saved to: {self.save_dir}")
        logging.info(f"Training history: {self.save_dir}/training_history.csv")
        logging.info(f"Training curves: {self.save_dir}/training_curves.png")
        logging.info(f"TensorBoard logs: {self.writer.log_dir}")
        logging.info("="*70)
        
        self.writer.close()

