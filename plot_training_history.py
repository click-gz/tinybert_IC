#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot training history from saved CSV or JSON file

Usage:
    python plot_training_history.py --history checkpoints/teacher/training_history.csv
    python plot_training_history.py --history checkpoints/teacher/training_history.json --output my_plot.png
"""

import argparse
import json
import csv
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def load_history_from_csv(csv_path):
    """Load training history from CSV file"""
    history = {
        'epoch': [],
        'train_loss': [],
        'train_f1': [],
        'dev_loss': [],
        'dev_f1': [],
        'learning_rate': []
    }
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history['epoch'].append(int(row['epoch']))
            history['train_loss'].append(float(row['train_loss']))
            history['train_f1'].append(float(row['train_f1']))
            history['dev_loss'].append(float(row['dev_loss']))
            history['dev_f1'].append(float(row['dev_f1']))
            history['learning_rate'].append(float(row['learning_rate']))
    
    return history


def load_history_from_json(json_path):
    """Load training history from JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        history = json.load(f)
    return history


def plot_training_history(history, output_path='training_plot.png'):
    """
    Plot training history with multiple subplots
    
    Args:
        history: Dictionary containing training history
        output_path: Path to save the plot
    """
    epochs = history['epoch']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['dev_loss'], 'r-s', label='Dev Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 curves
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_f1'], 'b-o', label='Train F1', linewidth=2, markersize=4)
    ax2.plot(epochs, history['dev_f1'], 'r-s', label='Dev F1', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Macro F1 Score', fontsize=12)
    ax2.set_title('Training and Validation F1 Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Mark best epoch
    best_epoch_idx = history['dev_f1'].index(max(history['dev_f1']))
    best_epoch = epochs[best_epoch_idx]
    best_f1 = history['dev_f1'][best_epoch_idx]
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax2.plot(best_epoch, best_f1, 'g*', markersize=15, label=f'Best (Epoch {best_epoch})')
    ax2.legend(fontsize=10)
    
    # Plot 3: Learning rate
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['learning_rate'], 'purple', linewidth=2)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Learning Rate', fontsize=12)
    ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Plot 4: Loss difference (overfitting indicator)
    ax4 = axes[1, 1]
    loss_diff = [train - dev for train, dev in zip(history['train_loss'], history['dev_loss'])]
    f1_diff = [train - dev for train, dev in zip(history['train_f1'], history['dev_f1'])]
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(epochs, loss_diff, 'b-o', label='Loss Gap (Train-Dev)', linewidth=2, markersize=4)
    line2 = ax4_twin.plot(epochs, f1_diff, 'r-s', label='F1 Gap (Train-Dev)', linewidth=2, markersize=4)
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Loss Gap', fontsize=12, color='b')
    ax4_twin.set_ylabel('F1 Gap', fontsize=12, color='r')
    ax4.set_title('Train-Dev Gap (Overfitting Indicator)', fontsize=14, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='r')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, fontsize=10)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Plot saved to: {pdf_path}")
    
    plt.close()


def print_summary(history):
    """Print summary statistics of training"""
    print("\n" + "="*70)
    print("Training Summary")
    print("="*70)
    
    best_epoch_idx = history['dev_f1'].index(max(history['dev_f1']))
    best_epoch = history['epoch'][best_epoch_idx]
    
    print(f"\nTotal Epochs: {len(history['epoch'])}")
    print(f"Best Epoch: {best_epoch}")
    print(f"\nBest Performance:")
    print(f"  Dev Loss:  {history['dev_loss'][best_epoch_idx]:.4f}")
    print(f"  Dev F1:    {history['dev_f1'][best_epoch_idx]:.4f}")
    print(f"  Train Loss: {history['train_loss'][best_epoch_idx]:.4f}")
    print(f"  Train F1:   {history['train_f1'][best_epoch_idx]:.4f}")
    
    print(f"\nFinal Performance (Epoch {history['epoch'][-1]}):")
    print(f"  Dev Loss:  {history['dev_loss'][-1]:.4f}")
    print(f"  Dev F1:    {history['dev_f1'][-1]:.4f}")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train F1:   {history['train_f1'][-1]:.4f}")
    
    # Overfitting analysis
    loss_gap = history['train_loss'][-1] - history['dev_loss'][-1]
    f1_gap = history['train_f1'][-1] - history['dev_f1'][-1]
    print(f"\nOverfitting Indicators (at final epoch):")
    print(f"  Loss Gap (Train-Dev):  {loss_gap:+.4f}")
    print(f"  F1 Gap (Train-Dev):    {f1_gap:+.4f}")
    
    if f1_gap > 0.05:
        print("  ⚠️  Warning: Possible overfitting detected (large F1 gap)")
    elif f1_gap < -0.05:
        print("  ⚠️  Warning: Model may be underfitting on training data")
    else:
        print("  ✓ Training appears well-balanced")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Plot training history')
    parser.add_argument(
        '--history',
        type=str,
        required=True,
        help='Path to training history file (CSV or JSON)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='training_history_plot.png',
        help='Output path for the plot'
    )
    
    args = parser.parse_args()
    
    # Load history
    if args.history.endswith('.csv'):
        history = load_history_from_csv(args.history)
    elif args.history.endswith('.json'):
        history = load_history_from_json(args.history)
    else:
        raise ValueError("History file must be .csv or .json")
    
    print(f"Loaded training history from: {args.history}")
    
    # Print summary
    print_summary(history)
    
    # Plot
    plot_training_history(history, args.output)


if __name__ == '__main__':
    main()

