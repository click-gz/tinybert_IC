"""
Utility functions for training
"""

import os
import random
import yaml
import numpy as np
import torch
import logging
from typing import Dict, Any


def set_seed(seed: int):
    """
    Set random seed for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logging(log_file: str = None):
    """
    Setup logging configuration
    
    Args:
        log_file: Optional path to log file
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers
    )


def create_save_dir(save_dir: str):
    """
    Create directory for saving checkpoints
    
    Args:
        save_dir: Path to save directory
    """
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Save directory created: {save_dir}")


def get_device(device_name: str = 'cuda') -> torch.device:
    """
    Get torch device
    
    Args:
        device_name: Device name ('cuda', 'cuda:0', 'cuda:1', etc. or 'cpu')
        
    Returns:
        torch.device object
    """
    if device_name.startswith('cuda') and torch.cuda.is_available():
        device = torch.device(device_name)
        # Extract GPU ID from device name (e.g., 'cuda:1' -> 1, 'cuda' -> 0)
        gpu_id = int(device_name.split(':')[1]) if ':' in device_name else 0
        logging.info(f"Using device: {device} ({torch.cuda.get_device_name(gpu_id)})")
    else:
        device = torch.device('cpu')
        logging.info(f"Using device: {device}")
    
    return device


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

