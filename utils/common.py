import torch
import random
import numpy as np


def stratified_sample_instances(class_labels, foreground_target, batch_capacity):

    fg_indices = torch.where(class_labels >= 1)[0]
    bg_indices = torch.where(class_labels == 0)[0]
    actual_fg = min(fg_indices.numel(), foreground_target)
    actual_bg = min(bg_indices.numel(), batch_capacity - actual_fg)
    fg_permutation = torch.randperm(fg_indices.numel(), device=fg_indices.device)[:actual_fg]
    bg_permutation = torch.randperm(bg_indices.numel(), device=bg_indices.device)[:actual_bg]
    
    selected_fg = fg_indices[fg_permutation]
    selected_bg = bg_indices[bg_permutation]
    fg_selection_mask = torch.zeros_like(class_labels, dtype=torch.bool)
    bg_selection_mask = torch.zeros_like(class_labels, dtype=torch.bool)
    
    fg_selection_mask[selected_fg] = True
    bg_selection_mask[selected_bg] = True
    
    return bg_selection_mask, fg_selection_mask


def set_random_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_computation_device():
  
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_model_summary(model):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("=" * 80)


def format_time(seconds):

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"