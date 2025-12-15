import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from dataset import JanitorialDataset
from train import MultiHeadProbe, get_transforms # Import from train to ensure consistency

def evaluate(args, dataset=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms
    transform = get_transforms(args.backbone)
    
    # Dataset
    if dataset is None:
        # Use 'test' split if available, else 'val'
        split = 'test'
        dataset = JanitorialDataset(args.csv, args.root, split=split, transform=transform, use_masks=args.use_masks)
        if len(dataset) == 0:
            print("Test split empty, falling back to val split.")
            split = 'val'
            dataset = JanitorialDataset(args.csv, args.root, split=split, transform=transform, use_masks=args.use_masks)
        
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Evaluating on {split} set ({len(dataset)} samples).")
    
    # Model
    model = MultiHeadProbe(args.backbone, num_tasks=11, num_states=2, device=device).to(device)
    
    # Load weights
    model_path = os.path.join(args.output_dir, f'model_{args.backbone.replace("/", "_")}.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights.")
        
    model.eval()
    
    all_task_preds = []
    all_task_labels = []
    all_state_preds = []
    all_state_labels = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            task_labels = batch['task_label'].to(device)
            state_labels = batch['state_label'].to(device)
            
            task_logits, state_logits = model(images)
            
            _, predicted_task = torch.max(task_logits.data, 1)
            _, predicted_state = torch.max(state_logits.data, 1)
            
            all_task_preds.extend(predicted_task.cpu().numpy())
            all_task_labels.extend(task_labels.cpu().numpy())
            
            # Filter ignored states
            mask = state_labels != -1
            if mask.sum() > 0:
                all_state_preds.extend(predicted_state[mask].cpu().numpy())
                all_state_labels.extend(state_labels[mask].cpu().numpy())
                
    # Task Metrics
    task_keys = sorted(JanitorialDataset.TASK_MAP.values())
    task_names = [k for k, v in sorted(JanitorialDataset.TASK_MAP.items(), key=lambda item: item[1])]
    
    # Check if all classes present
    unique_labels = sorted(list(set(all_task_labels)))
    print(f"Unique task labels in GT: {unique_labels}")
    
    print("\nTask Classification Report:")
    print(classification_report(all_task_labels, all_task_preds, target_names=task_names, labels=task_keys, zero_division=0))
    
    # Task Confusion Matrix
    cm_task = confusion_matrix(all_task_labels, all_task_preds, labels=task_keys)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_task, annot=True, fmt='d', xticklabels=task_names, yticklabels=task_names, cmap='Blues')
    plt.title(f'Task Confusion Matrix ({args.backbone})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_path = os.path.join(args.output_dir, f'confusion_matrix_task_{args.backbone.replace("/", "_")}.png')
    plt.savefig(save_path)
    print(f"Saved task confusion matrix to {save_path}")
    
    # State Metrics
    state_names = ['Active', 'Done']
    print("\nState Classification Report:")
    print(classification_report(all_state_labels, all_state_preds, target_names=state_names, zero_division=0))
    
    # State Confusion Matrix
    cm_state = confusion_matrix(all_state_labels, all_state_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_state, annot=True, fmt='d', xticklabels=state_names, yticklabels=state_names, cmap='Greens')
    plt.title(f'State Confusion Matrix ({args.backbone})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    save_path_state = os.path.join(args.output_dir, f'confusion_matrix_state_{args.backbone.replace("/", "_")}.png')
    plt.savefig(save_path_state)
    print(f"Saved state confusion matrix to {save_path_state}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='master_dataset.csv', help='Path to master_dataset.csv')
    parser.add_argument('--root', type=str, default='.', help='Root directory containing data/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--backbone', type=str, default='vit_small_patch16_dinov3', 
                        help='Backbone model name. Examples: vit_small_patch16_dinov3 (DINOv3), openai/clip-vit-base-patch16 (CLIP)')
    parser.add_argument('--use_masks', action='store_true', help='Use SAM 3 generated masks')
    args = parser.parse_args()
    
    evaluate(args)
