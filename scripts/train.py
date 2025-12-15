import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import time
import ssl

# Fix for SSL Certificate Error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from dataset import JanitorialDataset

# Import timm for DINOv3 and transformers for CLIP
import timm
from transformers import CLIPModel, CLIPProcessor

class MultiHeadProbe(nn.Module):
    def __init__(self, backbone_name, num_tasks, num_states, device):
        super().__init__()
        self.backbone_name = backbone_name
        self.device = device
        
        if 'dino' in backbone_name:
            # Load DINOv3 using timm
            print(f"Loading DINO model: {backbone_name}")
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
            self.embed_dim = self.backbone.num_features
            self.processor = None # DINOv3 uses standard transforms
            
        elif 'clip' in backbone_name:
            # Load CLIP using transformers
            print(f"Loading CLIP model: {backbone_name}")
            self.backbone = CLIPModel.from_pretrained(backbone_name).vision_model
            self.embed_dim = self.backbone.config.hidden_size
            self.processor = CLIPProcessor.from_pretrained(backbone_name)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
            
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Task Head (Multi-class)
        self.task_head = nn.Linear(self.embed_dim, num_tasks)
        
        # State Head (Binary)
        self.state_head = nn.Linear(self.embed_dim, num_states)
        
    def forward(self, x):
        if 'dino' in self.backbone_name:
            features = self.backbone(x)
        elif 'clip' in self.backbone_name:
            # CLIP expects pixel_values
            outputs = self.backbone(pixel_values=x)
            features = outputs.pooler_output
            
        task_logits = self.task_head(features)
        state_logits = self.state_head(features)
        
        return task_logits, state_logits

def get_transforms(backbone_name):
    if 'dino' in backbone_name:
        # Standard ImageNet transform for DINO
        # DINOv2 usually likes 518x518, DINOv3 (ViT-S/16) likes 224x224 or 384x384
        size = 224
        if 'dinov2' in backbone_name:
            size = 518
        elif 'patch16' in backbone_name:
            size = 224 # patch16 typically 224 training
            
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif 'clip' in backbone_name:
        # CLIP Processor handles normalization, but we need to resize/tensor first if using Dataset
        # Actually, let's just use standard CLIP mean/std
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    return None

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transforms
    transform = get_transforms(args.backbone)
    
    # Datasets
    train_dataset = JanitorialDataset(args.csv, args.root, split='train', transform=transform, use_masks=args.use_masks)
    val_dataset = JanitorialDataset(args.csv, args.root, split='val', transform=transform, use_masks=args.use_masks)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Model
    num_states = 2
    model = MultiHeadProbe(args.backbone, num_tasks=11, num_states=num_states, device=device).to(device)
    
    # Calculate Class Weights for State Imbalance
    # Simple count: Active (0) vs Done (1)
    state_counts = torch.zeros(num_states)
    for s in train_dataset.samples:
        lbl = s['state_label']
        if lbl != -1:
            state_counts[lbl] += 1
            
    print(f"Class Distribution: Active={int(state_counts[0])}, Done={int(state_counts[1])}")
    
    # Weight = Total / (Count * Num_Classes) or just Inverse Frequency
    # Let's use simple inverse: weight = Max_Count / Count
    max_count = torch.max(state_counts)
    weights = max_count / (state_counts + 1e-6) # Avoid zero div
    weights = weights.to(device)
    print(f"Using Weights: {weights}")

    # Loss & Optimizer
    criterion_task = nn.CrossEntropyLoss()
    criterion_state = nn.CrossEntropyLoss(weight=weights, ignore_index=-1) # Ignore 'Other'/-1
    
    optimizer = optim.Adam(list(model.task_head.parameters()) + list(model.state_head.parameters()), lr=args.lr)
    
    # Training Loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            task_labels = batch['task_label'].to(device)
            state_labels = batch['state_label'].to(device)
            
            optimizer.zero_grad()
            
            task_logits, state_logits = model(images)
            
            loss_task = criterion_task(task_logits, task_labels)
            loss_state = criterion_state(state_logits, state_labels)
            
            loss = loss_task + loss_state
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")
        
        # Validation
        validate(model, val_loader, device)
        
    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_{args.backbone.replace("/", "_")}.pth'))

def validate(model, loader, device):
    model.eval()
    correct_task = 0
    total_task = 0
    correct_state = 0
    total_state = 0
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            task_labels = batch['task_label'].to(device)
            state_labels = batch['state_label'].to(device)
            
            task_logits, state_logits = model(images)
            
            # Task Acc
            _, predicted_task = torch.max(task_logits.data, 1)
            total_task += task_labels.size(0)
            correct_task += (predicted_task == task_labels).sum().item()
            
            # State Acc (filter ignore_index -1)
            mask = state_labels != -1
            if mask.sum() > 0:
                _, predicted_state = torch.max(state_logits.data, 1)
                total_state += mask.sum().item()
                correct_state += (predicted_state[mask] == state_labels[mask]).sum().item()
                
    task_acc = 100 * correct_task / total_task if total_task > 0 else 0
    state_acc = 100 * correct_state / total_state if total_state > 0 else 0
    
    print(f"Validation - Task Acc: {task_acc:.2f}%, State Acc: {state_acc:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='master_dataset.csv', help='Path to master_dataset.csv')
    parser.add_argument('--root', type=str, default='.', help='Root directory containing data/')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='.')
    parser.add_argument('--backbone', type=str, default='vit_small_patch16_dinov3', 
                        help='Backbone model name. Examples: vit_small_patch16_dinov3 (DINOv3), openai/clip-vit-base-patch16 (CLIP)')
    parser.add_argument('--use_masks', action='store_true', help='Use SAM 3 generated masks to black out background')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    args = parser.parse_args()
    
    train(args)
