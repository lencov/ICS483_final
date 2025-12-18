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

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from dataset import JanitorialDataset

import timm
from transformers import CLIPModel, CLIPProcessor

class MultiHeadProbe(nn.Module):
    def __init__(self, backbone_name, num_tasks, num_states, device):
        super().__init__()
        self.backbone_name = backbone_name
        self.device = device
        
        if 'dino' in backbone_name:
            print(f"Loading DINO model: {backbone_name}")
            self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
            self.embed_dim = self.backbone.num_features
            self.processor = None
            
        elif 'clip' in backbone_name:
            print(f"Loading CLIP model: {backbone_name}")
            self.backbone = CLIPModel.from_pretrained(backbone_name).vision_model
            self.embed_dim = self.backbone.config.hidden_size
            self.processor = CLIPProcessor.from_pretrained(backbone_name)
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
            
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.task_head = nn.Linear(self.embed_dim, num_tasks)
        
        self.state_head = nn.Linear(self.embed_dim, num_states)
        
    def forward(self, x):
        if 'dino' in self.backbone_name:
            features = self.backbone(x)
        elif 'clip' in self.backbone_name:
            outputs = self.backbone(pixel_values=x)
            features = outputs.pooler_output
            
        task_logits = self.task_head(features)
        state_logits = self.state_head(features)
        
        return task_logits, state_logits

def get_transforms(backbone_name):
    if 'dino' in backbone_name:
        size = 224
        if 'dinov2' in backbone_name:
            size = 518
        elif 'patch16' in backbone_name:
            size = 224
            
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif 'clip' in backbone_name:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    return None

def train(args, train_dataset=None, val_dataset=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using Masks: {args.use_masks}")
    
    transform = get_transforms(args.backbone)
    
    if train_dataset is None:
        train_dataset = JanitorialDataset(args.csv, args.root, split='train', transform=transform, use_masks=args.use_masks, cache_images=args.cache_images, limit=args.limit)
    if val_dataset is None:
        val_dataset = JanitorialDataset(args.csv, args.root, split='val', transform=transform, use_masks=args.use_masks, cache_images=args.cache_images, limit=args.limit)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    num_states = 2
    model = MultiHeadProbe(args.backbone, num_tasks=11, num_states=num_states, device=device).to(device)
    
    state_counts = torch.zeros(num_states)
    for s in train_dataset.samples:
        lbl = s['state_label']
        if lbl != -1:
            state_counts[lbl] += 1
            
    print(f"Class Distribution: Active={int(state_counts[0])}, Done={int(state_counts[1])}")
    
    max_count = torch.max(state_counts)
    weights = max_count / (state_counts + 1e-6)
    weights = weights.to(device)
    print(f"Using Weights: {weights}")

    criterion_task = nn.CrossEntropyLoss()
    criterion_state = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)
    
    optimizer = optim.Adam(list(model.task_head.parameters()) + list(model.state_head.parameters()), lr=args.lr)
    
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
        
        validate(model, val_loader, device)
        
    model_filename = f'model_{args.backbone.replace("/", "_")}.pth'
    if hasattr(args, 'run_name') and args.run_name:
        model_filename = f'{args.run_name}.pth'
        
    torch.save(model.state_dict(), os.path.join(args.output_dir, model_filename))
    print(f"Saved model to {os.path.join(args.output_dir, model_filename)}")

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
            
            _, predicted_task = torch.max(task_logits.data, 1)
            total_task += task_labels.size(0)
            correct_task += (predicted_task == task_labels).sum().item()
            
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
    parser.add_argument('--cache_images', action='store_true', help='Cache all images to RAM for speed')
    parser.add_argument('--limit', type=int, default=None, help='Limit dataset size for debugging (e.g. 100)')
    args = parser.parse_args()
    
    train(args)
