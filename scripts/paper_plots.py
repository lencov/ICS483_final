import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dataset import JanitorialDataset
from PIL import Image
import random
import numpy as np

# Config
CSV_FILE = 'master_dataset.csv'
ROOT_DIR = '.'
OUTPUT_DIR = '.'

def plot_class_distribution():
    print("Generating Class Distribution Plot...")
    dataset = JanitorialDataset(CSV_FILE, ROOT_DIR, split='train', use_masks=False)
    
    # 1. State Distribution
    state_counts = {'Active': 0, 'Done': 0}
    active_idx = dataset.STATE_MAP.get('Active', -1) # Likely 0
    done_idx = dataset.STATE_MAP.get('Done', -1)     # Likely 1
    
    for s in dataset.samples:
        if s['state_label'] == active_idx:
            state_counts['Active'] += 1
        elif s['state_label'] == done_idx:
            state_counts['Done'] += 1
            
    # 2. Task Distribution
    # Invert the map: {0: 'General Cleaning'}
    inv_task_map = {v: k for k, v in dataset.TASK_MAP.items()}
    task_counts = {}
    
    for s in dataset.samples:
        t_name = inv_task_map.get(s['task_label'], 'Unknown')
        task_counts[t_name] = task_counts.get(t_name, 0) + 1
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # State Bar
    sns.barplot(x=list(state_counts.keys()), y=list(state_counts.values()), ax=ax1, palette='viridis')
    ax1.set_title('State Label Distribution (Imbalance)')
    ax1.set_ylabel('Count')
    for i, v in enumerate(state_counts.values()):
        ax1.text(i, v + 10, str(v), ha='center')
        
    # Task Bar
    # Sort by count
    sorted_tasks = dict(sorted(task_counts.items(), key=lambda item: item[1], reverse=True))
    sns.barplot(x=list(sorted_tasks.values()), y=list(sorted_tasks.keys()), ax=ax2, palette='magma')
    ax2.set_title('Task Label Distribution')
    ax2.set_xlabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'), dpi=300)
    print(f"Saved class_distribution.png")

def plot_methodology_strip():
    print("Generating Methodology Strip...")
    # Get a good 'Active' image
    dataset = JanitorialDataset(CSV_FILE, ROOT_DIR, split='train', use_masks=True)
    
    # Find active images
    active_indices = [i for i, s in enumerate(dataset.samples) if s['state_label'] == 0]
    if not active_indices:
        print("No active images found.")
        return
        
    # Select 5 random active images with masks
    available_indices = []
    for idx in active_indices:
        sample = dataset.samples[idx]
        base_name, _ = os.path.splitext(sample['path'])
        if os.path.exists(os.path.join(ROOT_DIR, "data", "masks", base_name + ".png")):
            available_indices.append(idx)
            
    if not available_indices:
        print("No active images with masks found.")
        return
        
    num_examples = min(5, len(available_indices))
    selected_indices = random.sample(available_indices, num_examples)
    
    for i, selected_idx in enumerate(selected_indices):
        # Load 3 views
        # 1. Original
        sample = dataset.samples[selected_idx]
        orig_path = os.path.join(ROOT_DIR, sample['path'])
        img_orig = Image.open(orig_path).convert('RGB')
        
        # 2. Mask
        base_name, _ = os.path.splitext(sample['path'])
        mask_path = os.path.join(ROOT_DIR, "data", "masks", base_name + ".png")
        img_mask = Image.open(mask_path).convert('L') # Gray
        
        # 3. Masked Input (What model sees)
        if img_mask.size != img_orig.size:
            img_mask = img_mask.resize(img_orig.size, Image.Resampling.NEAREST)
            
        black_bg = Image.new('RGB', img_orig.size, (0, 0, 0))
        img_final = Image.composite(img_orig, black_bg, img_mask)
        
        # Resize for plot
        H = 400
        W = int(H * img_orig.width / img_orig.height)
        img_orig = img_orig.resize((W, H))
        img_mask = img_mask.resize((W, H))
        img_final = img_final.resize((W, H))
        
        # Combine
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        axes[0].imshow(img_orig)
        axes[0].set_title("Original Input")
        axes[0].axis('off')
        
        axes[1].imshow(img_mask, cmap='gray')
        axes[1].set_title("SAM 3 Mask (ROI)")
        axes[1].axis('off')
        
        axes[2].imshow(img_final)
        axes[2].set_title("Model Input")
        axes[2].axis('off')
        
        plt.tight_layout()
        filename = f'methodology_visualization_{i}.png'
        plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
        print(f"Saved {filename}")
        plt.close(fig)

def plot_raw_distribution():
    print("Generating Raw Distribution Plot...")
    label_counts = {}
    
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Replicate dataset.py logic exactly to get the string
            v2 = row['v2_label']
            v1 = row['v1_label']
            
            state_str = 'Unlabeled'
            if v2 and v2 != 'Unlabeled':
                state_str = v2
            elif v1 and v1 != 'Unlabeled':
                state_str = v1
            
            label_counts[state_str] = label_counts.get(state_str, 0) + 1
            
    # Sort for plotting
    sorted_counts = dict(sorted(label_counts.items(), key=lambda item: item[1], reverse=True))
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(sorted_counts.keys()), y=list(sorted_counts.values()), palette='rocket')
    plt.title('Raw Class Distribution (Pre-Merge)')
    plt.ylabel('Count')
    
    # Add percentages
    total = sum(sorted_counts.values())
    for i, v in enumerate(sorted_counts.values()):
        pct = (v / total) * 100
        plt.text(i, v + 50, f"{v}\n({pct:.1f}%)", ha='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'raw_state_distribution.png'), dpi=300)
    print(f"Saved raw_state_distribution.png")

if __name__ == "__main__":
    plot_class_distribution()
    plot_methodology_strip()
    plot_raw_distribution()
