import os
import csv
import re
import random

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ORIGINAL_DIR = os.path.join(DATA_DIR, 'original')
LABELED_V1_DIR = os.path.join(DATA_DIR, 'v1_manual')
LABELED_V2_DIR = os.path.join(DATA_DIR, 'v2_auto')
OUTPUT_FILE = os.path.join(BASE_DIR, 'master_dataset.csv')

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic'}

# Site Split Configuration
# We hold out specific sites for testing/validation to ensure generalization
TEST_SITES = {'KOKO_MARINA', 'Safeway'} 
VAL_SITES = {'Manoa', 'Piikoi'}
# All other sites (WAIPIO, CENTER_OF_WAIKIKI, 677_ALA_MOANA, etc.) are TRAIN

def get_site_id(path):
    """Extracts site ID from the file path (first folder after original/)."""
    parts = path.split(os.sep)
    # path is relative to BASE_DIR, e.g. data/original/WAIPIO/...
    # parts: ['data', 'original', 'WAIPIO', ...]
    if len(parts) > 2 and parts[1] == 'original':
        return parts[2]
    return 'Unknown'

def get_task_label(path):
    """Extracts task label from file path."""
    path_lower = path.lower()
    
    if 'pressure' in path_lower: return 'Pressure Washing'
    if 'roof' in path_lower: return 'Roof Cleaning'
    if 'restroom' in path_lower or 'bathroom' in path_lower or 'toilet' in path_lower: return 'Restroom'
    if 'trash' in path_lower or 'rubbish' in path_lower or 'garbage' in path_lower: return 'Trash Removal'
    if 'glass' in path_lower or 'window' in path_lower: return 'Window/Glass'
    if 'sticker' in path_lower or 'graffiti' in path_lower: return 'Graffiti/Stickers'
    if 'light' in path_lower or 'lamp' in path_lower: return 'Lighting'
    if 'gutter' in path_lower: return 'Gutters'
    if 'floor' in path_lower or 'carpet' in path_lower or 'vacuum' in path_lower or 'mop' in path_lower: return 'Floor/Carpet'
    if 'vacant' in path_lower: return 'Vacant Unit'
    
    return 'General Cleaning'

def build_label_map(dataset_dir):
    """
    Builds a map of {filename: label} from a labeled dataset directory.
    Note: The labeled datasets use 'Parent_Filename' as the filename.
    """
    label_map = {}
    if not os.path.exists(dataset_dir):
        return label_map
        
    for label in os.listdir(dataset_dir):
        label_dir = os.path.join(dataset_dir, label)
        if not os.path.isdir(label_dir):
            continue
            
        for file in os.listdir(label_dir):
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                label_map[file] = label
    return label_map

def main():
    print("Building label maps...")
    v1_map = build_label_map(LABELED_V1_DIR)
    v2_map = build_label_map(LABELED_V2_DIR)
    
    print(f"Loaded {len(v1_map)} V1 labels and {len(v2_map)} V2 labels.")
    
    rows = []
    print("Scanning original images...")
    
    for root, dirs, files in os.walk(ORIGINAL_DIR):
        # Skip output directories and tool directories
        if 'labeled_dataset' in root or 'labeling_tool' in root or '.git' in root:
            continue
            
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                abs_path = os.path.join(root, file)
                # Rel path should be relative to BASE_DIR (project root) so it starts with data/original/...
                rel_path = os.path.relpath(abs_path, BASE_DIR)
                
                # Construct the key used in labeled datasets: Parent_Filename
                parent = os.path.basename(os.path.dirname(abs_path))
                key = f"{parent}_{file}"
                
                # Check for Grandparent_Parent_Filename (used by organize_dataset.py for collisions)
                grandparent = os.path.basename(os.path.dirname(os.path.dirname(abs_path)))
                key_long = f"{grandparent}_{parent}_{file}"
                
                # Lookups
                v1_label = v1_map.get(key, 'Unlabeled')
                
                # V2 lookup: try short key, then long key
                v2_label = v2_map.get(key)
                if not v2_label:
                    v2_label = v2_map.get(key_long, 'Unlabeled')
                
                task_label = get_task_label(rel_path)
                site_id = get_site_id(rel_path)
                
                # Split Logic
                if site_id in TEST_SITES:
                    split = 'test'
                elif site_id in VAL_SITES:
                    split = 'val'
                else:
                    split = 'train'
                
                # Validity Logic
                # Valid if it has a meaningful label in V1 OR V2
                # And is not 'Other' or 'Unlabeled' (unless we want to keep them for now)
                # For now, let's mark valid if it's not 'Other' in both
                is_valid = True
                if v1_label == 'Other' and v2_label == 'Other':
                    is_valid = False
                if v1_label == 'Unlabeled' and v2_label == 'Unlabeled':
                    is_valid = False # Or maybe True if we want to predict on them? Let's say False for training.
                
                rows.append({
                    'image_path': rel_path,
                    'site_id': site_id,
                    'task_label': task_label,
                    'v1_label': v1_label,
                    'v2_label': v2_label,
                    'split': split,
                    'is_valid': is_valid
                })

    print(f"Found {len(rows)} images.")
    
    # Write to CSV
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', newline='') as f:
        fieldnames = ['image_path', 'site_id', 'task_label', 'v1_label', 'v2_label', 'split', 'is_valid']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print("Done!")

if __name__ == '__main__':
    main()
