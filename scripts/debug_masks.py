import os
import csv
import random
from PIL import Image
import torch
from dataset import JanitorialDataset

# Quick config
CSV_FILE = 'master_dataset.csv'
ROOT_DIR = '.'
OUTPUT_IMAGE = 'debug_masked_samples.jpg'

def debug_masks():
    print("Loading dataset in Debug Mode...")
    # Use the same class logic as training
    dataset = JanitorialDataset(CSV_FILE, ROOT_DIR, split='train', use_masks=True)
    
    # Filter for "Active" (0) images only, as these are the hardest
    active_indices = [i for i, s in enumerate(dataset.samples) if s['state_label'] == 0]
    
    if not active_indices:
        print("No active images found?")
        return

    # Pick 5 random
    indices = random.sample(active_indices, min(5, len(active_indices)))
    
    debug_images = []
    print(f"Sampling {len(indices)} Active images...")
    
    for idx in indices:
        sample = dataset.samples[idx]
        path = sample['path']
        print(f"Processing: {path}")
        
        # Check if mask exists manually to report to user
        base_name, _ = os.path.splitext(sample['path'])
        mask_path = os.path.join(dataset.root_dir, "data", "masks", base_name + ".png")
        if os.path.exists(mask_path):
            print(f"  [OK] Mask found at: {mask_path}")
            # Load stats
            try:
                m = Image.open(mask_path).convert('L')
                import numpy as np
                m_arr = np.array(m)
                coverage = np.mean(m_arr > 128) * 100
                print(f"  [STATS] Mask Coverage: {coverage:.2f}% of image is kept.")
            except:
                pass
        else:
            print(f"  [ERROR] Mask NOT found at: {mask_path}")

        # Manually replicate load_image logic just to be safe/explicit
        # (Or rely on dataset.load_image if it's cleaner)
        pil_image = dataset.load_image(idx)
        debug_images.append(pil_image)
        
    # Stitch them together
    # Use fixed 300x300 for collage tiles
    tile_size = 300
    total_w = tile_size * len(debug_images)
    collage = Image.new('RGB', (total_w, tile_size))
    
    for i, img in enumerate(debug_images):
        img = img.resize((tile_size, tile_size))
        collage.paste(img, (i * tile_size, 0))
        
    print(f"Saving collage to {OUTPUT_IMAGE}...")
    collage.save(OUTPUT_IMAGE)
    print("Done! Download this image to see what the model sees.")

if __name__ == "__main__":
    debug_masks()
