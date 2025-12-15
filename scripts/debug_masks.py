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
        
        # Load exactly as dataset does (image + mask)
        # Note: __getitem__ returns (image, task, state, path)
        # But image is a Tensor if transform is set. 
        # Here we manually call load_image to get PIL Image (before transform)
        
        # Manually replicate load_image logic just to be safe/explicit
        # (Or rely on dataset.load_image if it's cleaner)
        pil_image = dataset.load_image(idx)
        debug_images.append(pil_image)
        
    # Stitch them together
    w, h = debug_images[0].size
    total_w = w * len(debug_images)
    collage = Image.new('RGB', (total_w, h))
    
    for i, img in enumerate(debug_images):
        # Resize to match first if needed (should be same if not resized, but load_image resizes to 640/cache)
        # Wait, if cache_images=False (default above), load_image returns full size?
        # Let's check dataset.py logic. 
        # "if self.cache_images: resize... self.image_cache[idx] = image; return image"
        # "return image" (at end)
        # So if cache_images=False, it returns Full Size image.
        
        # Resize for collage
        img = img.resize((300, 300))
        collage.paste(img, (i * 300, 0))
        
    print(f"Saving collage to {OUTPUT_IMAGE}...")
    collage.save(OUTPUT_IMAGE)
    print("Done! Download this image to see what the model sees.")

if __name__ == "__main__":
    debug_masks()
