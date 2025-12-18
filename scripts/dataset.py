import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()

class JanitorialDataset(Dataset):
    """
    Dataset for Janitorial Task and State Recognition.
    
    Task Labels (Multi-class):
        0: General Cleaning
        1: Pressure Washing
        2: Trash Removal
        3: Roof Cleaning
        4: Window/Glass
        5: Graffiti/Stickers
        6: Lighting
        7: Restroom
        8: Gutters
        9: Floor/Carpet
        10: Vacant Unit
        
    State Labels (Binary):
        0: Active (Before, Work_In_Progress)
        1: Done (After)
        -1: Ignore (Other, Unlabeled)
    """
    
    TASK_MAP = {
        'General Cleaning': 0,
        'Pressure Washing': 1,
        'Trash Removal': 2,
        'Roof Cleaning': 3,
        'Window/Glass': 4,
        'Graffiti/Stickers': 5,
        'Lighting': 6,
        'Restroom': 7,
        'Gutters': 8,
        'Floor/Carpet': 9,
        'Vacant Unit': 10
    }
    
    STATE_MAP = {
        'Before': 0,
        'Work_In_Progress': 0,
        'After': 1,
        'Other': -1,
        'Unlabeled': -1
    }

    def __init__(self, csv_file, root_dir, split='train', transform=None, use_masks=False, cache_images=False, limit=None):
        self.root_dir = root_dir
        self.transform = transform
        self.use_masks = use_masks
        self.cache_images = cache_images
        self.samples = []
        self.image_cache = {} # Index -> PIL Image
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Filter by split
                if row['split'] != split:
                    continue
                
                image_path = row['image_path']
                task_str = row['task_label']
                
                # Determine State Label
                v2 = row['v2_label']
                v1 = row['v1_label']
                
                state_str = 'Unlabeled'
                if v2 and v2 != 'Unlabeled':
                    state_str = v2
                elif v1 and v1 != 'Unlabeled':
                    state_str = v1
                
                task_idx = self.TASK_MAP.get(task_str, 0)
                state_idx = self.STATE_MAP.get(state_str, -1)
                
                self.samples.append({
                    'path': image_path,
                    'task_label': task_idx,
                    'state_label': state_idx,
                    'site_id': row['site_id']
                })
        
        # Limit dataset for debugging
        if limit is not None:
            print(f"Limiting dataset to {limit} samples (Debugging)")
            self.samples = self.samples[:limit]

        # Pre-cache images if requested
        if self.cache_images:
            print(f"Caching {len(self.samples)} images to RAM... (This make take a moment)")
            from tqdm import tqdm
            for idx in tqdm(range(len(self.samples)), desc="Caching"):
                self.load_image(idx)
                
    def __len__(self):
        return len(self.samples)

    def load_image(self, idx):
        """Helper to load image from disk (or cache if implemented later logic, but here we utilize it for pre-loading)"""
        # If already in cache (and we are using cache), return it
        if self.cache_images and idx in self.image_cache:
            return self.image_cache[idx]
            
        sample = self.samples[idx]
        path = os.path.join(self.root_dir, sample['path'])
        
        try:
            image = Image.open(path).convert('RGB')
            
            if self.use_masks:
                base_name, _ = os.path.splitext(sample['path'])
                mask_path_1 = os.path.join(self.root_dir, "data", "masks", base_name + ".png")
                mask_path_2 = os.path.join(self.root_dir, "masks", base_name + ".png")
                
                mask_path = mask_path_1 if os.path.exists(mask_path_1) else mask_path_2
                
                if os.path.exists(mask_path):
                    try:
                        mask = Image.open(mask_path).convert('L')
                        if mask.size != image.size:
                            mask = mask.resize(image.size, Image.NEAREST)
                        
                        black_bg = Image.new('RGB', image.size, (0, 0, 0))
                        image = Image.composite(image, black_bg, mask)
                        
                    except Exception as e:
                        print(f"Error loading mask {mask_path}: {e}")
            
            if self.cache_images:
                # OPTIMIZATION: Resize to a "Safe Max" before caching.
                # Raw 12MP images are ~36MB uncompressed in RAM.
                # Resizing to 640px makes them ~1MB.
                # We need 224px or 518px for the model, so 640 is plenty.
                image.thumbnail((640, 640), Image.Resampling.LANCZOS)
                self.image_cache[idx] = image
                
            return image
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return Image.new('RGB', (224, 224))
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load Image (From Cache or Disk)
        image = self.load_image(idx)
            
        if self.transform:
            image = self.transform(image)
            

            
        return {
            'image': image,
            'task_label': torch.tensor(sample['task_label'], dtype=torch.long),
            'state_label': torch.tensor(sample['state_label'], dtype=torch.long), # -1 for ignore
            'path': sample['path']
        }
