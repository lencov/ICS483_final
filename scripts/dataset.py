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

    def __init__(self, csv_file, root_dir, split='train', transform=None, use_masks=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_masks = use_masks
        self.samples = []
        
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
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = os.path.join(self.root_dir, sample['path'])
        
        try:
            image = Image.open(path).convert('RGB')
            
            if self.use_masks:
                base_name, _ = os.path.splitext(sample['path'])
                mask_path = os.path.join(self.root_dir, "data", "masks", base_name + ".png")
                
                if os.path.exists(mask_path):
                    try:
                        # Load mask (L mode = 8-bit pixels, black and white)
                        mask = Image.open(mask_path).convert('L')
                        # Resize if dimensions differ (SAM 3 might have resized or original image handling issues)
                        if mask.size != image.size:
                            mask = mask.resize(image.size, Image.NEAREST)
                            
                        # Apply mask: Keep image where mask > 0, else Black
                        # Composite syntax: composite(image1, image2, mask). 
                        # mask uses 0 as transparent (show image2) and 255 as opaque (show image1). 
                        # Wait, Image.composite(image, background, mask):
                        # "Interpolates between image and background using mask as alpha."
                        # If mask is 255 (white): shows image.
                        # If mask is 0 (black): shows background.
                        # SAM 3 mask is likely 1 where object is?
                        # In preprocessing script: 
                        # combined_mask = torch.any(masks, dim=0) -> True/False
                        # Image.fromarray(combined_mask * 255) -> 255 (White) for Object, 0 (Black) for Background.
                        # So Mask=255 means "This is the object".
                        
                        # So composite(image, black, mask) -> Shows image where mask=255 (Object), shows Black where mask=0.
                        # This is exactly what we want: Black out the background.
                        
                        black_bg = Image.new('RGB', image.size, (0, 0, 0))
                        image = Image.composite(image, black_bg, mask)
                        
                    except Exception as e:
                        print(f"Error loading mask {mask_path}: {e}")
                
        except Exception as e:
            print(f"Error loading {path}: {e}")
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'task_label': torch.tensor(sample['task_label'], dtype=torch.long),
            'state_label': torch.tensor(sample['state_label'], dtype=torch.long), # -1 for ignore
            'path': sample['path']
        }
