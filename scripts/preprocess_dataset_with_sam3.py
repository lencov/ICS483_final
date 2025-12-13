import os
import argparse
import torch
import torch.nn.functional as F
import csv
from PIL import Image
from tqdm import tqdm
import json

import pillow_heif
pillow_heif.register_heif_opener()

# Try to import sam3, assuming the environment is set up correctly
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Mapping from our Task Labels to SAM 3 Text Prompts
# ... (omitted for brevity in replacement, but I must match TargetContent precisely so I can't just omit)
# Actually, I'll validly replacing the import block and the loop logic separately.

# First chunk: Imports
import pillow_heif
pillow_heif.register_heif_opener()
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Mapping from our Task Labels to SAM 3 Text Prompts
# tailored to find the 'active' elements of the task
TASK_PROMPTS = {
    'Pressure Washing': ["pressure washer", "water hose", "wet surface", "worker"],
    'Trash Removal': ["trash bag", "garbage bin", "trash can", "debris", "worker"],
    'Roof Cleaning': ["roof", "worker", "ladder", "debris"],
    'Window/Glass': ["window", "squeegee", "worker", "ladder"],
    'Graffiti/Stickers': ["graffiti", "sticker", "spray paint", "wall stain"],
    'Lighting': ["light fixture", "ladder", "worker"],
    'Restroom': ["toilet", "sink", "urinal", "paper towel dispenser", "worker"],
    'Gutters': ["gutter", "leaves", "debris", "ladder"],
    'Floor/Carpet': ["floor buffer", "vacuum", "mop", "wet floor sign"],
    'Vacant Unit': ["empty room", "carpet", "furniture"], # Harder to define 'active' here
    'General Cleaning': ["cleaning cart", "mop", "broom", "worker"]
}

def preprocess_masks(args):
    # Setup Device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.device:
        device = args.device
    print(f"Using device: {device}")

    # Load Model
    print("Loading SAM 3 model...")
    model = build_sam3_image_model()
    model.to(device)
    model.eval()

    # DEBUG: Inspect model for non-CPU tensors
    print("DEBUG: Inspecting model devices...")
    for name, param in model.named_parameters():
        if param.device.type != device:
            print(f"WARNING: Parameter {name} is on {param.device}")
    
    for name, buf in model.named_buffers():
        if buf.device.type != device:
            print(f"WARNING: Buffer {name} is on {buf.device}")

    # Inspect for unregistered tensors in modules
    for name, module in model.named_modules():
        for key, val in module.__dict__.items():
            if isinstance(val, torch.Tensor):
                if val.device.type != device:
                    print(f"WARNING: Unregistered tensor {name}.{key} is on {val.device}")
            elif isinstance(val, (list, tuple)):
                # Check list of tensors
                for i, item in enumerate(val):
                    if isinstance(item, torch.Tensor):
                         if item.device.type != device:
                            print(f"WARNING: Unregistered tensor list item {name}.{key}[{i}] is on {item.device}")

    print("DEBUG: Inspection complete.")

    processor = Sam3Processor(model, device=device)
    print("Model loaded.")

    # Prepare Output Directory
    mask_dir = os.path.join(args.root, "data", "masks")
    os.makedirs(mask_dir, exist_ok=True)

    # Read CSV
    with open(args.csv, 'r') as f:
        reader = list(csv.DictReader(f))

    # Processing Loop
    print(f"Processing {len(reader)} images...")
    
    for row in tqdm(reader):
        image_path = os.path.join(args.root, row['image_path'])
        task_label = row['task_label']
        
        # Determine prompts for this image based on its task
        prompts = TASK_PROMPTS.get(task_label, ["object"]) # Default fallback
        
        # Output mask path
        # Match the directory structure of the original images
        rel_path = row['image_path']
        # Robustly replace extension with .png
        base_name, _ = os.path.splitext(rel_path)
        mask_rel_path = base_name + ".png"
        mask_out_path = os.path.join(mask_dir, mask_rel_path)
        
        # Skip if already exists
        if os.path.exists(mask_out_path) and not args.overwrite:
            continue
            
        os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)

        try:
            image = Image.open(image_path).convert("RGB")
            
            # 1. Set Image
            inference_state = processor.set_image(image)
            
            # 2. Set Text Prompts (Batch processing prompts for one image)
            # We combine all relevant keywords into one list. 
            # SAM 3 usually takes a single string or list of strings.
            # Let's try iterating or combining. 
            # The summary says: "exhaustively segment all instances of an... concept".
            # It implies one concept per prompt.
            # We can prompt multiple concepts and union the masks.
            
            combined_mask = None
            
            for prompt_text in prompts:
                output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
                masks = output["masks"] # [N, H, W]
                # scores = output["scores"]
                
                if masks is not None and len(masks) > 0:
                    # masks might be a list or tensor. Convert safely.
                    if not isinstance(masks, torch.Tensor):
                        masks = torch.tensor(masks)
                    
                    # Ensure masks is on CPU for logic
                    masks = masks.cpu()

                    # masks shape from SAM3 can be [1, 1, H, W] or [1, H, W] or [H, W]
                    # We want to collapse all leading dimensions to get "Any mask at pixel (i, j)"
                    if masks.ndim > 2:
                        # Flatten all init dimensions: (N, H, W) -> (N, H, W)
                        # or (B, N, H, W) -> (B*N, H, W)
                        masks = masks.view(-1, masks.shape[-2], masks.shape[-1])
                    
                    # Calculate transform
                    current_union = torch.any(masks, dim=0) # [H, W]
                    
                    if combined_mask is None:
                        combined_mask = current_union
                    else:
                        combined_mask = torch.logical_or(combined_mask, current_union)
            
            # Save Mask
            if combined_mask is not None:
                mask_img = Image.fromarray(combined_mask.cpu().numpy().astype('uint8') * 255)
                mask_img.save(mask_out_path)
            else:
                # Save empty black mask
                Image.new('L', image.size, 0).save(mask_out_path)

        except RuntimeError as e:
            print(f"RuntimeError processing {image_path}: {e}")
            # PIL Image objects do not have a .device attribute.
            # print(f"Image device: {image.device}") 
            print(f"Model backbone device: {processor.model.backbone.vision_backbone.trunk.pos_embed.device}")
            continue
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue

    print("Preprocessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='master_dataset.csv', help='Path to master_dataset.csv')
    parser.add_argument('--root', type=str, default='.', help='Root directory containing data/')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    
    preprocess_masks(args)
