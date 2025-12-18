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

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import pillow_heif
pillow_heif.register_heif_opener()
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

TASK_PROMPTS = {
    'Pressure Washing': ["pressure washer", "water hose", "wet surface", "worker", "spray gun", "concrete stain", "mold", "wall", "dirt", "grime", "gum", "tire marks", "bird droppings"],
    'Trash Removal': ["trash bag", "garbage bin", "trash can", "debris", "worker", "dumpster", "cardboard box", "litter", "overflowing bin", "liquid spill", "stain"],
    'Roof Cleaning': ["roof", "worker", "ladder", "debris", "moss", "leaves", "safety harness", "rope", "dirt", "stain", "bird droppings"],
    'Window/Glass': ["window", "squeegee", "worker", "ladder", "glass", "bucket", "water streak", "scaffold", "dirt", "fingerprints", "bird droppings", "smudge"],
    'Graffiti/Stickers': ["graffiti", "sticker", "spray paint", "wall stain", "tag", "poster", "adhesive residue", "utility pole", "electrical box", "glue", "scratch"],
    'Lighting': ["light fixture", "ladder", "worker", "light bulb", "broken glass", "lamp post", "ceiling light", "cobweb", "dust", "dirt", "bird droppings"],
    'Restroom': ["toilet", "sink", "urinal", "paper towel dispenser", "worker", "mirror", "stall door", "soap dispenser", "wet floor sign", "trash bin", "mess", "paper on floor", "water spill", "stain"],
    'Gutters': ["gutter", "leaves", "debris", "ladder", "downspout", "roof edge", "pine needles", "sludge", "dirt", "overflow", "bird droppings"],
    'Floor/Carpet': ["floor buffer", "vacuum", "mop", "wet floor sign", "stain", "dirt", "scuff mark", "carpet cleaner", "dust", "spill", "footprints"],
    'Vacant Unit': ["empty room", "carpet", "furniture", "trash", "debris", "wall damage", "paint bucket", "blinds", "dust", "cobweb", "bird droppings"], 
    'General Cleaning': ["cleaning cart", "mop", "broom", "worker", "leaves", "debris", "trash", "dirt", "dustpan", "leaf blower", "rake", "hose", "dust", "spill", "bird droppings"]
}

def preprocess_masks(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if args.device:
        device = args.device
    print(f"Using device: {device}")

    print("Loading SAM 3 model...")
    model = build_sam3_image_model()
    model.to(device)
    model.eval()

    print("DEBUG: Inspecting model devices...")
    for name, param in model.named_parameters():
        if param.device.type != device:
            print(f"WARNING: Parameter {name} is on {param.device}")
    
    for name, buf in model.named_buffers():
        if buf.device.type != device:
            print(f"WARNING: Buffer {name} is on {buf.device}")

    for name, module in model.named_modules():
        for key, val in module.__dict__.items():
            if isinstance(val, torch.Tensor):
                if val.device.type != device:
                    print(f"WARNING: Unregistered tensor {name}.{key} is on {val.device}")
            elif isinstance(val, (list, tuple)):
                for i, item in enumerate(val):
                    if isinstance(item, torch.Tensor):
                         if item.device.type != device:
                            print(f"WARNING: Unregistered tensor list item {name}.{key}[{i}] is on {item.device}")

    print("DEBUG: Inspection complete.")

    processor = Sam3Processor(model, device=device)
    print("Model loaded.")

    mask_dir = os.path.join(args.root, "data", "masks")
    os.makedirs(mask_dir, exist_ok=True)

    with open(args.csv, 'r') as f:
        reader = list(csv.DictReader(f))

    print(f"Processing {len(reader)} images...")
    
    for row in tqdm(reader):
        image_path = os.path.join(args.root, row['image_path'])
        task_label = row['task_label']
        
        prompts = TASK_PROMPTS.get(task_label, ["object"]).copy()
        
        norm_path = os.path.normpath(image_path)
        path_parts = norm_path.split(os.sep)
        
        IGNORED_FOLDERS = {
            'data', 'original', 'images', 'img', 'dcim', '100apple', 
            'log', 'logs', 'march', 'april', 'may', 'june', 'july', 'august', 
            'september', 'october', 'november', 'december', 'january', 'february',
            'waipio', 'shopping', 'center', '.', '..', 'ok', 'bad'
        }
        
        for part in path_parts:
            clean_part = part.lower()
            
            if clean_part == os.path.basename(image_path).lower():
                continue
                
            for ignore in ['ok ', 'bad ', 'ok_', 'bad_', '_']:
                clean_part = clean_part.replace(ignore, ' ')
            
            clean_part = ''.join([c for c in clean_part if not c.isdigit()])
            clean_part = clean_part.strip()
            
            words = clean_part.split()
            
            if len(clean_part) > 2 and clean_part not in IGNORED_FOLDERS:
                prompts.append(clean_part)
                
            for w in words:
                if len(w) > 3 and w not in IGNORED_FOLDERS:
                     prompts.append(w)
                     
        prompts = list(set(prompts))
        
        rel_path = row['image_path']
        base_name, _ = os.path.splitext(rel_path)
        mask_rel_path = base_name + ".png"
        mask_out_path = os.path.join(mask_dir, mask_rel_path)
        
        if os.path.exists(mask_out_path) and not args.overwrite:
            continue
            
        os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)

        try:
            image = Image.open(image_path).convert("RGB")
            
            inference_state = processor.set_image(image)
            
            combined_mask = None
            
            for prompt_text in prompts:
                output = processor.set_text_prompt(state=inference_state, prompt=prompt_text)
                masks = output["masks"] # [N, H, W]
                # scores = output["scores"]
                
                if masks is not None and len(masks) > 0:
                    if not isinstance(masks, torch.Tensor):
                        masks = torch.tensor(masks)
                    
                    masks = masks.cpu()

                    if masks.ndim > 2:
                        masks = masks.view(-1, masks.shape[-2], masks.shape[-1])
                    
                    current_union = torch.any(masks, dim=0) # [H, W]
                    
                    if combined_mask is None:
                        combined_mask = current_union
                    else:
                        combined_mask = torch.logical_or(combined_mask, current_union)
            
            if combined_mask is not None:
                mask_img = Image.fromarray(combined_mask.cpu().numpy().astype('uint8') * 255)
                mask_img.save(mask_out_path)
            else:
                Image.new('L', image.size, 0).save(mask_out_path)

        except RuntimeError as e:
            print(f"RuntimeError processing {image_path}: {e}")
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
