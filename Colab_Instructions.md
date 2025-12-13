# Running SAM 3 Masking on Google Colab (Split Workflow)

We will upload the **Code** and **Images** separately to handle the large dataset size efficiently.

## Step 1: Prepare Two Zip Files
Run these commands in your local terminal:

1. **Pack Images** (This might take a while, it zips `data/original`):
   ```bash
   python3 scripts/pack_images.py
   ```
   *Output: `images_upload.zip`*

2. **Pack Code** (Zips scripts, sam3, csv, etc.):
   ```bash
   python3 scripts/pack_code.py
   ```
   *Output: `code_upload.zip`*

   *(Note: You can also git pull the code if you set up a repo, but the zip is faster for a one-off run with no setup)*

## Step 2: Google Drive
1. Upload both `images_upload.zip` and `code_upload.zip` to the **root** of your Google Drive.

## Step 3: Colab Setup
1. Open [Google Colab](https://colab.research.google.com/).
2. Create or Open a Notebook.
3. **Runtime > Change runtime type > T4 GPU** (Required).
4. Run the cells below:

### Cell 1: Mount Drive & Setup
```python
from google.colab import drive
import os
import shutil

drive.mount('/content/drive')

# Working directory
WORK_DIR = "/content/ICS483_FINAL"
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)
%cd {WORK_DIR}

# 1. Bring in Code (Option A: Git - Recommended)
# Clone your repo (make sure it's public or you have a token)
!git clone https://github.com/lencov/ICS483_final . || git pull origin main

# 1. Bring in Code (Option B: Zip - No Setup)
if not os.path.exists("master_dataset.csv") and not os.path.isdir(".git"): 
    print("Copying Code Zip...")
    shutil.copy("/content/drive/MyDrive/code_upload.zip", "code_upload.zip")
    print("Unzipping Code...")
    shutil.unpack_archive("code_upload.zip", ".")

# 2. Bring in Images (Only if needed, this is large)
if not os.path.exists("data/original"):
    print("Copying Images (This may take time)...")
    shutil.copy("/content/drive/MyDrive/images_upload.zip", "images_upload.zip")
    print("Unzipping Images...")
    shutil.unpack_archive("images_upload.zip", ".")
```

### Cell 2: Install Dependencies
```python
# Install SAM 3 and helpers
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q pillow-heif matplotlib pandas flash-attn --no-build-isolation
!pip install -e sam3
```

### Cell 3: Run Mask Generation
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Run preprocessing
!python scripts/preprocess_dataset_with_sam3.py --csv master_dataset.csv --root . --device cuda
```

### Cell 4: Save Options
```python
# Option A: Interactive Download (Small files)
# from google.colab import files
# files.download('data/masks/some_mask.png')

# Option B: Bulk Save to Drive (Recommended)
print("Zipping output masks...")
shutil.make_archive("masks_output", 'zip', "data/masks")

print("Saving to Drive...")
shutil.copy("masks_output.zip", "/content/drive/MyDrive/masks_output.zip")
print("Done! Download masks_output.zip from your Drive.")
```

## Step 4: Local Integration
1. Download `masks_output.zip` from Drive.
2. Unzip into your project so you have `data/masks/`.
