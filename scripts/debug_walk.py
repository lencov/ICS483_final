import os
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic'}

start = time.time()
images = []
for root, dirs, files in os.walk(BASE_DIR):
    if 'labeled_dataset' in root or 'labeling_tool' in root:
        continue
    for file in files:
        if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
            images.append(os.path.join(root, file))

print(f"Found {len(images)} images in {time.time() - start:.2f} seconds")
