import os
import shutil
import re

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SOURCE_DIR = BASE_DIR 
DEST_DIR = os.path.join(BASE_DIR, 'labeled_dataset_v2')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic'}

CATEGORIES = ['Before', 'After', 'Work_In_Progress', 'Unlabeled']
for cat in CATEGORIES:
    os.makedirs(os.path.join(DEST_DIR, cat), exist_ok=True)

def classify_image(path):
    path_lower = path.lower()
    filename = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    
    context = f"{grandparent}/{parent}/{filename}".lower()
    
    if 'before' in context:
        return 'Before'
        
    if 'after' in context:
        return 'After'
    if 'done' in context:
        return 'After'
    if re.search(r'\bok\b', context): 
        return 'After'
        
    if 'cleaning' in context or 'work' in context or 'progress' in context:
        pass

    return 'Unlabeled'

def main():
    print(f"Scanning {SOURCE_DIR}...")
    count = {c: 0 for c in CATEGORIES}
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        if 'labeled_dataset' in root or 'labeling_tool' in root or '.git' in root:
            continue
            
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                src_path = os.path.join(root, file)
                category = classify_image(src_path)
                
                parent = os.path.basename(os.path.dirname(src_path))
                new_filename = f"{parent}_{file}"
                
                dest_path = os.path.join(DEST_DIR, category, new_filename)
                if os.path.exists(dest_path):
                    grandparent = os.path.basename(os.path.dirname(os.path.dirname(src_path)))
                    new_filename = f"{grandparent}_{parent}_{file}"
                    dest_path = os.path.join(DEST_DIR, category, new_filename)
                
                shutil.copy2(src_path, dest_path)
                count[category] += 1
                
                if count[category] % 100 == 0:
                    print(f"Processed {sum(count.values())} images...", end='\r')

    print("\n\nProcessing Complete!")
    print("-------------------")
    for cat, n in count.items():
        print(f"{cat}: {n}")

if __name__ == '__main__':
    main()
