import os
import shutil
import re

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SOURCE_DIR = BASE_DIR  # Scan from current directory
DEST_DIR = os.path.join(BASE_DIR, 'labeled_dataset_v2')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic'}

# Ensure destination directories exist
CATEGORIES = ['Before', 'After', 'Work_In_Progress', 'Unlabeled']
for cat in CATEGORIES:
    os.makedirs(os.path.join(DEST_DIR, cat), exist_ok=True)

def classify_image(path):
    """
    Classifies an image based on its file path.
    Returns: (category, task_name_guess)
    """
    path_lower = path.lower()
    filename = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    
    # Combined string for searching keywords
    # We prioritize the immediate context (filename, parent)
    context = f"{grandparent}/{parent}/{filename}".lower()
    
    # 1. Explicit "Before"
    if 'before' in context:
        return 'Before'
        
    # 2. Explicit "After" or "Done" or "Ok"
    # "ok" is tricky, might be part of a word like "broken". 
    # But based on user data: "ok Phase 2 roof", "ok Vacant unit" -> seems to be a prefix.
    if 'after' in context:
        return 'After'
    if 'done' in context:
        return 'After'
    # Check for "ok " as a whole word or prefix to avoid matching "broken"
    if re.search(r'\bok\b', context): 
        return 'After'
        
    # 3. Work In Progress
    if 'cleaning' in context or 'work' in context or 'progress' in context:
        # If it says "cleaning" but NOT "done" or "after", it might be the act of cleaning.
        # However, "Vacant unit cleaning" is a task name. 
        # Often "Vacant unit cleaning" folder contains "Before" and "After" subfolders.
        # If we are here, we didn't match Before/After/Done yet.
        # So if the folder is just "Vacant unit cleaning" and the file is "image.jpg", 
        # is it before or after? Hard to say. 
        # Let's be conservative: If it's ambiguous, put in Unlabeled or a specific "Ambiguous" pile?
        # The user said: "work being done".
        # Let's map "cleaning" to Unlabeled for now unless we are sure, 
        # OR put it in Work_In_Progress if it explicitly says "during" or similar?
        # Actually, let's stick to the plan: "cleaning", "work", "progress" -> Work_In_Progress
        # But we must be careful. "Vacant unit cleaning" is a noun phrase for the task.
        # Let's put these in 'Unlabeled' to be safe, or maybe a separate 'Potential_Task' folder?
        # Let's stick to 'Unlabeled' for ambiguous task names, 
        # and only use 'Work_In_Progress' for stronger indicators if we find them.
        # For now, I'll return 'Unlabeled' for just "cleaning" to avoid polluting the dataset.
        pass

    return 'Unlabeled'

def main():
    print(f"Scanning {SOURCE_DIR}...")
    count = {c: 0 for c in CATEGORIES}
    
    for root, dirs, files in os.walk(SOURCE_DIR):
        # Skip output directories and tool directories
        if 'labeled_dataset' in root or 'labeling_tool' in root or '.git' in root:
            continue
            
        for file in files:
            if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                src_path = os.path.join(root, file)
                category = classify_image(src_path)
                
                # Create a unique filename to avoid collisions
                # Format: ParentDir_Filename
                parent = os.path.basename(os.path.dirname(src_path))
                new_filename = f"{parent}_{file}"
                
                # If that's still not unique (e.g. multiple "Before" folders), include grandparent
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
