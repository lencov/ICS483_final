import os
import zipfile
from tqdm import tqdm

def zip_project(output_filename="project_upload.zip"):
    # files and folders to include
    include_paths = [
        "data",
        "scripts",
        "sam3",
        "master_dataset.csv",
        "requirements.txt",
    ]
    
    # Validation
    for p in include_paths:
        if not os.path.exists(p):
            print(f"Warning: {p} not found, skipping.")

    print(f"Creating {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in include_paths:
            if os.path.isfile(path):
                print(f"Adding {path}")
                zipf.write(path, path)
            elif os.path.isdir(path):
                print(f"Adding directory {path}")
                for root, dirs, files in os.walk(path):
                    # Skip common trash
                    if '__pycache__' in root or '.DS_Store' in root:
                        continue
                    
                    for file in files:
                        if file == '.DS_Store':
                            continue
                        
                        file_path = os.path.join(root, file)
                        # Archive name should be relative to project root
                        arcname = file_path
                        zipf.write(file_path, arcname)

    print(f"Done! Upload {output_filename} to your Google Drive.")

if __name__ == "__main__":
    zip_project()
