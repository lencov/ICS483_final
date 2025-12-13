import os
import zipfile

def zip_code(output_filename="code_upload.zip"):
    # Files/Dirs to include (Everything needed to run, minus huge data)
    include_paths = [
        "scripts",
        "sam3",
        "master_dataset.csv", # Essential
        "requirements.txt",
        "README.md",
        # Add other config files if necessary
    ]
    
    print(f"Creating {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in include_paths:
            if not os.path.exists(path):
                print(f"Warning: {path} not found")
                continue

            if os.path.isfile(path):
                print(f"Adding {path}")
                zipf.write(path, path)
            elif os.path.isdir(path):
                print(f"Adding directory {path}")
                for root, dirs, files in os.walk(path):
                    if '__pycache__' in root or '.DS_Store' in root or '.git' in root:
                        continue
                    
                    for file in files:
                        if file == '.DS_Store':
                            continue
                        
                        file_path = os.path.join(root, file)
                        arcname = file_path
                        zipf.write(file_path, arcname)

    print(f"Done! Upload {output_filename} to your Google Drive.")

if __name__ == "__main__":
    zip_code()
