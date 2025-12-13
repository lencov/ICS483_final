import os
import zipfile

def zip_images(output_filename="images_upload.zip"):
    # Target directory
    source_dir = "data/original"
    
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} not found.")
        return

    print(f"Creating {output_filename} from {source_dir}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the directory
        for root, dirs, files in os.walk(source_dir):
            if '__pycache__' in root or '.DS_Store' in root:
                continue
            
            for file in files:
                if file == '.DS_Store':
                    continue
                
                file_path = os.path.join(root, file)
                # Keep the structure "data/original/..." in the zip
                # ensuring when unpacked it merges nicely if root is same
                arcname = file_path
                print(f"Adding {arcname}")
                zipf.write(file_path, arcname)

    print(f"Done! Upload {output_filename} to your Google Drive.")

if __name__ == "__main__":
    zip_images()
