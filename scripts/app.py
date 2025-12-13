import os
import json
import shutil
from flask import Flask, render_template, jsonify, request, send_from_directory, send_file, Response

app = Flask(__name__)

# Configuration
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = os.path.join(BASE_DIR, 'data', 'v2_auto')
STATE_FILE = os.path.join(os.path.dirname(__file__), 'labels_v2.json')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.heic'}

# Categories
CATEGORIES = ['Before', 'After', 'Work_In_Progress', 'Unlabeled', 'Other']

# Ensure directories exist
for cat in CATEGORIES:
    os.makedirs(os.path.join(DATASET_DIR, cat), exist_ok=True)

def get_all_images():
    """
    Returns a list of all images in the dataset, categorized.
    Format: [{'path': rel_path, 'category': category_name}, ...]
    """
    images = []
    for cat in CATEGORIES:
        cat_dir = os.path.join(DATASET_DIR, cat)
        if not os.path.exists(cat_dir):
            continue
            
        for root, dirs, files in os.walk(cat_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in IMAGE_EXTENSIONS:
                    # Store relative path from DATASET_DIR
                    # But wait, frontend needs to know which category it's in to display it?
                    # Actually, let's just return the path relative to DATASET_DIR
                    rel_path = os.path.relpath(os.path.join(root, file), DATASET_DIR)
                    images.append({
                        'path': rel_path,
                        'category': cat,
                        'filename': file
                    })
    
    # Sort by category (Unlabeled first) then filename
    # We want user to focus on Unlabeled first.
    images.sort(key=lambda x: (0 if x['category'] == 'Unlabeled' else 1, x['category'], x['filename']))
    return images

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/images')
def api_images():
    images = get_all_images()
    
    # Statistics
    stats = {cat: 0 for cat in CATEGORIES}
    for img in images:
        stats[img['category']] += 1
        
    return jsonify({
        'images': images,
        'stats': stats
    })

@app.route('/images/<path:filename>')
def serve_image(filename):
    # Filename here is relative to DATASET_DIR
    abs_path = os.path.join(DATASET_DIR, filename)
    
    if not os.path.exists(abs_path):
        return "File not found", 404

    # HEIC Conversion
    if filename.lower().endswith('.heic'):
        try:
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                temp_path = tmp.name
            
            subprocess.run(['sips', '-s', 'format', 'jpeg', abs_path, '--out', temp_path], 
                         check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            with open(temp_path, 'rb') as f:
                data = f.read()
            os.remove(temp_path)
            
            return Response(data, mimetype='image/jpeg')
            
        except Exception as e:
            print(f"Error converting HEIC: {e}")
            return send_from_directory(DATASET_DIR, filename)

    return send_from_directory(DATASET_DIR, filename)

@app.route('/api/label', methods=['POST'])
def api_label():
    data = request.json
    rel_path = data.get('image_path') # e.g., "Unlabeled/image.jpg"
    new_label = data.get('label')     # e.g., "Before"
    
    if not rel_path or new_label not in CATEGORIES:
        return jsonify({'error': 'Invalid data'}), 400

    src_abs = os.path.join(DATASET_DIR, rel_path)
    filename = os.path.basename(rel_path)
    
    # Destination
    dst_abs = os.path.join(DATASET_DIR, new_label, filename)
    
    # If source doesn't exist (maybe already moved?), check if it's already at dest
    if not os.path.exists(src_abs):
        if os.path.exists(dst_abs):
            return jsonify({'status': 'already_moved', 'new_path': os.path.join(new_label, filename)})
        return jsonify({'error': 'Source file not found'}), 404

    try:
        shutil.move(src_abs, dst_abs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'status': 'success',
        'new_path': os.path.join(new_label, filename)
    })

@app.route('/api/undo', methods=['POST'])
def api_undo():
    # Undo in this context means moving back to previous location?
    # Since we don't track history in a file anymore (we rely on FS), 
    # implementing generic undo is harder without a history stack.
    # For now, let's skip undo or implement a simple in-memory stack.
    return jsonify({'error': 'Undo not supported in V2 yet. Just move it back!'}), 400

if __name__ == '__main__':
    print(f"Starting Labeling Tool V2...")
    print(f"Serving from: {DATASET_DIR}")
    app.run(debug=False, port=5001)
