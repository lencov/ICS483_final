let images = [];
let currentIndex = 0;

document.addEventListener('DOMContentLoaded', () => {
    fetchImages();
    setupEventListeners();
});

async function fetchImages() {
    try {
        const response = await fetch('/api/images');
        const data = await response.json();
        images = data.images;
        updateStats(data.stats);

        // Find first Unlabeled image
        currentIndex = images.findIndex(img => img.category === 'Unlabeled');
        if (currentIndex === -1) currentIndex = 0;

        updateUI();
    } catch (error) {
        console.error('Error fetching images:', error);
        alert('Failed to load images.');
    }
}

function updateStats(stats) {
    const container = document.getElementById('stats-display');
    container.innerHTML = Object.entries(stats)
        .map(([key, val]) => `<span class="stat-item">${key}: ${val}</span>`)
        .join(' | ');
}

function updateUI() {
    if (images.length === 0) return;

    const imgData = images[currentIndex];
    const imgElement = document.getElementById('current-image');
    const overlay = document.getElementById('filename-overlay');
    const badge = document.getElementById('category-badge');
    const counter = document.getElementById('counter');
    const currentCat = document.getElementById('current-category');

    imgElement.src = `/images/${imgData.path}`;
    overlay.textContent = imgData.filename;

    badge.textContent = imgData.category;
    badge.className = `category-badge badge-${imgData.category}`;

    currentCat.textContent = imgData.category;
    counter.textContent = `${currentIndex + 1} / ${images.length}`;
    document.getElementById('jump-input').value = currentIndex + 1;
}

async function labelImage(newLabel) {
    const imgData = images[currentIndex];

    // Optimistic UI update
    const oldCategory = imgData.category;
    imgData.category = newLabel;

    try {
        const response = await fetch('/api/label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_path: imgData.path, label: newLabel })
        });

        const data = await response.json();

        if (data.status === 'success' || data.status === 'already_moved') {
            // Update path in our local model since it moved on server
            imgData.path = data.new_path;

            // Move to next image
            if (currentIndex < images.length - 1) {
                currentIndex++;
            }
            updateUI();

            // Refresh stats in background
            fetch('/api/images').then(r => r.json()).then(d => updateStats(d.stats));
        } else {
            alert('Error: ' + data.error);
            imgData.category = oldCategory; // Revert
        }
    } catch (error) {
        console.error('Error:', error);
        imgData.category = oldCategory;
    }
}

function setupEventListeners() {
    document.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT') return;

        switch (e.key) {
            case '1': labelImage('Before'); break;
            case '2': labelImage('After'); break;
            case '3': labelImage('Work_In_Progress'); break;
            case '4': labelImage('Other'); break;
            case 'ArrowRight':
                if (currentIndex < images.length - 1) { currentIndex++; updateUI(); }
                break;
            case 'ArrowLeft':
                if (currentIndex > 0) { currentIndex--; updateUI(); }
                break;
        }
    });

    document.getElementById('jump-input').addEventListener('change', (e) => {
        let val = parseInt(e.target.value);
        if (val >= 1 && val <= images.length) {
            currentIndex = val - 1;
            updateUI();
        }
    });
}
