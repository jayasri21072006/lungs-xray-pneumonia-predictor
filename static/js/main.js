document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadCard = document.getElementById('upload-card');
    const resultCard = document.getElementById('result-card');
    const imagePreview = document.getElementById('image-preview');
    const diagnosisStatus = document.getElementById('diagnosis-status');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const loading = document.getElementById('loading');
    const resetBtn = document.getElementById('reset-btn');

    let selectedFile = null;

    // Trigger file input
    dropZone.addEventListener('click', () => fileInput.click());

    // Handle drag & drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('active');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('active');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('active');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        selectedFile = file;
        analyzeBtn.disabled = false;
        dropZone.querySelector('h3').innerText = file.name;
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        loading.classList.remove('hidden');
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `Server returned status ${response.status}` }));
                throw new Error(errorData.error || 'Server error');
            }

            const data = await response.json();
            showResult(data);
            
        } catch (err) {
            console.error('Fetch error:', err);
            alert('Communication Error: ' + err.message + '\n\nPlease ensure the Flask server is running at http://127.0.0.1:5000');
        } finally {
            loading.classList.add('hidden');
        }
    });

    function showResult(data) {
        uploadCard.classList.add('hidden');
        resultCard.classList.remove('hidden');
        
        imagePreview.src = data.image_url;
        diagnosisStatus.innerText = data.prediction;
        diagnosisStatus.className = 'status-badge ' + 
            (data.prediction === 'NORMAL' ? 'status-normal' : 'status-pneumonia');
        
        // Animated progress
        setTimeout(() => {
            confidenceBar.style.width = data.confidence;
            confidenceText.innerText = data.confidence;
        }, 100);
    }

    resetBtn.addEventListener('click', () => {
        resultCard.classList.add('hidden');
        uploadCard.classList.remove('hidden');
        analyzeBtn.disabled = true;
        dropZone.querySelector('h3').innerText = 'Upload Chest X-Ray';
        selectedFile = null;
        confidenceBar.style.width = '0%';
    });
});
