const API_BASE_URL = 'http://localhost:8000/api/v1';

// Tab Switching
document.querySelectorAll('.tab-btn').forEach(button => {
    button.addEventListener('click', () => {
        // Remove active class from all buttons and contents
        document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

        // Add active class to clicked button and target content
        button.classList.add('active');
        document.getElementById(button.dataset.tab).classList.add('active');
    });
});

// Query Form Submission
document.getElementById('query-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const submitBtn = e.target.querySelector('button');
    const outputArea = document.getElementById('query-output');
    const resultsContainer = document.getElementById('query-results');

    // UI Feedback
    const originalBtnText = submitBtn.textContent;
    submitBtn.textContent = 'Processing...';
    submitBtn.disabled = true;
    resultsContainer.classList.add('hidden');

    // Gather Data
    const formData = {
        question: document.getElementById('question').value,
        max_results: parseInt(document.getElementById('max_results').value),
        confidence_threshold: parseFloat(document.getElementById('confidence_threshold').value)
    };

    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        const data = await response.json();
        
        // Display Results
        outputArea.textContent = JSON.stringify(data, null, 2);
        resultsContainer.classList.remove('hidden');
    } catch (error) {
        outputArea.textContent = `Error: ${error.message}`;
        resultsContainer.classList.remove('hidden');
    } finally {
        submitBtn.textContent = originalBtnText;
        submitBtn.disabled = false;
    }
});

// Upload Form Submission
document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const submitBtn = e.target.querySelector('button');
    const outputArea = document.getElementById('upload-output');
    const resultsContainer = document.getElementById('upload-results');

    // UI Feedback
    const originalBtnText = submitBtn.textContent;
    submitBtn.textContent = 'Uploading...';
    submitBtn.disabled = true;
    resultsContainer.classList.add('hidden');

    // Gather Data
    const formData = new FormData();
    const fileInput = document.getElementById('file');
    
    if (fileInput.files.length === 0) {
        alert('Please select a file');
        submitBtn.textContent = originalBtnText;
        submitBtn.disabled = false;
        return;
    }

    formData.append('file', fileInput.files[0]);
    formData.append('document_type', document.getElementById('document_type').value);
    
    // Create metadata object
    const metadata = {
        department: document.getElementById('department').value || 'General',
        timestamp: new Date().toISOString()
    };
    formData.append('metadata', JSON.stringify(metadata));

    try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
            // Note: Content-Type header is automatically set for FormData
        });

        const data = await response.json();
        
        // Display Results
        outputArea.textContent = JSON.stringify(data, null, 2);
        resultsContainer.classList.remove('hidden');
    } catch (error) {
        outputArea.textContent = `Error: ${error.message}`;
        resultsContainer.classList.remove('hidden');
    } finally {
        submitBtn.textContent = originalBtnText;
        submitBtn.disabled = false;
    }
});
