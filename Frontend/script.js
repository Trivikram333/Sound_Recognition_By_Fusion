document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('audioUpload');
    const modelSelect = document.getElementById('modelSelect').value;
    const loadingText = document.getElementById('loading');
    const predictionText = document.getElementById('predictionText');
    const waveform = document.getElementById('waveform');
    const logmel=document.getElementById('melPlot');
    const rpPlot = document.getElementById('rpPlot');
    const resultsContainer = document.getElementById('PredictionResults'); 
    if (fileInput.files.length === 0) {
        alert("Please select or record an audio file first.");
        return;
    }

    // Package the file AND the selected model
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("model_choice", modelSelect);

    loadingText.style.display = 'block';
    resultsContainer.style.display = 'none';
    predictionText.innerText = "Prediction: Calculating...";

    try {
        const response = await fetch("http://localhost:8000/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        
        if (data.status === "success") {
            predictionText.innerText = `Prediction: ${data.top_class} (${data.confidence}%)`;
            waveform.src = data.wave;
            logmel.src = data.Logmel;
            rpPlot.src = data.rp_image;

            resultsContainer.style.display = 'block';
        } else {
            alert("Error: " + data.error);
        }
    } catch (error) {
        console.error("Network Error:", error);
        alert("Failed to connect to backend.");
    } finally {
        loadingText.style.display = 'none';
    }
});
// Tab Navigation Logic
function switchTab(event, targetId) {
    // 1. Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.style.display = 'none';
    });

    // 2. Remove 'active' styling from all navigation buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // 3. Show the selected tab content
    document.getElementById(targetId).style.display = 'block';

    // 4. Highlight the button that was just clicked
    event.currentTarget.classList.add('active');
}

// Trigger Training
document.getElementById('trainBtn').addEventListener('click', async () => {
    const model = document.getElementById('modelSelect').value;
    alert(`Initiating training for ${model}...`);
    await fetch("http://localhost:8000/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ epochs: 50, model_name: model })
    });
});

// Save Model Checkpoint
document.getElementById('saveBtn').addEventListener('click', async () => {
    const model = document.getElementById('modelSelect').value;
    const formData = new FormData();
    formData.append("model_choice", model);
    
    const response = await fetch("http://localhost:8000/api/save-model", {
        method: "POST",
        body: formData
    });
    const data = await response.json();
    alert(data.message);
});