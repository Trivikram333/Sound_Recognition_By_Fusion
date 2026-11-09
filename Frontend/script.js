<<<<<<< HEAD
const API = "http://localhost:8000";

// Turns any backend error shape into readable text
function errorMessage(data) {
    if (!data) return "Unknown error";
    if (data.error) return data.error;
    if (Array.isArray(data.detail)) return data.detail.map(d => d.msg).join(", ");
    return data.detail || data.message || "Unknown error";
}

/* ---------------- Predict ---------------- */
=======
>>>>>>> 202da7f9308cc9f513203728b3b384a02f600ee6
document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('audioUpload');
    const modelSelect = document.getElementById('modelSelect').value;
    const loadingText = document.getElementById('loading');
    const predictionText = document.getElementById('predictionText');
    const waveform = document.getElementById('waveform');
<<<<<<< HEAD
    const logmel = document.getElementById('melPlot');
    const rpPlot = document.getElementById('rpPlot');
    const resultsContainer = document.getElementById('PredictionResults');

=======
    const logmel=document.getElementById('melPlot');
    const rpPlot = document.getElementById('rpPlot');
    const resultsContainer = document.getElementById('PredictionResults'); 
>>>>>>> 202da7f9308cc9f513203728b3b384a02f600ee6
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
<<<<<<< HEAD
        const response = await fetch(`${API}/api/predict`, { method: "POST", body: formData });
        const data = await response.json();

        if (response.ok && data.status === "success") {
=======
        const response = await fetch("http://localhost:8000/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        
        if (data.status === "success") {
>>>>>>> 202da7f9308cc9f513203728b3b384a02f600ee6
            predictionText.innerText = `Prediction: ${data.top_class} (${data.confidence}%)`;
            waveform.src = data.wave;
            logmel.src = data.Logmel;
            rpPlot.src = data.rp_image;
<<<<<<< HEAD
            resultsContainer.style.display = 'block';
        } else {
            predictionText.innerText = "Prediction: --";
            alert("Error: " + errorMessage(data));
=======

            resultsContainer.style.display = 'block';
        } else {
            alert("Error: " + data.error);
>>>>>>> 202da7f9308cc9f513203728b3b384a02f600ee6
        }
    } catch (error) {
        console.error("Network Error:", error);
        alert("Failed to connect to backend.");
    } finally {
        loadingText.style.display = 'none';
    }
});
<<<<<<< HEAD

/* ---------------- Tab navigation ---------------- */
function switchTab(event, targetId) {
    document.querySelectorAll('.tab-content').forEach(tab => { tab.style.display = 'none'; });
    document.querySelectorAll('.nav-btn').forEach(btn => { btn.classList.remove('active'); });
    document.getElementById(targetId).style.display = 'block';
    event.currentTarget.classList.add('active');
}

/* ---------------- Dev training ---------------- */
const trainBtn = document.getElementById('trainBtn');
const trainStatus = document.getElementById('trainStatus');   // optional <p id="trainStatus">
let pollTimer = null;

function setTrainStatus(text) {
    if (trainStatus) trainStatus.textContent = text;
}

function stopPolling() {
    clearInterval(pollTimer);
    pollTimer = null;
    trainBtn.disabled = false;
}

async function pollTrainStatus() {
    try {
        const s = await (await fetch(`${API}/api/train-status`)).json();
        if (s.is_training) {
            setTrainStatus(`Training ${s.model_name}: epoch ${s.current_epoch}/${s.total_epochs}, loss ${s.current_loss}`);
        } else {
            stopPolling();
            setTrainStatus(`Run ended for ${s.model_name}: ${s.current_epoch}/${s.total_epochs} epochs, last loss ${s.current_loss}`);
        }
    } catch (e) {
        stopPolling();
        setTrainStatus("Lost connection to the backend.");
    }
}

trainBtn.addEventListener('click', async () => {
    const model_name = document.getElementById('modelSelectDev').value;   // sresnet | resnet | cnn_gru | vgg16
    const epochsInput = document.getElementById('epochsInput');           // add id="epochsInput" in the HTML
    const epochs = epochsInput ? parseInt(epochsInput.value, 10) : 50;

    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 500) {
        alert("Epochs must be a whole number between 1 and 500.");
        return;
    }

    trainBtn.disabled = true;
    try {
        const res = await fetch(`${API}/api/train`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ epochs, model_name })
        });
        const data = await res.json();
        if (!res.ok) {
            alert("Could not start training: " + errorMessage(data));
            trainBtn.disabled = false;
            return;
        }
        setTrainStatus(data.message);
        pollTimer = setInterval(pollTrainStatus, 2000);
    } catch (e) {
        alert("Failed to connect to backend.");
        trainBtn.disabled = false;
    }
});

/* ---------------- Save model checkpoint ---------------- */
document.getElementById('saveBtn').addEventListener('click', async () => {
    const model = document.getElementById('modelSelectDev').value;
    const formData = new FormData();
    formData.append("model_choice", model);

    try {
        const res = await fetch(`${API}/api/save-model`, { method: "POST", body: formData });
        const data = await res.json();
        alert(res.ok ? data.message : "Could not save: " + errorMessage(data));
    } catch (e) {
        alert("Failed to connect to backend.");
    }
=======
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
>>>>>>> 202da7f9308cc9f513203728b3b384a02f600ee6
});