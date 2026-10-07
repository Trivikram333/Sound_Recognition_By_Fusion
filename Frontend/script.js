document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('audioUpload');
    const modelSelect = document.getElementById('modelSelect').value;
    const loadingText = document.getElementById('loading');
    const predictionText = document.getElementById('predictionText');
    
    if (fileInput.files.length === 0) {
        alert("Please select or record an audio file first.");
        return;
    }

    // Package the file AND the selected model
    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    formData.append("model_choice", modelSelect);

    loadingText.style.display = 'block';
    predictionText.innerText = "Prediction: Calculating...";

    try {
        const response = await fetch("http://localhost:8000/api/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();
        
        if (data.status === "success") {
            predictionText.innerText = `Prediction: ${data.top_class} (${data.confidence}%)`;
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