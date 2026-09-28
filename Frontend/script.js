async function uploadAudio() {
      const fileInput = document.getElementById('audioInput');
      const resultDiv = document.getElementById('result');
      if (fileInput.files.length === 0) {
        alert("Please select an audio file first.");
        return;
      }
      const file = fileInput.files[0];
      resultDiv.innerText = "Processing audio...";
      const formData = new FormData();
      formData.append("file", file);

      try {
        const response = await fetch("http://localhost:8000/predict", {
          method: "POST",
          body: formData
        });
        if (!response.ok) {
          throw new Error("Server error: " + response.statusText);
        }
        const data = await response.json();
        resultDiv.innerHTML = `
          Prediction: <b>${data.prediction}</b><br>
          Confidence: <b>${(data.confidence * 100).toFixed(2)}%</b>`;
      } catch (error) {
        resultDiv.innerText = "Error uploading file: " + error.message;
      }
    }