from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import librosa
import numpy as np
import io
import uvicorn
# Import your new pre-trained models
from src.models import AudioResNet, AudioVGG16

app = FastAPI(title="ESC Comparative API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Load the models into memory once when the server starts
print("Loading pre-trained models into RAM...")
models_dict = {
    "resnet": AudioResNet().eval(),
    "vgg16": AudioVGG16().eval()
    # You will add SResNet and CNN-GRU here once you build them in Week 3/4
}
print("Models ready!")

@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...), model_choice: str = Form(...)):
    try:
        # 1. Read and decode the uploaded audio
        audio_bytes = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # 2. Extract the Log-Mel Spectrogram (The exact math from Week 1)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # 3. Wrap it in a PyTorch Tensor and add Batch & Channel dimensions -> Shape: [1, 1, 128, X]
        tensor_mel = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 4. Select the user's requested model (fallback to ResNet if not found)
        active_model = models_dict.get(model_choice, models_dict["resnet"])
        
        # 5. Run REAL Neural Network Inference
        with torch.no_grad():
            outputs = active_model(tensor_mel)
            
            # Convert raw network output into percentages (0 to 1)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Find the highest percentage and its class ID
            confidence, predicted_idx = torch.max(probabilities, 0)
            
        return JSONResponse(content={
            "status": "success",
            "top_class": f"ESC-50 Class Code: {predicted_idx.item()}",
            "confidence": round(confidence.item() * 100, 2)
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})