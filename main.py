from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import torch
import base64
import librosa
import numpy as np
import io
import uvicorn
import matplotlib
from pydantic import BaseModel
from fastapi import BackgroundTasks
import torch.nn as nn
import torch.optim as optim
import os
from src.dataset import train_loader  
matplotlib.use('Agg') # Add this to prevent the server from crashing when drawing graphs
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
    def generate_recurrence_plot_base64(y, m=3, tau=2, downsample_factor=100):
        """
        Generates a Threshold-Free Recurrence Plot via Phase Space Reconstruction.
        """
        # 1. PAA (Piecewise Downsampling)
        # 5 seconds of audio = 110,250 data points. 
        # An RP is an N x N matrix. Without downsampling, it would create a 12-billion pixel image and crash your RAM.
        y_paa = y[::downsample_factor] 
        
        # 2. Phase Space Reconstruction (Time-Delay Embedding)
        # Reconstructs the 1D audio wave into an m-dimensional trajectory
        N = len(y_paa) - (m - 1) * tau
        if N <= 0: return None # Failsafe for extremely short audio
        
        phase_space = np.array([y_paa[i : i + (m - 1)*tau + 1 : tau] for i in range(N)])
        
        # 3. Distance Matrix (Threshold-Free RP)
        distances = pdist(phase_space, metric='euclidean')
        rp_matrix = squareform(distances)
        
        # 4. Convert to Image and Base64
        buf = io.BytesIO()
        # 'inferno' or 'magma' are great colormaps for chaotic dynamics
        plt.imsave(buf, rp_matrix, cmap='inferno', format='png')
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    try:
        # 1. Read and decode the uploaded audio
        audio_bytes = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        # 2. Extract the Log-Mel Spectrogram (The exact math from Week 1)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        buf_mel = io.BytesIO()
        plt.imsave(buf_mel, log_mel, cmap='magma', format='png')
        mel_base64 = base64.b64encode(buf_mel.getvalue()).decode('utf-8')
        
        buf_wave = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6, 2)) # Create a wide, short canvas
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#2563eb") # Draw the blue wave
        ax.axis('off') # Hide the axes and borders
        
        fig.savefig(buf_wave, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig) # Clear RAM
        wave_base64 = base64.b64encode(buf_wave.getvalue()).decode('utf-8')

        rp_base64 = generate_recurrence_plot_base64(y)
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
            "wave":f"data:image/png;base64,{wave_base64}",
            "Logmel":f"data:image/png;base64,{mel_base64}",
            "rp_image": f"data:image/png;base64,{rp_base64}",
            "top_class": f"ESC-50 Class Code: {predicted_idx.item()}",
            "confidence": round(confidence.item() * 100, 2)
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# Global dictionary to track live training progress for the frontend
training_state = {
    "is_training": False,
    "model_name": "",
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0
}

def actual_pytorch_training_loop(model_name: str, epochs: int):
    """
    The actual deep learning training engine.
    Runs completely in the background without freezing the web server.
    """
    global training_state
    training_state["is_training"] = True
    training_state["model_name"] = model_name
    training_state["total_epochs"] = epochs
    
    # 1. Pull the requested model from memory and set to Training Mode
    active_model = models_dict.get(model_name, models_dict["resnet"])
    active_model.train() # Activates Dropout and BatchNorm updates
    
    # 2. Define the exact Math for Learning
    criterion = nn.CrossEntropyLoss() # Standard for 50-class classification
    optimizer = optim.Adam(active_model.parameters(), lr=0.001)
    
    try:
        # NOTE: You will import your PyTorch DataLoader from src.dataset here.
        # e.g., from src.dataset import train_loader
        
        for epoch in range(epochs):
            training_state["current_epoch"] = epoch + 1
            running_loss = 0.0
            
            # --- REAL PYTORCH BACKPROPAGATION LOOP ---
            for inputs, labels in train_loader:
                optimizer.zero_grad()               # Clear old gradients
                outputs = active_model(inputs)      # Forward pass (Predict)
                loss = criterion(outputs, labels)   # Calculate error
                loss.backward()                     # Backward pass (Calculate gradients)
                optimizer.step()                    # Update weights 
                running_loss += loss.item()
            
            # Simulate processing time for now so the UI doesn't instantly jump to 100%
            import time
            time.sleep(2) 
            
            training_state["current_loss"] = running_loss
            print(f"Epoch {epoch+1}/{epochs} Completed. Loss: {running_loss}")

    except Exception as e:
        print(f"Training failed: {e}")
        
    finally:
        # Ensure the server knows training finished even if an error occurred
        training_state["is_training"] = False
        active_model.eval() # Return to Inference Mode for the User Prediction tab


class TrainConfig(BaseModel):
    epochs: int = 50
    model_name: str = "sresnet"
@app.post("/api/train")
async def trigger_training(config: TrainConfig, background_tasks: BackgroundTasks):
    """
    Accepts the UI request and kicks the PyTorch loop to a background thread.
    """
    if training_state["is_training"]:
        return JSONResponse(status_code=400, content={"error": "A model is already training."})
    
    # Hand the heavy PyTorch function to FastAPI's background thread manager
    background_tasks.add_task(actual_pytorch_training_loop, config.model_name, config.epochs)
    
    return JSONResponse(content={
        "status": "success",
        "message": f"Training started in the background for {config.model_name}.",
    })

@app.get("/api/train-status")
async def get_training_status():
    """
    Allows the frontend Statistics/Training tab to poll the live epoch and loss data.
    """
    return JSONResponse(content=training_state)

@app.post("/api/save-model")
async def save_checkpoint(model_choice: str = Form(...)):
    """
    Physically extracts the tensor weights from RAM and saves them to your hard drive.
    """
    if training_state["is_training"]:
        return JSONResponse(status_code=400, content={"error": "Cannot save while training is active."})
        
    active_model = models_dict.get(model_choice, models_dict["resnet"])
    
    # Ensure a weights folder exists in your VSCode directory
    os.makedirs("weights", exist_ok=True)
    save_path = f"weights/{model_choice}_best_model.pth"
    
    # Save the PyTorch dictionary securely
    torch.save(active_model.state_dict(), save_path)
    
    return JSONResponse(content={
        "status": "success",
        "message": f"Model successfully saved to {save_path}"
    })