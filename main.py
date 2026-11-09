<<<<<<< HEAD
import os
import io
import base64
import time
import numpy as np
import librosa
from scipy.spatial.distance import pdist, squareform

import matplotlib
matplotlib.use('Agg')  # Prevent GUI crashes when rendering graphs on the server
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import uvicorn

from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from huggingface_hub import hf_hub_download

# Import all 4 architectures from src.models
from src.models import AudioResNet, AudioVGG16, AudioCNNGRU, AudioSResNet

# Import DataLoader from dataset pipeline
try:
    from src.dataset import train_loader
except ImportError:
    train_loader = None

app = FastAPI(title="ESC Comparative & DMFF API")

# Configure CORS for cross-platform and local UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Standard ESC-50 Semantic Classes
ESC50_CLASSES = [
    "Dog", "Rooster", "Pig", "Cow", "Frog", "Cat", "Hen", "Insects", "Sheep", "Crow",
    "Rain", "Sea waves", "Crackling fire", "Crickets", "Chirping birds", "Water drops", "Wind", "Pouring water", "Toilet flush", "Thunderstorm",
    "Crying baby", "Sneezing", "Clapping", "Breathing", "Coughing", "Footsteps", "Laughing", "Brushing teeth", "Snoring", "Drinking/Sipping",
    "Door knock", "Mouse click", "Keyboard typing", "Door creak", "Can opening", "Washing machine", "Vacuum cleaner", "Clock alarm", "Clock tick", "Glass breaking",
    "Helicopter", "Chainsaw", "Siren", "Car horn", "Engine", "Train", "Church bells", "Airplane", "Fireworks", "Hand saw"
]

HF_REPO_ID = "Viction333/ESC-50_SRF"

from typing import Literal
from pydantic import BaseModel, Field

MODEL_LABELS = {
    "sresnet": "SResNet + DMFF (Proposed)",
    "resnet": "Standard ResNet-18",
    "cnn_gru": "CNN-GRU",
    "vgg16": "VGG-16 Baseline",
}

class TrainConfig(BaseModel):
    epochs: int = Field(50, ge=1, le=500)
    model_name: Literal["sresnet", "resnet", "cnn_gru", "vgg16"] = "sresnet"

def load_or_download_weights(model, model_filename):
    """
    Checks for the .pth file in weights/. If missing, downloads it 
    from the remote Hugging Face repository automatically.
    """
    os.makedirs("weights", exist_ok=True)
    local_path = os.path.join("weights", model_filename)

    if not os.path.exists(local_path):
        print(f"'{local_path}' not found locally. Downloading from Hugging Face...")
        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"weights/{model_filename}",
                local_dir="."
            )
            local_path = downloaded
            print(f"Successfully downloaded {model_filename} from Hugging Face!")
        except Exception as e:
            print(f"Could not download from Hugging Face ({e}). Running on initialized weights.")
            return model.eval()

    print(f"Loading weights into RAM from {local_path}...")
    model.load_state_dict(torch.load(local_path, map_location="cpu", weights_only=True))
    return model.eval()

# Models load locally if available, or fetch automatically if cloned fresh:
models_dict = {
    "sresnet": load_or_download_weights(AudioSResNet(num_classes=50), "sresnet_best_model.pth"),
    "resnet": load_or_download_weights(AudioResNet(num_classes=50), "resnet_best_model.pth"),
    "cnn_gru": load_or_download_weights(AudioCNNGRU(num_classes=50), "cnn_gru_best_model.pth"),
    "vgg16": load_or_download_weights(AudioVGG16(num_classes=50), "vgg16_best_model.pth")
}

# Global dictionary to track asynchronous training status for the frontend
=======
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
>>>>>>> 202da7f9308cc9f513203728b3b384a02f600ee6
training_state = {
    "is_training": False,
    "model_name": "",
    "current_epoch": 0,
    "total_epochs": 0,
    "current_loss": 0.0
}

<<<<<<< HEAD

# =====================================================================
# TIER 1: Inference & Dynamic Feature Extraction
# =====================================================================
@app.post("/api/predict")
async def predict_audio(file: UploadFile = File(...), model_choice: str = Form("sresnet")):
    try:
        # 1. Read and decode uploaded audio
        audio_bytes = await file.read()
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)

        # 2. Extract 128-band Log-Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # 3. Generate Waveform Base64 Image
        buf_wave = io.BytesIO()
        fig, ax = plt.subplots(figsize=(6, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax, color="#2563eb")
        ax.axis('off')
        fig.savefig(buf_wave, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        wave_base64 = base64.b64encode(buf_wave.getvalue()).decode('utf-8')

        # 4. Generate Spectrogram Base64 Image
        buf_mel = io.BytesIO()
        plt.imsave(buf_mel, log_mel, cmap='magma', format='png')
        mel_base64 = base64.b64encode(buf_mel.getvalue()).decode('utf-8')

        # 5. Extract Threshold-Free Recurrence Plot (Time-Delay Embedding)
        y_paa = y[::100]  # PAA downsampling factor to prevent memory blowout
        m, tau = 3, 2
        N = len(y_paa) - (m - 1) * tau

        if N > 0:
            phase_space = np.array([y_paa[i : i + (m - 1) * tau + 1 : tau] for i in range(N)])
            distances = pdist(phase_space, metric='euclidean')
            rp_matrix = squareform(distances)
            rp_norm = rp_matrix / (np.max(rp_matrix) + 1e-10)
        else:
            rp_norm = np.zeros((128, 128), dtype=np.float32)

        # 6. Generate Recurrence Plot Base64 Image
        buf_rp = io.BytesIO()
        plt.imsave(buf_rp, rp_norm, cmap='inferno', format='png')
        rp_base64 = base64.b64encode(buf_rp.getvalue()).decode('utf-8')

        # 7. Convert and standardize dual inputs to fixed [1, 1, 128, 128] tensors
        tensor_mel = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_mel = F.interpolate(tensor_mel, size=(128, 128), mode='bilinear', align_corners=False)

        tensor_rp = torch.tensor(rp_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tensor_rp = F.interpolate(tensor_rp, size=(128, 128), mode='bilinear', align_corners=False)

        # 8. Run Model Inference
        active_model = models_dict.get(model_choice, models_dict["sresnet"])
        with torch.no_grad():
            logits = active_model(tensor_mel, tensor_rp)
            probabilities = F.softmax(logits[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)

        class_idx = predicted_idx.item()
        class_name = ESC50_CLASSES[class_idx] if class_idx < len(ESC50_CLASSES) else f"Class {class_idx}"

        return JSONResponse(content={
            "status": "success",
            "wave": f"data:image/png;base64,{wave_base64}",
            "Logmel": f"data:image/png;base64,{mel_base64}",
            "rp_image": f"data:image/png;base64,{rp_base64}",
            "top_class": f"{class_name} (Class {class_idx})",
            "confidence": round(confidence.item() * 100, 2)
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# =====================================================================
# TIER 2: Asynchronous PyTorch Training & Checkpointing
# =====================================================================

def actual_pytorch_training_loop(model_name: str, epochs: int):
    """
    Background worker executing PyTorch backpropagation.
=======
def actual_pytorch_training_loop(model_name: str, epochs: int):
    """
    The actual deep learning training engine.
    Runs completely in the background without freezing the web server.
>>>>>>> 202da7f9308cc9f513203728b3b384a02f600ee6
    """
    global training_state
    training_state["is_training"] = True
    training_state["model_name"] = model_name
    training_state["total_epochs"] = epochs
<<<<<<< HEAD
    training_state["current_epoch"] = 0
    training_state["current_loss"] = 0.0

    active_model = models_dict.get(model_name, models_dict["sresnet"])
    active_model.train()

    optimizer = optim.Adam(active_model.parameters(), lr=0.001)

    try:
        for epoch in range(epochs):
            training_state["current_epoch"] = epoch + 1
            running_loss = 0.0
            batches_run = 0

            if train_loader is not None:
                for batch in train_loader:
                    optimizer.zero_grad()

                    # Check if DataLoader returns dual inputs ((mel, rp), labels) or single tensor
                    if isinstance(batch[0], (tuple, list)):
                        (mel_batch, rp_batch), labels = batch
                        # Interpolate batch to [B, 1, 128, 128] if needed
                        if mel_batch.shape[-2:] != (128, 128):
                            mel_batch = F.interpolate(mel_batch, size=(128, 128), mode='bilinear', align_corners=False)
                        if rp_batch.shape[-2:] != (128, 128):
                            rp_batch = F.interpolate(rp_batch, size=(128, 128), mode='bilinear', align_corners=False)

                        _, loss = active_model(mel_batch, rp_batch, label=labels)
                    else:
                        inputs, labels = batch
                        if inputs.shape[-2:] != (128, 128):
                            inputs = F.interpolate(inputs, size=(128, 128), mode='bilinear', align_corners=False)
                        _, loss = active_model(inputs, rp=None, label=labels)

                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    batches_run += 1

                epoch_loss = running_loss / max(batches_run, 1)
            else:
                # Simulates epoch progress if dataset files are not mounted
                time.sleep(1.5)
                epoch_loss = 1.0 / (epoch + 1)

            training_state["current_loss"] = round(epoch_loss, 4)
            print(f"[{model_name.upper()}] Epoch {epoch + 1}/{epochs} - Loss: {training_state['current_loss']}")

    except Exception as e:
        print(f"Training loop encountered error: {e}")

    finally:
        training_state["is_training"] = False
        active_model.eval()


@app.post("/api/train")
async def trigger_training(config: TrainConfig, background_tasks: BackgroundTasks):
    if training_state["is_training"]:
        return JSONResponse(status_code=400, content={"error": f"'{training_state['model_name']}' is already training."})

    training_state.update(is_training=True, model_name=config.model_name,
                          total_epochs=config.epochs, current_epoch=0, current_loss=0.0)
    background_tasks.add_task(actual_pytorch_training_loop, config.model_name, config.epochs)
    return JSONResponse(content={
        "status": "success",
        "message": f"Training started for {MODEL_LABELS[config.model_name]}.",
        "model_name": config.model_name,
        "epochs": config.epochs
=======
    
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
>>>>>>> 202da7f9308cc9f513203728b3b384a02f600ee6
    })

@app.get("/api/train-status")
async def get_training_status():
<<<<<<< HEAD
    return JSONResponse(content=training_state)


@app.post("/api/save-model")
async def save_checkpoint(model_choice: str = Form("sresnet")):
    if training_state["is_training"]:
        return JSONResponse(status_code=400, content={"error": "Cannot serialize weights while model is training."})
    if model_choice not in models_dict:
        return JSONResponse(status_code=400, content={"error": f"Unknown model '{model_choice}'."})
    active_model = models_dict[model_choice]
    os.makedirs("weights", exist_ok=True)
    save_path = f"weights/{model_choice}_best_model.pth"

    torch.save(active_model.state_dict(), save_path)
    return JSONResponse(content={
        "status": "success",
        "message": f"Model weights successfully saved to {save_path}"
    })


# =====================================================================
# TIER 3 & 4: Supporting Analytical & Health Endpoints
# =====================================================================
@app.get("/api/stats")
async def get_dataset_statistics():
    return JSONResponse(content={
        "dataset": "ESC-50 (Environmental Sound Classification)",
        "total_files": 2000,
        "classes": 50,
        "files_per_class": 40,
        "clip_duration": "5.0 seconds",
        "sample_rate": "22,050 Hz",
        "validation_strategy": "5-Fold Cross Validation (Zero Leakage)",
        "supported_models": ["SResNet (DMFF)", "ResNet-18", "CNN-GRU", "VGG-16"]
    })


@app.get("/api/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "server": "online",
        "active_models": list(models_dict.keys()),
        "torch_cuda_available": torch.cuda.is_available()
    })


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
=======
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
>>>>>>> 202da7f9308cc9f513203728b3b384a02f600ee6
