import os
import torch
import librosa
import numpy as np
import pandas as pd

AUDIO_DIR = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\audio"
OUT_DIR = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\processed"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\meta\esc50.csv")

print("Extracting features and saving as PyTorch tensors...")
for index, row in df.iterrows():
    filename = row['filename']
    file_path = os.path.join(AUDIO_DIR, filename)
    
    # 1. Load Audio
    y, sr = librosa.load(file_path, sr=22050)
    
    # 2. Linear Feature: Log-Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    # 3. Nonlinear Feature Prep: PAA (Halve the sequence length)
    M = len(y) // 2
    y_paa = np.mean(y[:M*2].reshape(-1, 2), axis=1)
    
    # 4. Save as a dictionary tensor
    tensor_data = {
        'log_mel': torch.tensor(log_mel, dtype=torch.float32),
        'paa': torch.tensor(y_paa, dtype=torch.float32),
        'label': torch.tensor(row['target'], dtype=torch.long)
    }
    
    out_name = filename.replace('.wav', '.pt')
    torch.save(tensor_data, os.path.join(OUT_DIR, out_name))
    
    if (index + 1) % 500 == 0:
        print(f"Processed {index + 1}/2000 files...")

print("Preprocessing complete. Tensors saved to data/processed/")