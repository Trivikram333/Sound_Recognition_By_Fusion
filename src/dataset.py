import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

class ESC50Dataset(Dataset):
    def __init__(self, csv_file, audio_dir, folds):
        """
        Custom PyTorch Dataset for ESC-50.
        Filters the dataset to only include specific folds (e.g.,2, 3, 4, 5 for training).
        """
        # Load the metadata CSV
        self.metadata = pd.read_csv(csv_file)
        
        # Filter the dataset to only include the requested folds
        self.metadata = self.metadata[self.metadata['fold'].isin(folds)]
        
        self.audio_dir = audio_dir
        
    def __len__(self):
        return len(self.metadata)
        
    def __getitem__(self, idx):
        # 1. Get file path and label
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        label = row['target'] # ESC-50 class ID (0 to 49)
        
        # 2. Load Audio (Enforcing 22,050 Hz)
        y, sr = librosa.load(file_path, sr=22050)
        
        # 3. Extract Log-Mel Spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # 4. Convert to PyTorch Tensor with Channel Dimension -> Shape: [1, 128, X]
        tensor_mel = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        
        return tensor_mel, torch.tensor(label, dtype=torch.long)


try:
    train_dataset = ESC50Dataset(
        csv_file=r'C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\meta\esc50.csv', 
        audio_dir=r'C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\audio', 
        folds=[2, 3, 4, 5]  # Train on folds 2-5, reserve fold 1 for validation/testing
    )

    # train_loader handles batching (feeding 32 audio files at a time to the GPU/CPU)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
except FileNotFoundError:
    print("Warning: ESC-50 dataset files not found. train_loader is empty.")
    train_loader = None 