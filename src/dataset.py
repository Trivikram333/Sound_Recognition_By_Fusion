import os
import pandas as pd
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T
class ESC50Dataset(Dataset):
    def __init__(self, csv_file, audio_dir=r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\audio", is_training=True):
        self.metadata = pd.read_csv(csv_file).reset_index(drop=True)
        self.audio_dir = audio_dir
        self.is_training = is_training
        
        # SpecAugment transforms applied strictly to training batches
        self.time_mask = T.TimeMasking(time_mask_param=35)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=16)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = os.path.join(self.audio_dir, row['filename'])
        data = torch.load(file_path, weights_only=True)
        mel = data['log_mel'].unsqueeze(0)  # Shape: [1, 128, 128]
        rp = data['rp'].unsqueeze(0)        # Shape: [1, 128, 128]
        label = data['label']

        # 4. Apply SpecAugment if training
        if self.is_training:
            mel = self.time_mask(mel)
            mel = self.freq_mask(mel)

        return (mel, rp), label


TRAIN_CSV = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\splits\train_fold1.csv"
VAL_CSV = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\splits\val_fold1.csv"

# Configure workers based on OS (Windows multiprocessing safe default = 0)
num_workers = 2 if os.name != 'nt' else 0

try:
    train_dataset = ESC50Dataset(csv_file=TRAIN_CSV, audio_dir="audio/", is_training=True)
    val_dataset = ESC50Dataset(csv_file=VAL_CSV, audio_dir="audio/", is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
except FileNotFoundError:
    print("Pre-split CSV files not found. Ensure make_splits has been executed.")
    train_loader, val_loader = None, None