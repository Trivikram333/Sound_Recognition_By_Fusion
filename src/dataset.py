import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

class ESC50Dataset(Dataset):
    def __init__(self, split_csv, processed_dir=r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\processed", is_train=True):
        self.df = pd.read_csv(split_csv)
        self.processed_dir = processed_dir
        self.is_train = is_train
        
        # SpecAugment parameters matching the paper's augmentation strategy
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx]['filename'].replace('.wav', '.pt')
        data = torch.load(os.path.join(self.processed_dir, filename), weights_only=True)
        
        log_mel = data['log_mel']
        paa = data['paa']
        label = data['label']
        
        # Add channel dimension for CNN: (128, 216) -> (1, 128, 216)
        log_mel = log_mel.unsqueeze(0)
        
        # Apply SpecAugment only during training
        if self.is_train:
            log_mel = self.freq_mask(log_mel)
            log_mel = self.time_mask(log_mel)
            
        return log_mel, paa, label

# --- Test the DataLoader throughput ---
if __name__ == "__main__":
    train_dataset = ESC50Dataset(r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\splits\train_fold1.csv", is_train=True)
    # num_workers=4 optimizes loading speed
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    
    log_mel_batch, paa_batch, labels = next(iter(train_loader))
    print(f"Log-Mel Batch Shape: {log_mel_batch.shape}") # Expected: [32, 1, 128, 216]
    print(f"PAA Batch Shape: {paa_batch.shape}")         # Expected: [32, 55125]
    print(f"Labels Shape: {labels.shape}")               # Expected: [32]