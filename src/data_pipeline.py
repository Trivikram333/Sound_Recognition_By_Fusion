import os
import librosa
import numpy as np
import pandas as pd

CSV_PATH = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\meta\esc50.csv"
AUDIO_DIR = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\audio"

def validate_dataset():
    print("Loading metadata...")
    df = pd.read_csv(CSV_PATH)
    total_files, valid_count, error_count = len(df), 0, 0

    for index, row in df.iterrows():
        file_path = os.path.join(AUDIO_DIR, row['filename'])
        if not os.path.exists(file_path):
            error_count += 1
            continue
            
        duration = librosa.get_duration(path=file_path)
        if round(duration, 2) != 5.00:
            print(f"[WARNING] Anomalous duration in {row['filename']}: {duration} sec")
            error_count += 1
        else:
            valid_count += 1
            
    print(f"Validated {total_files} files. Valid: {valid_count} | Errors: {error_count}")

def standardize_audio_length(file_path, target_sr=22050, window_sec=5.0, hop_sec=2.5):
    """Segments audio into perfectly sized 5.0-second chunks using sliding windows."""
    y, sr = librosa.load(file_path, sr=target_sr)
    target_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    total_samples = len(y)
    chunks = []

    if total_samples < target_samples:
        repeats = int(np.ceil(target_samples / total_samples))
        chunks.append(np.tile(y, repeats)[:target_samples])
    elif total_samples == target_samples:
        chunks.append(y)
    else:
        start = 0
        while start + target_samples <= total_samples:
            chunks.append(y[start : start + target_samples])
            start += hop_samples
        last_chunk_end = (start - hop_samples) + target_samples
        if last_chunk_end < total_samples:
            chunks.append(y[total_samples - target_samples : total_samples])

    return np.array(chunks)

if __name__ == "__main__":
    validate_dataset()