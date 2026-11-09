import os
import glob
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt

def run_dataset_verification(csv_path=r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\meta\esc50.csv", audio_dir=r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\audio"):
    print("=" * 60)
    print("ESC-50 DATASET VERIFICATION & INTEGRITY AUDIT")
    print("=" * 60)

    # 1. Verify Metadata and File Counts
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing metadata file: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Total metadata records: {len(df)}")
    print(f"Total unique classes: {df['target'].nunique()} (Expected: 50)")

    audio_files = glob.glob(os.path.join(audio_dir, "*.wav"))
    print(f"Total .wav files detected on disk: {len(audio_files)} (Expected: 2000)")

    # 2. Audit Audio Files (Durations, Corruptions, SNR)
    durations = []
    snr_estimates = []
    corrupt_files = []
    
    print("\nAuditing audio integrity (sampling rate = 22,050 Hz)...")
    for _, row in df.iterrows():
        filepath = os.path.join(audio_dir, row["filename"])
        if not os.path.exists(filepath):
            corrupt_files.append((row["filename"], "File Missing"))
            continue
            
        try:
            y, sr = librosa.load(filepath, sr=22050)
            duration = librosa.get_duration(y=y, sr=sr)
            durations.append(duration)
            
            # Filter pure silence / calculate approximate Signal-to-Noise Ratio (dB)
            signal_power = np.mean(y ** 2)
            if signal_power < 1e-7:
                corrupt_files.append((row["filename"], "Zero/Near-Zero Signal"))
                snr_estimates.append(0.0)
            else:
                noise_estimate = np.var(y[: int(sr * 0.1)]) # 100ms noise floor estimate
                snr = 10 * np.log10(signal_power / (noise_estimate + 1e-8))
                snr_estimates.append(snr)
                
        except Exception as e:
            corrupt_files.append((row["filename"], str(e)))

    print(f"\nCorrupt / Anomaly Files Detected: {len(corrupt_files)}")
    if corrupt_files:
        for f, reason in corrupt_files:
            print(f"  - {f}: {reason}")
    else:
        print("  All 2,000 files passed integrity verification.")

    # 3. Generate Statistical Histograms (Week 2 Evaluation Deliverable)
    os.makedirs("reports", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # A. Duration Distribution
    axes[0].hist(durations, bins=20, color="#2563eb", edgecolor="black")
    axes[0].set_title("Clip Duration Distribution (sec)")
    axes[0].set_xlabel("Seconds")
    axes[0].set_ylabel("File Count")

    # B. Class Balance Distribution
    class_counts = df["category"].value_counts()
    axes[1].bar(range(len(class_counts)), class_counts.values, color="#10b981")
    axes[1].set_title("Class Balance (40 clips / category)")
    axes[1].set_xlabel("Class Index (0-49)")
    axes[1].set_ylabel("Clips")

    # C. Estimated SNR Distribution
    axes[2].hist(snr_estimates, bins=30, color="#f59e0b", edgecolor="black")
    axes[2].set_title("Estimated Signal-to-Noise Ratio (dB)")
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("File Count")

    plt.tight_layout()
    output_graph_path = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\results\dataset_stats.png"
    plt.savefig(output_graph_path, dpi=300)
    plt.close()

    print(f"\nVerification summary saved:")
    print(f"  - Statistical plots: {output_graph_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_dataset_verification()