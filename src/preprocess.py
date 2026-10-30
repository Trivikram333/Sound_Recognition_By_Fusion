import os
import torch
import librosa
import numpy as np
import pandas as pd

AUDIO_DIR = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\audio"
OUT_DIR = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\processed"
META_CSV = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\meta\esc50.csv"
os.makedirs(OUT_DIR, exist_ok=True)

SR = 22050
EPS = 1e-10
PAA_LEN = 5 * SR // 2  # 110250: fixed length after PAA so tensors can be batched


def normalize_amplitude(y):
    """Scale so the loudest sample has magnitude 1."""
    return y / (np.max(np.abs(y)) + EPS)


def stes_crop(y, sr, frame_ms=25, shift_ms=10, ratio=0.6):
    """
    STES endpoint detection: keep the longest run of low-entropy frames.
    Steps follow the paper: frame + Hamming window -> FFT power spectrum
    -> normalize to a distribution -> spectral entropy -> threshold.
    The paper does not give the threshold T, so `ratio` here is our own
    adaptive choice (T sits `ratio` of the way between min and max entropy).
    Tune it by plotting a few clips.
    """
    frame_len = int(sr * frame_ms / 1000)
    hop = int(sr * shift_ms / 1000)
    if len(y) < frame_len:
        return y

    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop).T
    frames = frames * np.hamming(frame_len)

    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2          # P(k) = |X(k)|^2
    energy = power.sum(axis=1)
    p = power / (energy[:, None] + EPS)                        # p(k)
    H = -np.sum(p * np.log2(p + EPS), axis=1)                  # entropy

    # ESC-50 pads short clips with digital silence. Those frames have p = 0,
    # so H = 0 (the lowest entropy) and would wrongly count as "event" frames.
    voiced = energy > 1e-4 * energy.max()
    if not voiced.any():
        return y

    Hv = H[voiced]
    T = Hv.min() + ratio * (Hv.max() - Hv.min())
    active = voiced & (H < T)

    # longest consecutive run of active frames
    edges = np.flatnonzero(np.diff(np.concatenate(([0], active.astype(np.int8), [0]))))
    starts, ends = edges[::2], edges[1::2]
    if len(starts) == 0:
        return y
    k = np.argmax(ends - starts)
    start = starts[k] * hop
    end = min(ends[k] * hop + frame_len, len(y))

    if end - start < int(0.1 * sr):  # too short to be useful, keep full clip
        return y
    return y[start:end]


def paa_halve(y):
    """PAA with segment size 2: each pair of samples -> its mean."""
    M = len(y) // 2
    return np.mean(y[:M * 2].reshape(-1, 2), axis=1)

import numpy as np
import torch
import torch.nn.functional as F


"""def compute_rp(y_paa, m=3, tau=2, series_len=2048, out_size=128):

    Threshold-free Recurrence Plot from the PAA signal.

    y_paa      : 1D array/tensor (output of STES + PAA)
    m          : embedding dimension (paper doesn't state a value)
    tau        : time delay in samples (paper doesn't state a value)
    series_len : length the signal is shrunk to before building the matrix,
                 so the N x N distance matrix fits in memory
    out_size   : the RP is resized to out_size x out_size
    returns    : float32 tensor of shape (out_size, out_size), values in [0, 1]

    y = np.asarray(y_paa, dtype=np.float32)

    # 1. Shrink to series_len points by averaging equal segments (extra PAA)
    if len(y) >= series_len:
        seg = len(y) // series_len
        y = y[: series_len * seg].reshape(series_len, seg).mean(axis=1)
    else:
        y = np.resize(y, series_len)

    # 2. Time-delay embedding: y_i = (x_i, x_{i+tau}, ..., x_{i+(m-1)tau})
    N = series_len - (m - 1) * tau
    Y = np.stack([y[i * tau: i * tau + N] for i in range(m)], axis=1)  # (N, m)

    # 3. Euclidean distance between every pair of state vectors
    Yt = torch.from_numpy(Y)
    D = torch.cdist(Yt, Yt)  # (N, N)

    # 4. Threshold-free: keep the distances instead of a 0/1 cutoff,
    #    scaled to [0, 1] (0 = same state, 1 = most different)
    D = D / (D.max() + 1e-10)

    # 5. Resize to a fixed image size
    D = F.adaptive_avg_pool2d(D[None, None], out_size)[0, 0]
    return D.float()
"""
df = pd.read_csv(META_CSV)

print("Extracting features and saving as PyTorch tensors...")
for index, row in df.iterrows():
    filename = row['filename']
    file_path = os.path.join(AUDIO_DIR, filename)

    # 1. Load audio
    y, sr = librosa.load(file_path, sr=SR)

    # 2. Amplitude normalization
    y = normalize_amplitude(y)

    # 3. STES endpoint detection (nonlinear branch only)
    y_event = stes_crop(y, sr)

    # 4. Linear feature: Log-Mel (from the full normalized clip, fixed shape)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # 5. Nonlinear prep: PAA on the cropped event, then fixed length
    y_paa = paa_halve(y_event)
    y_paa = np.resize(y_paa, PAA_LEN)  # repeats if short, truncates if long

    tensor_data = {
        'log_mel': torch.tensor(log_mel, dtype=torch.float32),
        'paa': torch.tensor(y_paa, dtype=torch.float32),
        'label': torch.tensor(row['target'], dtype=torch.long),
    }

    out_name = filename.replace('.wav', '.pt')
    torch.save(tensor_data, os.path.join(OUT_DIR, out_name))

    if (index + 1) % 500 == 0:
        print(f"Processed {index + 1}/{len(df)} files...")

print("Preprocessing complete. Tensors saved to data/processed/")