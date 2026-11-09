import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
EPS=1e-10
sr=22050
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

y,sr=librosa.load(librosa.ex('trumpet'), sr=sr)
fig, ax = plt.subplots(nrows=4, sharex=True, sharey=True)
ax[0].set_xlabel('time (s)')
ax[0].set_ylabel('Amplitude')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('time (s)')
librosa.display.waveshow(y, sr=sr, ax=ax[0])
y = normalize_amplitude(y)
y = stes_crop(y, sr)
librosa.display.waveshow(y, sr=sr, ax=ax[1])
plt.show()