# Environmental Sound Classification Using SResNet \& Dual-Feature Fusion 🎧🤖

### 📌 Overview

In real-world acoustic monitoring, identifying ambient sound events—such as emergency alarms, industrial machinery, animal vocalizations, or road traffic hazards—is a critical, resource-intensive challenge. Conventional Environmental Sound Classification (ESC) frameworks rely almost exclusively on linear time-frequency representations (e.g., Log-Mel spectrograms, MFCCs), which discard signal phase information and fail to capture complex, non-linear physical dynamics.

This project leverages **SResNet**, an optimized deep neural network, combined with a **dual-modal parallel feature fusion** approach. By integrating linear spectral representations (Log-Mel, GFCC) with nonlinear dynamic representations (**Threshold-Free Recurrence Plots**), this system accurately characterizes both macroscopic frequency patterns and microscopic chaotic dynamics to automate environmental sound recognition with state-of-the-art accuracy.

\---

### 🎯 Problem Statement

*"Given a raw environmental audio clip, can a deep learning model predict the correct acoustic event category by jointly learning from linear spectral content and nonlinear phase-space dynamics?"*

This is framed as a multi-class acoustic classification problem:

* **Input:** Raw environmental audio recordings (`.wav` format).
* **Feature Processing:** Parallel extraction of linear acoustic features (Log-Mel, GFCC) and nonlinear state-space dynamics (Threshold-Free Recurrence Plots).
* **Output:** Predicted sound event category (e.g., Siren, Dog Bark, Drilling, Sea Waves) with associated Softmax confidence scores.

\---

### 🛠️ Tech Stack \& Tools

* **Programming \& Environment:** Python 3.10+, Jupyter Notebook / VS Code
* **Data Manipulation \& Scientific Computing:** NumPy, pandas, SciPy
* **Audio Signal Processing:** `librosa`, `soundfile`
* **Auditory Filterbanks \& Nonlinear Dynamics:** `spafe` (for GFCC extraction), custom phase-space delay embedding algorithms
* **Deep Learning Framework:** Keras with TensorFlow backend *(Optional GPU acceleration via PyTorch / Torchaudio)*
* **Data Visualization:** Matplotlib, Seaborn
* **Web App Deployment:** Streamlit / Gradio

\---

### 🗄️ Datasets Used

High-quality acoustic data is essential for training an accurate sound classifier. This project utilizes two established ESC benchmark datasets:

* **ESC-50:** A curated dataset of 2,000 environmental audio recordings (5 seconds each) spanning 5 major acoustic domains: animal sounds, natural soundscapes/water, human non-speech, interior/domestic sounds, and exterior/urban noises (comprising 50 fine-grained subcategories). Samples are augmented via repeated sampling, time stretching, and pitch shifting to expand the dataset to 6,000 clips.
* **UrbanSound8K (US8K):** A standard urban sound benchmark containing 8,732 audio clips (each $\\le$ 4 seconds) pre-arranged into 10 cross-validation folds. It includes 10 balanced urban sound classes: air conditioner, car horn, children playing, dog bark, drilling, engine idling, gun shot, jackhammer, siren, and street music.

> \*\*Note on Data:\*\* Due to GitHub's file storage limits, raw audio dataset files are not tracked in this repository. You can download them directly from their official public repositories or use the data-download scripts provided in the `data/` folder.

\---

### 📊 Methodology

* **Data Acquisition \& Normalization:** Ingesting raw audio recordings and normalizing waveform amplitudes across all samples.
* **Data Preprocessing \& Sequence Compression:**

  * Applying **Short-Time Spectral Entropy (STES)** endpoint detection to crop silent and non-informative background noise.
  * Utilizing **Piecewise Aggregate Approximation (PAA)** to halve time-series length while preserving underlying wave trajectories, preventing GPU out-of-memory bottlenecks during recurrence calculation.
* **Dual-Pathway Feature Engineering:**

  * *Linear Stream:* Extracting Log-Mel Spectrograms and Gammatone Frequency Cepstral Coefficients (GFCC).
  * *Nonlinear Stream:* Performing phase space reconstruction via time-delay embedding ($m, \\tau$) to construct continuous **Threshold-Free Recurrence Plots (RP)**, compressed via dilated convolutions ($\\text{Rate}=2$).
* **Dual-Modal Feature Fusion (DMFF):** Converting spatial maps into tokens and employing bidirectional cross-attention modules (CFE) to dynamically align and cross-reference linear frequency tokens with nonlinear recurrence tokens.
* **Model Training (SResNet):** Feeding the fused tensor into an enhanced ResNet-18 backbone equipped with **SwiGLU-gated residual blocks** to suppress acoustic static and optimize gradient flow, supervised by categorical cross-entropy loss over 100 iterations with SGD.
* **Evaluation:** Assessing model performance against standard baseline models (VGG-16, standard ResNet-18, CNN-GRU, CNN-RNN, and St-HQCNN) across official cross-validation folds.
* **Deployment:** Packaging the pipeline into an interactive web application where users can upload audio files, visualize the generated Spectrogram and Recurrence Plot side-by-side, and receive real-time classification predictions.

\---

### 👨‍🏫 Acknowledgments

* **Reference Paper:** *"A deep learning framework for environmental sound classification by fusing linear and nonlinear features"* (Zeng, Zhou, \& Cai, *EURASIP Journal on Advances in Signal Processing*, 2026)
* **Category:** Audio Signal Processing / Acoustic Deep Learning

\---

### 📊 System Architecture \& Methodology

#### Research Methodology \& System Architecture Workflow:

!\[System Architecture](System Architecture and Methodology-Sound Recg.png)

#### Core Data Splitting \& Experimental Protocol:

* **ESC-50 Validation Protocol:** 5-fold cross-validation evaluated over 6,000 augmented audio samples (time stretching, pitch shifting, repeated sampling).
* **UrbanSound8K Validation Protocol:** Official 10-fold cross-validation across 8,732 audio slices to ensure zero sample leakage between training and testing folds.
* **Training Hyperparameters:** SGD Optimizer, Initial Learning Rate = $0.01$, Momentum = $0.9$, Weight Decay = $1\\text{e-}4$, Batch Size = $64$, Training Iterations = $100$.

