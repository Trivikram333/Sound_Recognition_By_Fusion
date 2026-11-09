import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def _format_inputs(mel, rp, target_size=(128, 128)):
    """Ensures mel and rp have 4D shape [Batch, 1, 128, 128] so their dimensions always match."""
    if mel.dim() == 3:
        mel = mel.unsqueeze(1)
    if rp is None:
        rp = mel
    elif rp.dim() == 3:
        rp = rp.unsqueeze(1)

    # Resize spatial dimensions to (128, 128) if either doesn't match
    if mel.shape[-2:] != target_size:
        mel = F.interpolate(mel, size=target_size, mode='bilinear', align_corners=False)
    if rp.shape[-2:] != target_size:
        rp = F.interpolate(rp, size=target_size, mode='bilinear', align_corners=False)

    return mel, rp


# =====================================================================
# 1. Baseline: AudioResNet (ResNet-18)
# =====================================================================
class AudioResNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.criterion = nn.CrossEntropyLoss()

        # Adapt first conv layer to accept 2 channels: Channel 0 (Mel), Channel 1 (RP)
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            self.model.conv1.weight[:, 0:1, :, :] = original_conv1.weight[:, [0], :, :]
            self.model.conv1.weight[:, 1:2, :, :] = original_conv1.weight[:, [1], :, :]

        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, mel, rp=None, label=None):
        mel, rp = _format_inputs(mel, rp)
        if rp is None:
            rp = mel
        elif rp.dim() == 3:
            rp = rp.unsqueeze(1)

        # Early fusion across channels: [Batch, 2, 128, 128]
        x = torch.cat([mel, rp], dim=1)
        logits = self.model(x)

        if label is not None:
            loss = self.criterion(logits, label)
            return logits, loss
        return logits


# =====================================================================
# 2. Baseline: AudioVGG16 (VGG-16)
# =====================================================================
class AudioVGG16(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.criterion = nn.CrossEntropyLoss()

        # Adapt first conv layer to accept 2 channels: Channel 0 (Mel), Channel 1 (RP)
        original_conv1 = self.model.features[0]
        self.model.features[0] = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1)

        with torch.no_grad():
            self.model.features[0].weight[:, 0:1, :, :] = original_conv1.weight[:, [0], :, :]
            self.model.features[0].weight[:, 1:2, :, :] = original_conv1.weight[:, [1], :, :]
            self.model.features[0].bias[:] = original_conv1.bias[:]

        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, mel, rp=None, label=None):
        mel, rp = _format_inputs(mel, rp)
        if rp is None:
            rp = mel
        elif rp.dim() == 3:
            rp = rp.unsqueeze(1)

        x = torch.cat([mel, rp], dim=1)
        logits = self.model(x)

        if label is not None:
            loss = self.criterion(logits, label)
            return logits, loss
        return logits


# =====================================================================
# 3. Baseline: AudioCNNGRU (CRNN)
# =====================================================================
class AudioCNNGRU(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        # 4-stage convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> [32, 64, 64]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # -> [64, 32, 32]

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),  # -> [128, 16, 32] (pool frequency only)

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1))   # -> [128, 8, 32]
        )

        # 128 channels * 8 frequency bins = 1024 feature dimension per time step
        self.gru = nn.GRU(
            input_size=1024,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Bidirectional outputs: 128 * 2 = 256
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, mel, rp=None, label=None):
        mel, rp = _format_inputs(mel, rp)

        x = torch.cat([mel, rp], dim=1)  # [B, 2, 128, 128]
        feat = self.conv_blocks(x)       # [B, 128, 8, 32]

        # Reshape to sequence: [Batch, Time (32), Feature (1024)]
        b, c, f, t = feat.shape
        feat = feat.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        gru_out, _ = self.gru(feat)      # [B, 32, 256]
        pooled = gru_out.mean(dim=1)     # Global average pooling over time -> [B, 256]
        logits = self.classifier(pooled) # [B, 50]

        if label is not None:
            loss = self.criterion(logits, label)
            return logits, loss
        return logits


# =====================================================================
# 4. Proposed Model: AudioSResNet with Dual Model Feature Fusion (DMFF)
# =====================================================================
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += res
        return self.relu(out)


class AudioSResNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        # Stream A: Spectral Feature Scanning (Log-Mel pathway)
        self.stream_mel = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),   # -> [64, 64, 64]
            ResidualBlock(64, 128, stride=2),  # -> [128, 32, 32]
            ResidualBlock(128, 256, stride=2), # -> [256, 16, 16]
            nn.AdaptiveAvgPool2d((1, 1))       # -> [256, 1, 1]
        )

        # Stream B: Dynamic Feature Scanning (Recurrence Plot pathway)
        self.stream_rp = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 64, stride=2),   # -> [64, 64, 64]
            ResidualBlock(64, 128, stride=2),  # -> [128, 32, 32]
            ResidualBlock(128, 256, stride=2), # -> [256, 16, 16]
            nn.AdaptiveAvgPool2d((1, 1))       # -> [256, 1, 1]
        )

        # Cross-Feature Enhancement (CFE) attention gates
        self.gate_mel = nn.Linear(256, 256)
        self.gate_rp = nn.Linear(256, 256)

        # Classification Head (256 + 256 = 512 dimensions)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, mel, rp=None, label=None):
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        if rp is None:
            rp = mel
        elif rp.dim() == 3:
            rp = rp.unsqueeze(1)

        # Extract independent latent features
        feat_mel = self.stream_mel(mel).flatten(1)  # [B, 256]
        feat_rp = self.stream_rp(rp).flatten(1)    # [B, 256]

        # Cross-Feature Enhancement (CFE)
        attn_mel = torch.sigmoid(self.gate_mel(feat_rp))
        attn_rp = torch.sigmoid(self.gate_rp(feat_mel))

        enhanced_mel = feat_mel * (1.0 + attn_mel)
        enhanced_rp = feat_rp * (1.0 + attn_rp)

        # Dual-stream concatenation
        fused = torch.cat([enhanced_mel, enhanced_rp], dim=1)  # [B, 512]
        logits = self.classifier(fused)                        # [B, 50]

        if label is not None:
            loss = self.criterion(logits, label)
            return logits, loss
        return logits


# =====================================================================
# Standalone Training Execution: All 4 Architectures
# =====================================================================
if __name__ == "__main__":
    import os
    import glob
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

    # 1. Dataset helper for batching .pt files
    class PTDataset(Dataset):
        def __init__(self, file_paths):
            self.file_paths = file_paths

        def __len__(self):
            return len(self.file_paths)

        def __getitem__(self, idx):
            data = torch.load(self.file_paths[idx], weights_only=True)
            return data["log_mel"], data["rp"], data["label"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing batch training on device: {device}")

    # 2. Locate preprocessed .pt files
    processed_dir = os.path.join("data", "processed")
    if not os.path.exists(processed_dir):
        processed_dir = r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\processed"

    pt_files = glob.glob(os.path.join(processed_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in '{processed_dir}'. Run preprocessing first.")

    print(f"Loaded {len(pt_files)} audio tensor files.")
    train_loader = DataLoader(PTDataset(pt_files), batch_size=32, shuffle=True)

    # 3. Define all 4 model pipelines
    models_to_train = {
        "resnet": AudioResNet(num_classes=50),
        "vgg16": AudioVGG16(num_classes=50),
        "cnn_gru": AudioCNNGRU(num_classes=50),
        "sresnet": AudioSResNet(num_classes=50)
    }

    os.makedirs("weights", exist_ok=True)
    epochs_per_model = 10

    # 4. Sequential training loop
    for model_name, model in models_to_train.items():
        print("\n" + "=" * 55)
        print(f"TRAINING MODEL: {model_name.upper()}")
        print("=" * 55)

        model = model.to(device)
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(1, epochs_per_model + 1):
            total_loss = 0.0
            correct = 0
            total = 0

            for mel_b, rp_b, label_b in train_loader:
                mel_b, rp_b, label_b = mel_b.to(device), rp_b.to(device), label_b.to(device)

                optimizer.zero_grad()
                logits, loss = model(mel_b, rp_b, label=label_b)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * label_b.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == label_b).sum().item()
                total += label_b.size(0)

            epoch_loss = total_loss / total
            acc = (correct / total) * 100.0
            print(f"[{model_name.upper()}] Epoch [{epoch:02d}/{epochs_per_model:02d}] - Loss: {epoch_loss:.4f} | Accuracy: {acc:.2f}%")

        # 5. Save individual checkpoint matching main.py naming conventions
        checkpoint_path = os.path.join("weights", f"{model_name}_best_model.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to: {checkpoint_path}")

    print("\nAll 4 model checkpoints successfully generated in /weights.")