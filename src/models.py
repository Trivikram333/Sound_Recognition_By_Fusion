import torch
import torch.nn as nn
import torchvision.models as models

class AudioResNet(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        # 1. Download the pre-trained ImageNet weights
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. Hack the Input: Change 3-channel RGB to 1-channel Audio Spectrogram
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Copy the pre-trained edge-detection weights from the Red channel to our new 1-channel input
        with torch.no_grad():
            self.model.conv1.weight[:] = original_conv1.weight[:, [0], :, :]
            
        # 3. Hack the Output: Change 1000 ImageNet classes to 50 ESC-50 classes
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class AudioVGG16(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        
        # VGG's first layer is inside the 'features' block
        original_conv1 = self.model.features[0]
        self.model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        
        with torch.no_grad():
            self.model.features[0].weight[:] = original_conv1.weight[:, [0], :, :]
            self.model.features[0].bias[:] = original_conv1.bias[:]
            
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)