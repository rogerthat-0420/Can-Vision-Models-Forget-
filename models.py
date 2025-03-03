import torch.nn as nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.model(x)
