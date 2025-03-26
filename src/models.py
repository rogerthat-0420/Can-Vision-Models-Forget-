import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet50

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.model = resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        # initializing the new fc layer properly. 
        init.kaiming_normal_(self.model.fc.weight)
        if self.model.fc.bias is not None:
            init.zeros_(self.model.fc.bias)

    def forward(self, x):
        return self.model(x)

class ViT(nn.Module):
    pass

def get_model(name, num_classes):
    if name == 'resnet50':
        return ResNet50(num_classes)
    elif name == 'vit':
        return ViT(num_classes)
    else:
        raise NotImplementedError(f"Model {name} is not implemented.")
